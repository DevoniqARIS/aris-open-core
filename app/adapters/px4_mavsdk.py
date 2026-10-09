from __future__ import annotations

import asyncio
import math
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Any

from .base import DroneAdapter
from ..models import Telemetry, Capture


class PX4AdapterError(RuntimeError):
    pass


async def _first(aiter, timeout: float):
    async def get_one():
        async for item in aiter:
            return item
        raise PX4AdapterError("MAVSDK telemetry stream ended unexpectedly")
    return await asyncio.wait_for(get_one(), timeout)


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1-a)))


@dataclass
class PX4MavsdkConfig:
    system_address: str = "udpin://0.0.0.0:14540"
    connection_timeout_s: float = 20.0
    health_timeout_s: float = 30.0
    command_timeout_s: float = 60.0
    arrival_radius_m: float = 2.0
    altitude_tolerance_m: float = 1.5
    telemetry_timeout_s: float = 3.0
    camera_timeout_s: float = 15.0


class PX4MavsdkDrone(DroneAdapter):
    """PX4/MAVLink adapter using the official MAVSDK Python gRPC client.

    The SDK is imported lazily so simulation users do not need MAVSDK installed.
    Tests can inject a System-compatible fake through ``system_factory``.
    ARIS task altitudes are relative to home; MAVSDK goto_location expects AMSL,
    so the adapter converts them using the MAVSDK home altitude.
    """

    def __init__(
        self,
        runtime: Path | str = Path("runtime"),
        config: PX4MavsdkConfig | None = None,
        system_factory: Callable[[], Any] | None = None,
        media_fetcher: Callable[[str, Path], Awaitable[None] | None] | None = None,
    ):
        self.cfg = config or PX4MavsdkConfig()
        self.runtime = Path(runtime)
        self.connection_timeout_s = self.cfg.connection_timeout_s
        self._system_factory = system_factory
        self._media_fetcher = media_fetcher
        self._drone = None
        self._connected = False
        self._home_abs_alt_m: float | None = None
        self._home_lat_lon: tuple[float, float] | None = None

    def _make_system(self):
        if self._system_factory:
            return self._system_factory()
        try:
            from mavsdk_grpc import System
        except ImportError as exc:
            raise PX4AdapterError(
                "PX4 support requires mavsdk-grpc. Install requirements-px4.txt"
            ) from exc
        return System()

    async def connect(self):
        if self._connected:
            return
        self._drone = self._make_system()
        await asyncio.wait_for(
            self._drone.connect(system_address=self.cfg.system_address),
            self.cfg.connection_timeout_s,
        )
        state = await _first(
            self._drone.core.connection_state(), self.cfg.connection_timeout_s
        )
        if not getattr(state, "is_connected", False):
            # Some backends emit disconnected first; continue until connected.
            async def wait_connected():
                async for item in self._drone.core.connection_state():
                    if getattr(item, "is_connected", False):
                        return item
            await asyncio.wait_for(wait_connected(), self.cfg.connection_timeout_s)

        async def wait_health():
            async for health in self._drone.telemetry.health():
                if (
                    getattr(health, "is_global_position_ok", False)
                    and getattr(health, "is_home_position_ok", False)
                ):
                    return health
        await asyncio.wait_for(wait_health(), self.cfg.health_timeout_s)

        home = await _first(self._drone.telemetry.home(), self.cfg.telemetry_timeout_s)
        self._home_abs_alt_m = float(home.absolute_altitude_m)
        self._home_lat_lon = (float(home.latitude_deg), float(home.longitude_deg))
        self._connected = True

    def _require(self):
        if not self._connected or self._drone is None:
            raise PX4AdapterError("PX4 adapter is not connected")
        return self._drone

    async def telemetry(self):
        d = self._require()
        pos, battery, health, in_air = await asyncio.gather(
            _first(d.telemetry.position(), self.cfg.telemetry_timeout_s),
            _first(d.telemetry.battery(), self.cfg.telemetry_timeout_s),
            _first(d.telemetry.health(), self.cfg.telemetry_timeout_s),
            _first(d.telemetry.in_air(), self.cfg.telemetry_timeout_s),
        )
        rem = float(getattr(battery, "remaining_percent", 1.0))
        if math.isnan(rem) or rem < 0:
            rem = 0.0
        battery_pct = rem * 100.0 if rem <= 1.0 else rem
        return Telemetry(
            lat=float(pos.latitude_deg),
            lon=float(pos.longitude_deg),
            altitude_m=max(0.0, float(pos.relative_altitude_m)),
            battery_percent=max(0.0, min(100.0, battery_pct)),
            gps_ok=bool(
                getattr(health, "is_global_position_ok", False)
                and getattr(health, "is_home_position_ok", False)
            ),
            link_ok=self._connected,
            airborne=bool(in_air),
            observed_at=datetime.now(timezone.utc),
        )

    async def _wait_altitude(self, altitude_m: float, require_airborne: bool = True):
        deadline = asyncio.get_running_loop().time() + self.cfg.command_timeout_s
        while True:
            t = await self.telemetry()
            if abs(t.altitude_m - altitude_m) <= self.cfg.altitude_tolerance_m:
                if not require_airborne or t.airborne:
                    return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"PX4 did not reach altitude {altitude_m:.1f} m before timeout"
                )
            await asyncio.sleep(0.25)

    async def takeoff(self, altitude_m):
        d = self._require()
        await asyncio.wait_for(d.action.set_takeoff_altitude(float(altitude_m)), 5)
        await asyncio.wait_for(d.action.arm(), 10)
        await asyncio.wait_for(d.action.takeoff(), 10)
        await self._wait_altitude(float(altitude_m), require_airborne=True)

    async def go_to(self, task, speed_mps):
        d = self._require()
        if self._home_abs_alt_m is None:
            raise PX4AdapterError("PX4 home altitude is unavailable")
        try:
            await asyncio.wait_for(d.action.set_current_speed(float(speed_mps)), 5)
        except Exception:
            # Some autopilots reject speed changes in certain modes. The position
            # command itself is still valid, so do not silently abort navigation.
            pass
        absolute_alt_m = self._home_abs_alt_m + float(task.altitude_m)
        await asyncio.wait_for(
            d.action.goto_location(
                float(task.lat),
                float(task.lon),
                float(absolute_alt_m),
                float(task.heading_deg),
            ),
            10,
        )

        deadline = asyncio.get_running_loop().time() + self.cfg.command_timeout_s
        while True:
            t = await self.telemetry()
            horizontal = _distance_m(t.lat, t.lon, float(task.lat), float(task.lon))
            vertical = abs(t.altitude_m - float(task.altitude_m))
            if horizontal <= self.cfg.arrival_radius_m and vertical <= self.cfg.altitude_tolerance_m:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"PX4 did not reach task {getattr(task, 'id', '?')}; "
                    f"{horizontal:.1f} m horizontal / {vertical:.1f} m vertical remain"
                )
            await asyncio.sleep(0.25)

    async def hold(self):
        d = self._require()
        await asyncio.wait_for(d.action.hold(), 10)

    async def return_home(self):
        d = self._require()
        if self._home_lat_lon is None:
            raise PX4AdapterError("PX4 home position is unavailable")
        await asyncio.wait_for(d.action.return_to_launch(), 10)
        deadline = asyncio.get_running_loop().time() + self.cfg.command_timeout_s
        while True:
            t = await self.telemetry()
            # PX4 RTL may land automatically depending on vehicle configuration.
            if not t.airborne:
                return
            horizontal = _distance_m(
                t.lat, t.lon, self._home_lat_lon[0], self._home_lat_lon[1]
            )
            if horizontal <= self.cfg.arrival_radius_m:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"PX4 RTL did not reach home region; {horizontal:.1f} m remain"
                )
            await asyncio.sleep(0.25)

    async def land(self):
        d = self._require()
        t = await self.telemetry()
        if not t.airborne:
            return
        await asyncio.wait_for(d.action.land(), 10)
        deadline = asyncio.get_running_loop().time() + self.cfg.command_timeout_s
        while True:
            if not (await self.telemetry()).airborne:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("PX4 landing was not confirmed before timeout")
            await asyncio.sleep(0.25)

    async def abort(self):
        d = self._require()
        try:
            await asyncio.wait_for(d.action.return_to_launch(), 10)
        except Exception:
            await asyncio.wait_for(d.action.land(), 10)
        deadline = asyncio.get_running_loop().time() + self.cfg.command_timeout_s
        while True:
            try:
                if not (await self.telemetry()).airborne:
                    return
            except Exception:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("PX4 recovery/landing could not be confirmed")
            await asyncio.sleep(0.25)

    async def _download_media(self, url: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self._media_fetcher is not None:
            result = self._media_fetcher(url, destination)
            if asyncio.iscoroutine(result):
                await result
            return

        def fetch():
            with urllib.request.urlopen(url, timeout=self.cfg.camera_timeout_s) as src:
                with destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        await asyncio.to_thread(fetch)

    async def capture_image(self, mission_id, task, attempt):
        d = self._require()
        if not hasattr(d, "camera"):
            raise PX4AdapterError("Connected MAVSDK system has no camera plugin")

        # Subscribe before triggering so the resulting CaptureInfo is not missed.
        capture_stream = getattr(d.camera, "capture_info", None)
        if capture_stream is None:
            raise PX4AdapterError(
                "MAVSDK camera does not expose capture_info; cannot obtain evidence file"
            )

        async def next_capture():
            async for info in capture_stream():
                if getattr(info, "is_success", True):
                    return info
        waiter = asyncio.create_task(next_capture())
        try:
            await asyncio.wait_for(d.camera.take_photo(), 10)
            info = await asyncio.wait_for(waiter, self.cfg.camera_timeout_s)
        finally:
            if not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

        url = str(getattr(info, "file_url", "") or "")
        if not url:
            raise PX4AdapterError(
                "Photo was triggered, but MAVSDK did not report a downloadable file URL"
            )
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"
        out = self.runtime / "images" / mission_id
        path = out / f"{task.id}-{attempt}{suffix}"
        await self._download_media(url, path)
        if not path.exists() or path.stat().st_size == 0:
            raise PX4AdapterError("Camera evidence download produced an empty file")
        return Capture(
            image_path=str(path.resolve()),
            source="px4",
            captured_at=datetime.now(timezone.utc),
            telemetry=await self.telemetry(),
        )

    async def close(self):
        self._connected = False
        # mavsdk-grpc System owns its server process and cleans it up on interpreter
        # exit. Avoid reaching into private SDK methods here.
