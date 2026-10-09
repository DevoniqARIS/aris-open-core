import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.adapters.px4_mavsdk import PX4MavsdkDrone, PX4MavsdkConfig
from app.models import ObservationTask, MissionPlanRequest
from app.adapters.capabilities import capability_profile


async def stream(items):
    for item in items:
        yield item


class Core:
    def connection_state(self):
        return stream([SimpleNamespace(is_connected=True)])


class TelemetryPlugin:
    def __init__(self):
        self.lat = 47.397742
        self.lon = 8.545594
        self.rel = 0.0
        self.airborne = False
        self.home_abs = 488.0
        self.battery_value = 0.82

    def health(self):
        return stream([SimpleNamespace(is_global_position_ok=True, is_home_position_ok=True)])
    def home(self):
        return stream([SimpleNamespace(latitude_deg=self.lat, longitude_deg=self.lon, absolute_altitude_m=self.home_abs)])
    def position(self):
        return stream([SimpleNamespace(latitude_deg=self.lat, longitude_deg=self.lon, relative_altitude_m=self.rel, absolute_altitude_m=self.home_abs+self.rel)])
    def battery(self):
        return stream([SimpleNamespace(remaining_percent=self.battery_value)])
    def in_air(self):
        return stream([self.airborne])


class Action:
    def __init__(self, telemetry):
        self.t = telemetry
        self.calls = []
    async def set_takeoff_altitude(self, a): self.calls.append(("set_takeoff_altitude", a))
    async def arm(self): self.calls.append(("arm",))
    async def takeoff(self):
        self.calls.append(("takeoff",)); self.t.airborne=True; self.t.rel=5.0
    async def set_current_speed(self, s): self.calls.append(("speed", s))
    async def goto_location(self, lat, lon, abs_alt, yaw):
        self.calls.append(("goto", lat, lon, abs_alt, yaw)); self.t.lat=lat; self.t.lon=lon; self.t.rel=abs_alt-self.t.home_abs
    async def hold(self): self.calls.append(("hold",))
    async def return_to_launch(self): self.calls.append(("rtl",)); self.t.lat=47.397742; self.t.lon=8.545594
    async def land(self): self.calls.append(("land",)); self.t.airborne=False; self.t.rel=0.0


class Camera:
    def __init__(self): self.calls=[]
    async def take_photo(self): self.calls.append("photo")
    def capture_info(self):
        return stream([SimpleNamespace(is_success=True, file_url="memory://capture.jpg")])


class FakeSystem:
    def __init__(self):
        self.core=Core(); self.telemetry=TelemetryPlugin(); self.action=Action(self.telemetry); self.camera=Camera(); self.address=None
    async def connect(self, system_address): self.address=system_address


def make_drone(tmp_path):
    fake=FakeSystem()
    async def fetch(url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg-bytes")
    d=PX4MavsdkDrone(runtime=tmp_path, config=PX4MavsdkConfig(command_timeout_s=2), system_factory=lambda: fake, media_fetcher=fetch)
    return d, fake


def task(alt=5.0):
    return ObservationTask(id="t1", requirement_id="r1", lat=47.397800, lon=8.545650, altitude_m=alt, heading_deg=90, label="north_facade")


def test_models_and_capabilities_advertise_px4():
    req=MissionPlanRequest(asset_id="a", prompt="inspect", adapter="px4")
    assert req.adapter == "px4"
    c=capability_profile("px4")
    assert c.vendor == "PX4/MAVLink"
    assert c.validation == "software_only"


def test_connect_and_telemetry(tmp_path):
    d,f=make_drone(tmp_path)
    asyncio.run(d.connect())
    t=asyncio.run(d.telemetry())
    assert f.address == "udpin://0.0.0.0:14540"
    assert t.gps_ok and t.link_ok
    assert t.battery_percent == 82.0


def test_takeoff_and_goto_converts_relative_altitude_to_amsl(tmp_path):
    d,f=make_drone(tmp_path)
    asyncio.run(d.connect())
    asyncio.run(d.takeoff(5.0))
    asyncio.run(d.go_to(task(7.0), 2.5))
    goto=[c for c in f.action.calls if c[0]=="goto"][-1]
    assert abs(goto[3] - 495.0) < 1e-6
    assert goto[4] == 90.0
    t=asyncio.run(d.telemetry())
    assert abs(t.altitude_m-7.0) < 1e-6


def test_hold_rtl_land_and_abort_paths(tmp_path):
    d,f=make_drone(tmp_path)
    asyncio.run(d.connect())
    asyncio.run(d.takeoff(5.0))
    asyncio.run(d.hold())
    asyncio.run(d.return_home())
    asyncio.run(d.land())
    assert ("hold",) in f.action.calls and ("rtl",) in f.action.calls and ("land",) in f.action.calls
    assert asyncio.run(d.telemetry()).airborne is False


def test_camera_capture_downloads_evidence(tmp_path):
    d,f=make_drone(tmp_path)
    asyncio.run(d.connect())
    cap=asyncio.run(d.capture_image("m1", task(), 1))
    assert cap.source == "px4"
    assert Path(cap.image_path).exists()
    assert Path(cap.image_path).read_bytes() == b"fake-jpeg-bytes"
    assert f.camera.calls == ["photo"]


def test_px4_is_treated_as_live_adapter_for_lock_retention():
    from pathlib import Path
    text=(Path(__file__).resolve().parents[1]/"app"/"executor.py").read_text()
    assert 'plan.adapter not in {"dji", "parrot", "px4"}' in text
