import asyncio
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw
from .base import DroneAdapter
from ..models import Telemetry, Capture


class SimDrone(DroneAdapter):
    """In-memory kinematic stub, not DJI SDK or physics simulation."""

    def __init__(self, home=None, runtime=Path("runtime"), delay=0.01):
        lat, lon = home or (49.00035, 25.0001)
        self.state = Telemetry(
            lat=lat,
            lon=lon,
            altitude_m=0,
            battery_percent=100,
            gps_ok=True,
            link_ok=True,
            airborne=False,
            observed_at=datetime.now(timezone.utc),
        )
        self.home = (lat, lon)
        self.runtime = Path(runtime)
        self.delay = delay

    async def connect(self):
        await asyncio.sleep(self.delay)

    async def telemetry(self):
        return self.state.model_copy(update={"observed_at": datetime.now(timezone.utc)})

    async def takeoff(self, altitude_m):
        self.state.airborne = True
        self.state.altitude_m = altitude_m
        self.state.battery_percent -= 1
        await asyncio.sleep(self.delay)

    async def go_to(self, task, speed_mps):
        self.state.lat = task.lat
        self.state.lon = task.lon
        self.state.altitude_m = task.altitude_m
        self.state.battery_percent = max(0, self.state.battery_percent - 2)
        await asyncio.sleep(self.delay)

    async def capture_image(self, mission_id, task, attempt):
        out = self.runtime / "images" / mission_id
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{task.id}.jpg"
        img = Image.new("RGB", (640, 480), "#809080")
        d = ImageDraw.Draw(img)
        d.rectangle((80, 80, 550, 400), outline="black", width=5)
        d.text(
            (90, 200), "SYNTHETIC TEST IMAGE - NOT INSPECTION EVIDENCE", fill="black"
        )
        img.save(p)
        return Capture(
            image_path=str(p.resolve()),
            source="synthetic",
            captured_at=datetime.now(timezone.utc),
            telemetry=await self.telemetry(),
        )

    async def hold(self):
        await asyncio.sleep(self.delay)

    async def return_home(self):
        self.state.lat, self.state.lon = self.home
        await asyncio.sleep(self.delay)

    async def land(self):
        self.state.airborne = False
        self.state.altitude_m = 0

    async def abort(self):
        await self.return_home()
        await self.land()
