"""Test doubles only. No measured perception accuracy is implied by these fixtures."""

from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
from app.models import Asset, LatLon, MissionPlanRequest, Capture
from app.store import Store
from app.mission_interpreter import DeterministicMissionInterpreter
from app.planner import InspectionPlanner
from app.executor import MissionExecutor
from app.safety import SafetyGate, SafetyConfig
from app.quality import QualityInspector, read_image
from app.perception import Assessment
from app.security import HumanPresenceDetector
from app.completion_engine import CompletionEngine
from app.evidence_verifier import EvidenceVerifier
from app.gap_replanner import GapReplanner
from app.adapters.sim import SimDrone


def asset():
    return Asset(
        id="asset",
        name="Fixture",
        height_m=10,
        boundary=[
            LatLon(lat=49, lon=25),
            LatLon(lat=49, lon=25.0002),
            LatLon(lat=49.0002, lon=25.0002),
            LatLon(lat=49.0002, lon=25),
        ],
        flight_boundary=[
            LatLon(lat=48.9997, lon=24.9996),
            LatLon(lat=48.9997, lon=25.0006),
            LatLon(lat=49.0005, lon=25.0006),
            LatLon(lat=49.0005, lon=24.9996),
        ],
    )


def texture(path, seed=12):
    rng = np.random.default_rng(seed)
    a = rng.integers(45, 210, (480, 640), dtype=np.uint8)
    img = Image.fromarray(a).convert("RGB")
    d = ImageDraw.Draw(img)
    for _ in range(80):
        x, y = rng.integers([0, 0], [610, 450])
        d.rectangle((int(x), int(y), int(x) + 20, int(y) + 20), fill=(50, 50, 50))
    img.save(path)
    return path


class FixturePerception:
    """Deliberate test double to isolate orchestration from vision accuracy."""

    def __init__(self, visible=True, presence="absent", issues=None):
        self.visible = visible
        self.presence = presence
        self.issues = issues or []

    async def assess(self, path, requirement, asset):
        _, h = read_image(path)
        return Assessment(
            status="assessed",
            provider="TEST_DOUBLE",
            image_sha256=h,
            target_visible=self.visible,
            criterion_met=self.visible,
            presence=self.presence,
            issues=self.issues,
            explanation="TEST DOUBLE, not a model assessment",
        )


class RecordedFixtureDrone(SimDrone):
    async def capture_image(self, mission_id, task, attempt):
        directory = self.runtime / "images" / mission_id
        directory.mkdir(parents=True, exist_ok=True)
        p = texture(directory / (task.id + ".png"))
        return Capture(
            image_path=str(p),
            source="recorded",
            captured_at=datetime.now(timezone.utc),
            telemetry=await self.telemetry(),
        )


class LowQualityFirstDrone(RecordedFixtureDrone):
    async def capture_image(self, mission_id, task, attempt):
        capture = await super().capture_image(mission_id, task, attempt)
        if attempt == 1:
            Image.new("RGB", (640, 480), "#808080").save(capture.image_path)
        return capture


def setup(
    tmp, perception=None, config=None, prompt="Inspect exterior and roof", mode="replay"
):
    store = Store(tmp / "db.sqlite")
    a = asset()
    store.save_asset(a)
    spec = DeterministicMissionInterpreter().interpret(a, prompt)
    req = MissionPlanRequest(asset_id=a.id, prompt=prompt, adapter=mode, use_gpt=False)
    p = InspectionPlanner().plan_from_spec(a, req, spec)
    store.save_mission(p)
    ex = MissionExecutor(
        store,
        SafetyGate(config or SafetyConfig(watchdog_interval_s=0.01)),
        QualityInspector(),
        HumanPresenceDetector(),
        CompletionEngine(EvidenceVerifier()),
        GapReplanner(),
        perception or FixturePerception(),
    )
    return store, a, p, ex
