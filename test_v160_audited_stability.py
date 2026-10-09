import asyncio
from pathlib import Path

from app.adaptive import AdaptiveMissionExecutor
from app.maneuver import MissionAction, NaturalLanguageMission, _normalize_intent
from app.adapters.capabilities import capability_profile

ROOT = Path(__file__).resolve().parents[1]


def null_action(kind, **kw):
    base = dict(
        kind=kind, altitude_m=None, speed_mps=None, duration_s=None, distance_m=None,
        direction=None, heading_deg=None, turn_degrees=None, target=None,
        stop_distance_m=None, radius_m=None, laps=None, orbit_direction=None,
        corners=None, corner_strategy=None, inspection_targets=None, coverage_mode=None,
        coverage_strategy=None, evidence_requirement=None, standoff_m=None,
        object_class=None, object_selection=None, discovered_inspection_style=None,
        altitude_policy=None, repeat=1,
    )
    base.update(kw)
    return MissionAction(**base)


class FakeWorld:
    def __init__(self):
        self.position = [-12.0, -12.0, .24]
        self.collisions = []

    def state(self):
        return {
            "position": list(self.position),
            "velocity": [0, 0, 0],
            "collisions": list(self.collisions),
        }


class FakeTelemetry:
    def __init__(self, airborne):
        self.airborne = airborne


class FakeDrone:
    def __init__(self):
        self.world = FakeWorld()
        self.airborne = False
        self.obj = {
            "id": "building_dynamic",
            "class": "building",
            "position": [0, 0, 2],
            "size": [8, 6, 4],
            "top_z": 4,
            "range_m": 10,
        }

    def capabilities(self):
        return capability_profile("physics")

    async def connect(self):
        pass

    async def telemetry(self):
        return FakeTelemetry(self.airborne)

    async def takeoff(self, altitude_m):
        self.airborne = True
        self.world.position = [-12, -12, altitude_m + .24]

    async def go_to_local(self, x, y, z, heading, speed, label):
        self.world.position = [x, y, z + .24]

    async def sense_obstacles(self, range_m=10, altitude_m=0):
        return []

    async def sense_objects(self, range_m=14):
        return [self.obj]

    async def object_visible(self, object_id):
        return object_id == self.obj["id"]

    async def orient_camera_local(self, *args, **kwargs):
        pass

    async def capture_discovered(self, mission_id, object_id, label, attempt):
        return {
            "image_path": f"/tmp/{object_id}-{attempt}.png",
            "sha256": "x",
        }

    async def hold(self):
        pass

    async def return_home(self):
        self.world.position = [-12, -12, self.world.position[2]]

    async def land(self):
        self.world.position[2] = .24
        self.airborne = False

    async def abort(self):
        self.world.position[2] = .24
        self.airborne = False

    async def close(self):
        pass


def test_adaptive_normalizer_does_not_invent_asset_center_for_bare_orbit():
    raw = {
        "disposition": "execute",
        "summary": "x",
        "clarification_question": None,
        "assumptions": [],
        "actions": [{
            "kind": "orbit",
            "target": None,
            "radius_m": 6,
            "laps": 1,
            "orbit_direction": "clockwise",
        }],
    }
    normalized = _normalize_intent(raw, adaptive_inspection=True)
    orbit = next(a for a in normalized["actions"] if a["kind"] == "orbit")
    assert orbit["target"] is None


def test_adaptive_prompt_has_no_exact_demo_command_magic_phrase():
    text = (ROOT / "app" / "maneuver.py").read_text(encoding="utf-8")
    exact = (
        "Takeoff to 12m. Find a building. Fly to it. Avoid obstacles. "
        "You are free to adjust altitude in order to avoid collisions. "
        "Circle the building. Land at home."
    )
    assert exact not in text


def test_backend_hot_planning_is_offloaded_from_event_loop():
    text = (ROOT / "app" / "adaptive.py").read_text(encoding="utf-8")
    assert "await asyncio.to_thread(\n                        self.planner3d.plan" in text
    assert "goal=await asyncio.to_thread(" in text


def test_state_polling_no_longer_calls_bullet_natively():
    text = (ROOT / "simulation" / "world.py").read_text(encoding="utf-8")
    start = text.index("    def state(self):")
    end = text.index("\n    def telemetry", start)
    block = text[start:end]
    assert "getBasePositionAndOrientation" not in block
    assert "_state_cache" in block


def test_backend_error_logs_are_exportable():
    text = (ROOT / "simulation" / "server.py").read_text(encoding="utf-8")
    assert '".txt"' in text and '".log"' in text
    assert "backend_error.txt" in text


def test_exact_semantic_mission_completes_in_fake_world():
    mission = NaturalLanguageMission(
        disposition="execute",
        summary="discover building, orbit, return",
        clarification_question=None,
        assumptions=[],
        actions=[
            null_action("takeoff", altitude_m=12),
            null_action(
                "find_and_inspect",
                altitude_m=12,
                speed_mps=2,
                standoff_m=7.5,
                object_class="building",
                object_selection="nearest",
                discovered_inspection_style="orbit",
                evidence_requirement="clear_photo",
                altitude_policy="adaptive",
            ),
            null_action("return_home", speed_mps=2),
            null_action("land"),
        ],
    )
    events = []
    ex = AdaptiveMissionExecutor(
        lambda kind, **fields: events.append((kind, fields)),
        memory=None,
    )
    out = asyncio.run(
        ex.run(mission, FakeDrone(), "test-run", "adaptive_complex")
    )
    assert out["status"] == "COMPLETE"
    assert out["landed_confirmed"] is True
    assert any(e[0] == "object_discovered" for e in events)
    assert any(e[0] == "navigation_goal_reached" for e in events)
