import asyncio
import math
from app.maneuver import GPTManeuverInterpreter
from app.adaptive import AdaptiveMissionExecutor

def action(kind, **kw):
    d = {
        "kind": kind, "altitude_m": None, "speed_mps": None, "duration_s": None,
        "distance_m": None, "direction": None, "heading_deg": None, "turn_degrees": None,
        "target": None, "stop_distance_m": None, "radius_m": None, "laps": None,
        "orbit_direction": None, "corners": None, "corner_strategy": None,
        "evidence_requirement": None, "standoff_m": None, "repeat": 1,
    }
    d.update(kw)
    return d

class FakeModel:
    async def generate_with_meta(self, instructions, content, schema, schema_name="x"):
        assert "inspect the building from all directions" in instructions.lower()
        return {
            "disposition":"execute",
            "summary":"Inspect the building from all directions at 5 m, circle it, return home and land.",
            "clarification_question":None,
            "assumptions":[],
            "actions":[
                action("takeoff", altitude_m=5),
                action("inspect_corners", altitude_m=5, speed_mps=2,
                       corners=["corner_ne","corner_se","corner_sw","corner_nw"],
                       corner_strategy="nearest_safe", evidence_requirement="clear_photo",
                       standoff_m=7.5),
                action("orbit", altitude_m=5, speed_mps=2, target="asset_center",
                       radius_m=6, laps=1, orbit_direction="clockwise"),
                action("return_home", speed_mps=2),
                action("land")
            ]
        }, {"response_id":"resp","model":"fake","status":"completed"}

def test_exact_user_request_has_evidence_and_orbit():
    intent, _ = asyncio.run(GPTManeuverInterpreter(client=FakeModel()).interpret(
        "Inspect the building from all directions at 5m altitude. Circle the building for full inspection. "
        "When finished, return to your starting position and land.",
        adaptive_inspection=True
    ))
    kinds = [a.kind for a in intent.actions]
    assert "inspect_corners" in kinds
    assert "orbit" in kinds
    ins = next(a for a in intent.actions if a.kind == "inspect_corners")
    assert len(ins.corners) == 4
    assert ins.altitude_m == 5

def test_orbit_candidates_expand_beyond_default_radius():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    candidates = ex._orbit_goal_candidates((0.0,0.0), 0.0, 6.0)
    radii = [r for _,_,r in candidates]
    assert radii[0] >= 7.5
    assert radii[-1] > radii[0]

def test_north_point_has_farther_alternative():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    candidates = ex._orbit_goal_candidates((0.0,0.0), math.pi/2, 6.0)
    assert any(y > 10.0 for x,y,r in candidates)
