import asyncio
import math
import pytest
from datetime import datetime, timezone

from app.maneuver import (
    NaturalLanguageMission,
    MissionAction,
    ManeuverCompiler,
    ManeuverSafetyGate,
    GPTManeuverInterpreter,
    ClarificationRequired,
)
from app.models import Asset, LatLon
from app.geo import offset_m


ORIGIN = LatLon(lat=49.0, lon=25.0)
HOME = (-12.0, -12.0)


def geo(x, y):
    return offset_m(ORIGIN, y, x)


def scene_asset():
    return Asset(
        id="simulation-building",
        name="Calibration building",
        height_m=4,
        boundary=[geo(-4,-3), geo(4,-3), geo(4,3), geo(-4,3)],
        flight_boundary=[geo(-24,-24), geo(24,-24), geo(24,24), geo(-24,24)],
        exclusion_zones=[],
        alternate_views={},
    )


def action(kind, **kwargs):
    base = dict(
        kind=kind, altitude_m=None, speed_mps=None, duration_s=None,
        distance_m=None, direction=None, heading_deg=None, turn_degrees=None,
        target=None, stop_distance_m=None, radius_m=None, laps=None,
        orbit_direction=None, repeat=1,
    )
    base.update(kwargs)
    return MissionAction(**base)


def mission(actions, assumptions=None):
    return NaturalLanguageMission(
        disposition="execute",
        summary="test mission",
        clarification_question=None,
        assumptions=assumptions or [],
        actions=actions,
    )


def test_free_form_semantic_sequence_compiles_from_values_not_phrases():
    m = mission([
        action("takeoff", altitude_m=8),
        action("move", distance_m=10, direction="forward", speed_mps=2),
        action("turn", turn_degrees=-90),
        action("move", duration_s=4, direction="forward", speed_mps=3),
        action("return_home", speed_mps=2),
        action("land"),
    ])
    p = ManeuverCompiler(home_xy=HOME, initial_heading_deg=90).compile(
        "Go up, head out ten metres, hang a left, keep going for four seconds, then come home.",
        m,
    )
    straights = [c for c in p.commands if c.kind == "straight"]
    assert len(straights) == 2
    assert straights[0].distance_m == pytest.approx(10)
    assert straights[0].heading_deg == pytest.approx(90)
    assert straights[1].distance_m == pytest.approx(12)
    assert straights[1].heading_deg == pytest.approx(0)
    assert p.commands[-1].kind == "land"


def test_same_action_type_different_numbers_produces_different_path():
    c = ManeuverCompiler(home_xy=HOME, initial_heading_deg=90)
    a = mission([action("takeoff", altitude_m=8), action("move", distance_m=5, direction="forward", speed_mps=2), action("return_home"), action("land")])
    b = mission([action("takeoff", altitude_m=8), action("move", distance_m=11, direction="forward", speed_mps=2), action("return_home"), action("land")])
    pa = c.compile("short", a)
    pb = c.compile("long", b)
    sa = next(x for x in pa.commands if x.kind == "straight")
    sb = next(x for x in pb.commands if x.kind == "straight")
    assert sa.x_m != sb.x_m
    assert sa.distance_m == 5
    assert sb.distance_m == 11


def test_repeat_is_compiled_dynamically():
    m = mission([
        action("takeoff", altitude_m=8),
        action("turn", turn_degrees=45, repeat=3),
        action("return_home"),
        action("land"),
    ])
    p = ManeuverCompiler(home_xy=HOME, initial_heading_deg=90).compile("turn right three times", m)
    turns = [c for c in p.commands if c.kind == "turn"]
    assert [t.heading_deg for t in turns] == pytest.approx([135, 180, 225])


def test_named_semantic_target_is_resolved_by_aris_not_gpt_coordinates():
    m = mission([
        action("takeoff", altitude_m=8),
        action("go_to", target="asset_north", speed_mps=2),
        action("return_home"),
        action("land"),
    ])
    p = ManeuverCompiler(home_xy=HOME).compile("Go to the north side of the building and come back.", m)
    goto = next(c for c in p.commands if c.kind == "goto")
    assert goto.x_m == pytest.approx(0)
    assert goto.y_m == pytest.approx(8)


def test_safety_gate_blocks_out_of_bounds_semantic_mission():
    m = mission([
        action("takeoff", altitude_m=8),
        action("move", distance_m=40, direction="west", speed_mps=2),
        action("return_home"),
        action("land"),
    ])
    p = ManeuverCompiler(home_xy=HOME).compile("go west forty metres", m)
    with pytest.raises(ValueError, match="flight boundary"):
        ManeuverSafetyGate(geo, home_xy=HOME).validate(scene_asset(), p)


class FakeParaphraseModel:
    async def generate_with_meta(self, instructions, content, schema, schema_name="x"):
        return {
            "disposition": "execute",
            "summary": "Take off, move forward briefly, turn left, then return.",
            "clarification_question": None,
            "assumptions": ["'briefly' interpreted as 3 seconds", "speed interpreted as 2 m/s"],
            "actions": [
                {
                    "kind":"takeoff","altitude_m":8,"speed_mps":None,"duration_s":None,
                    "distance_m":None,"direction":None,"heading_deg":None,"turn_degrees":None,
                    "target":None,"stop_distance_m":None,"radius_m":None,"laps":None,
                    "orbit_direction":None,"repeat":1
                },
                {
                    "kind":"move","altitude_m":None,"speed_mps":2,"duration_s":3,
                    "distance_m":None,"direction":"forward","heading_deg":None,"turn_degrees":None,
                    "target":None,"stop_distance_m":None,"radius_m":None,"laps":None,
                    "orbit_direction":None,"repeat":1
                },
                {
                    "kind":"turn","altitude_m":None,"speed_mps":None,"duration_s":None,
                    "distance_m":None,"direction":None,"heading_deg":None,"turn_degrees":-90,
                    "target":None,"stop_distance_m":None,"radius_m":None,"laps":None,
                    "orbit_direction":None,"repeat":1
                },
                {
                    "kind":"return_home","altitude_m":None,"speed_mps":2,"duration_s":None,
                    "distance_m":None,"direction":None,"heading_deg":None,"turn_degrees":None,
                    "target":None,"stop_distance_m":None,"radius_m":None,"laps":None,
                    "orbit_direction":None,"repeat":1
                },
                {
                    "kind":"land","altitude_m":None,"speed_mps":None,"duration_s":None,
                    "distance_m":None,"direction":None,"heading_deg":None,"turn_degrees":None,
                    "target":None,"stop_distance_m":None,"radius_m":None,"laps":None,
                    "orbit_direction":None,"repeat":1
                }
            ]
        }, {"response_id":"resp_paraphrase","model":"fake","status":"completed"}


def test_interpreter_accepts_conversational_paraphrase_without_magic_phrase():
    intent, meta = asyncio.run(
        GPTManeuverInterpreter(client=FakeParaphraseModel()).interpret(
            "Go up, scoot forward for a bit, hang a left, and bring it back."
        )
    )
    assert intent.disposition == "execute"
    assert [a.kind for a in intent.actions] == ["takeoff","move","turn","return_home","land"]
    assert intent.actions[1].duration_s == 3
    assert meta["response_id"] == "resp_paraphrase"


class FakeClarifyModel:
    async def generate_with_meta(self, instructions, content, schema, schema_name="x"):
        return {
            "disposition": "clarify",
            "summary": "The destination is not identifiable from available context.",
            "clarification_question": "Which location do you mean by 'over there'?",
            "assumptions": [],
            "actions": []
        }, {"response_id":"resp_clarify","model":"fake","status":"completed"}


def test_unresolvable_reference_requests_clarification_not_fake_coordinate():
    with pytest.raises(ClarificationRequired, match="Which location"):
        asyncio.run(
            GPTManeuverInterpreter(client=FakeClarifyModel()).interpret(
                "Take off and fly over there."
            )
        )
