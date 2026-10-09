import asyncio
from pathlib import Path
import pytest

from app.maneuver import GPTManeuverInterpreter, NaturalLanguageMission
from app.adaptive import AdaptiveMissionExecutor


def _action(kind, **kw):
    fields = {
        "kind": kind,
        "altitude_m": None,
        "speed_mps": None,
        "duration_s": None,
        "distance_m": None,
        "direction": None,
        "heading_deg": None,
        "turn_degrees": None,
        "target": None,
        "stop_distance_m": None,
        "radius_m": None,
        "laps": None,
        "orbit_direction": None,
        "corners": None,
        "corner_strategy": None,
        "evidence_requirement": None,
        "standoff_m": None,
        "repeat": 1,
    }
    fields.update(kw)
    return fields


class FakeUnifiedModel:
    def __init__(self, response):
        self.response = response

    async def generate_with_meta(self, instructions, content, schema, schema_name="x"):
        assert schema_name == "aris_unified_mission_ir"
        assert "ADAPTIVE ACTION" in instructions
        return self.response, {"response_id": "resp_v12", "model": "fake", "status": "completed"}


def test_adaptive_interpreter_preserves_user_sequence_and_corner_order():
    raw = {
        "disposition": "execute",
        "summary": "Start northwest, inspect clockwise, hover east, return.",
        "clarification_question": None,
        "assumptions": [],
        "actions": [
            _action("takeoff", altitude_m=10),
            _action(
                "inspect_corners",
                altitude_m=10,
                speed_mps=2,
                corners=["corner_nw", "corner_ne", "corner_se", "corner_sw"],
                corner_strategy="given_order",
                evidence_requirement="clear_photo",
                standoff_m=8,
            ),
            _action("go_to", target="asset_east", speed_mps=2),
            _action("hover", duration_s=3),
            _action("return_home", speed_mps=2),
            _action("land"),
        ],
    }
    intent, _ = asyncio.run(
        GPTManeuverInterpreter(client=FakeUnifiedModel(raw)).interpret(
            "Start at the northwest corner, inspect all four corners clockwise, "
            "hover for 3 seconds on the east side, then return home and land.",
            adaptive_inspection=True,
        )
    )
    assert [a.kind for a in intent.actions] == [
        "takeoff", "inspect_corners", "go_to", "hover", "return_home", "land"
    ]
    assert intent.actions[1].corners == [
        "corner_nw", "corner_ne", "corner_se", "corner_sw"
    ]
    assert intent.actions[1].corner_strategy == "given_order"


def test_different_language_intents_create_different_semantic_missions():
    a = NaturalLanguageMission.model_validate({
        "disposition": "execute",
        "summary": "North only",
        "clarification_question": None,
        "assumptions": [],
        "actions": [
            _action("takeoff", altitude_m=8),
            _action("inspect_corners", altitude_m=8, speed_mps=2,
                    corners=["corner_nw", "corner_ne"],
                    corner_strategy="nearest_safe",
                    evidence_requirement="clear_photo", standoff_m=7),
            _action("return_home", speed_mps=2),
            _action("land"),
        ],
    })
    b = NaturalLanguageMission.model_validate({
        "disposition": "execute",
        "summary": "Southwest then northeast with hover",
        "clarification_question": None,
        "assumptions": [],
        "actions": [
            _action("takeoff", altitude_m=12),
            _action("inspect_corners", altitude_m=12, speed_mps=1.5,
                    corners=["corner_sw", "corner_ne"],
                    corner_strategy="given_order",
                    evidence_requirement="clear_photo", standoff_m=10),
            _action("hover", duration_s=5),
            _action("land"),
        ],
    })
    assert a.model_dump() != b.model_dump()
    assert a.actions[1].corners != b.actions[1].corners
    assert [x.kind for x in a.actions] != [x.kind for x in b.actions]


def test_nearest_safe_order_depends_on_runtime_start_position():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    order_a = ex._ordered_corners(
        ["corner_ne", "corner_nw", "corner_se", "corner_sw"],
        "nearest_safe", (-12, -12), 9,
    )
    order_b = ex._ordered_corners(
        ["corner_ne", "corner_nw", "corner_se", "corner_sw"],
        "nearest_safe", (12, 12), 9,
    )
    assert order_a != order_b


def test_adaptive_safety_gate_rejects_speed_over_demo_limit():
    intent = NaturalLanguageMission.model_validate({
        "disposition": "execute",
        "summary": "too fast",
        "clarification_question": None,
        "assumptions": [],
        "actions": [
            _action("takeoff", altitude_m=8),
            _action("move", distance_m=5, direction="forward", speed_mps=6),
            _action("land"),
        ],
    })
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    with pytest.raises(ValueError, match="SafetyGate: speed"):
        ex._validate_intent(intent)


def test_server_no_longer_uses_narrow_adaptive_interpreter():
    root = Path(__file__).resolve().parents[1]
    src = (root / "simulation" / "server.py").read_text(encoding="utf-8")
    assert "GPTAdaptiveInspectionInterpreter" not in src
    assert "adaptive_inspection=True" in src
