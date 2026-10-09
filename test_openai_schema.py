from app.ai_client import _structured_schema
from app.maneuver import NaturalLanguageMission


def _assert_strict_objects(node):
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(props.keys())
        assert "default" not in node
        for value in node.values():
            _assert_strict_objects(value)
    elif isinstance(node, list):
        for value in node:
            _assert_strict_objects(value)


def test_mission_ir_schema_is_openai_strict_compatible_shape():
    raw = NaturalLanguageMission.model_json_schema()
    strict = _structured_schema(raw)
    _assert_strict_objects(strict)

    action = strict["$defs"]["MissionAction"]
    assert "altitude_m" in action["required"]
    assert "target" in action["required"]
    assert "repeat" in action["required"]
    altitude = action["properties"]["altitude_m"]
    assert {"type": "null"} in altitude["anyOf"]
