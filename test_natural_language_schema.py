from app.ai_client import _structured_schema
from app.maneuver import NaturalLanguageMission


def walk(node):
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(props.keys())
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


def test_natural_language_ir_schema_is_strict_output_compatible():
    schema = _structured_schema(NaturalLanguageMission.model_json_schema())
    walk(schema)
    action = schema["$defs"]["MissionAction"]
    assert "altitude_m" in action["required"]
    assert "repeat" in action["required"]
