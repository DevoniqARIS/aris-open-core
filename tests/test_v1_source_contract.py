from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_browser_reads_mission_ir_actions_not_legacy_steps():
    js = (ROOT / "simulation" / "web" / "main.js").read_text(encoding="utf-8")
    assert "maneuver_intent.actions.entries()" in js
    assert "maneuver_intent.steps.entries()" not in js
    assert "maneuver_intent.assumptions" in js


def test_no_phrase_keyed_flight_fallback_in_maneuver_source():
    src = (ROOT / "app" / "maneuver.py").read_text(encoding="utf-8")
    forbidden = [
        'if "three circles" in',
        'if "hover at 6m" in',
        'if "fly straight" in',
        "play_circle_demo",
        "preset_trajectory",
    ]
    for token in forbidden:
        assert token not in src
