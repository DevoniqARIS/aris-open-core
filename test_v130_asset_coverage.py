import asyncio
from app.maneuver import GPTManeuverInterpreter, NaturalLanguageMission
from app.adaptive import AdaptiveMissionExecutor, SurfaceEvidence


def action(kind, **kw):
    d = {
        "kind": kind, "altitude_m": None, "speed_mps": None, "duration_s": None,
        "distance_m": None, "direction": None, "heading_deg": None, "turn_degrees": None,
        "target": None, "stop_distance_m": None, "radius_m": None, "laps": None,
        "orbit_direction": None, "corners": None, "corner_strategy": None,
        "inspection_targets": None, "coverage_mode": None, "coverage_strategy": None,
        "evidence_requirement": None, "standoff_m": None, "repeat": 1,
    }
    d.update(kw)
    return d


class FakeAllDirectionsModel:
    async def generate_with_meta(self, instructions, content, schema, schema_name="x"):
        assert 'never translate "all directions"' in instructions.lower()
        return {
            "disposition":"execute",
            "summary":"Inspect all four sides at 8 m, orbit for inspection, return home and land.",
            "clarification_question":None,
            "assumptions":["one orbit lap used because count was not specified"],
            "actions":[
                action("takeoff", altitude_m=8),
                action("inspect_asset", altitude_m=8, speed_mps=2,
                       inspection_targets=["north_facade","east_facade","south_facade","west_facade"],
                       coverage_mode="all_sides", coverage_strategy="nearest_safe",
                       evidence_requirement="clear_photo", standoff_m=7.5),
                action("orbit", altitude_m=8, speed_mps=2, target="asset_center",
                       radius_m=8, laps=1, orbit_direction="clockwise"),
                action("return_home", speed_mps=2),
                action("land"),
            ]
        }, {"response_id":"resp_v130","model":"fake","status":"completed"}


def test_exact_user_request_maps_to_facades_not_corners():
    intent, _ = asyncio.run(GPTManeuverInterpreter(client=FakeAllDirectionsModel()).interpret(
        "Inspect the building from all four directions at 8m altitude. Circle the building for full inspection. "
        "Avoid obstacles. Find ways to complete the mission. When finished, return to your starting position and land.",
        adaptive_inspection=True,
    ))
    asset = next(a for a in intent.actions if a.kind == "inspect_asset")
    assert set(asset.inspection_targets) == {"north_facade","east_facade","south_facade","west_facade"}
    assert not any(a.kind == "inspect_corners" for a in intent.actions)
    assert any(a.kind == "orbit" for a in intent.actions)
    assert asset.altitude_m == 8


def test_all_sides_normalizer_expands_surfaces_without_corners():
    raw = {
        "disposition":"execute", "summary":"all sides", "clarification_question":None,
        "assumptions":[], "actions":[
            action("takeoff", altitude_m=8),
            action("inspect_asset", altitude_m=8, coverage_mode="all_sides",
                   coverage_strategy="nearest_safe", evidence_requirement="clear_photo"),
            action("land")
        ]
    }
    from app.maneuver import _normalize_intent
    m = NaturalLanguageMission.model_validate(_normalize_intent(raw))
    inspect = next(a for a in m.actions if a.kind == "inspect_asset")
    assert inspect.inspection_targets == ["north_facade","east_facade","south_facade","west_facade"]
    assert inspect.corners is None


def test_surface_marker_groups_are_distinct():
    groups = SurfaceEvidence.EXPECTED
    assert groups["north_facade"] == [0,1,2,3]
    assert groups["east_facade"] == [4,5,6,7]
    assert groups["south_facade"] == [8,9,10,11]
    assert groups["west_facade"] == [12,13,14,15]
    assert groups["roof"] == [16,17,18,19]


def test_surface_viewpoints_are_generated_from_surface_geometry():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    north = ex._surface_candidates("north_facade", altitude=8, standoff=7.5)
    east = ex._surface_candidates("east_facade", altitude=8, standoff=7.5)
    assert north != east
    assert all(z == 8 for x,y,z in north)
    assert any(y > 9 for x,y,z in north)
    assert any(x > 10 for x,y,z in east)
