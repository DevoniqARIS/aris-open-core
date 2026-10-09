import math
from pathlib import Path

from app.adaptive import Adaptive3DPlanner, AdaptiveMissionExecutor


def building_obstacle(no_fly=False):
    return {
        "name": "primary_asset_keepout",
        "position": [0.0, 0.0, 2.0],
        "size": [8.8, 6.8, 4.0],
        "top_z": 4.0,
        "no_fly": no_fly,
    }


def test_physical_building_is_overflyable_above_clearance():
    planner = Adaptive3DPlanner()
    planner.add_obstacles([building_obstacle(no_fly=False)])
    assert planner.segment_clear((-8, 0, 12), (8, 0, 12)) is True


def test_true_no_fly_region_remains_infinite_height():
    planner = Adaptive3DPlanner()
    planner.add_obstacles([building_obstacle(no_fly=True)])
    assert planner.segment_clear((-8, 0, 20), (8, 0, 20)) is False


def test_planner_can_escape_an_inflated_clearance_buffer():
    planner = Adaptive3DPlanner()
    # At z=5 the aircraft is outside actual 4m-tall geometry, but it can be
    # inside the 1m vertical clearance buffer. It must be able to leave safely.
    planner.add_obstacles([building_obstacle(no_fly=False)])
    start = (-4.0, -4.0, 5.0)
    goal = (-10.0, -10.0, 7.0)
    assert planner._blocked(*start) is True
    assert planner._physical_blocked(*start) is False
    route = planner.plan(start, goal, allow_altitude=True, nominal_altitude=5.0)
    assert route[0] == start
    assert route[-1] == goal
    assert len(route) >= 2


def test_orbit_region_has_reachable_candidate_near_discovered_building():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    ex.planner3d.add_obstacles([building_obstacle(no_fly=False)])

    obj = {
        "id": "building_primary",
        "class": "building",
        "position": [0.0, 0.0, 2.0],
        "size": [8.0, 6.0, 4.0],
        "top_z": 4.0,
    }
    nominal = 12.0
    radius = max(7.5, max(obj["size"][:2]) / 2 + 4.0)
    theta = 0.0
    sector = []
    for dr in (0.0, 2.5, 5.0, 7.0, 10.0):
        for dz in (0.0, 3.0, 6.0, -2.0, 8.0):
            z = max(5.0, min(20.0, nominal + dz))
            r = radius + dr
            sector.append((r * math.cos(theta), r * math.sin(theta), z))

    # This position was pathological when the target building was treated as a
    # no-fly column / inflated-buffer trap.
    goal = ex._best_3d_goal(
        (-4.0, -4.0, 12.0),
        sector,
        allow_altitude=True,
        nominal_altitude=12.0,
    )
    assert goal is not None


def test_sim_world_does_not_mark_primary_building_as_no_fly():
    root = Path(__file__).resolve().parents[1]
    text = (root / "simulation" / "world.py").read_text(encoding="utf-8")
    block = text[text.index("'name': 'primary_asset_keepout'"):][:220]
    assert "'no_fly': False" in block
