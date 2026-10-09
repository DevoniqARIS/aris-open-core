"""Fast, offline ARIS Open Core sanity check."""
import asyncio
import math

from app.maneuver import NaturalLanguageMission, MissionAction
from app.adaptive import Adaptive3DPlanner


def main():
    mission = NaturalLanguageMission(
        disposition="execute",
        summary="Community self-test",
        assumptions=[],
        actions=[
            MissionAction(kind="takeoff", altitude_m=8),
            MissionAction(
                kind="find_and_inspect",
                object_class="building",
                object_selection="nearest",
                discovered_inspection_style="orbit",
                evidence_requirement="clear_photo",
                altitude_m=8,
                altitude_policy="adaptive",
            ),
            MissionAction(kind="return_home"),
            MissionAction(kind="land"),
        ],
    )
    planner = Adaptive3DPlanner()
    planner.add_obstacles([{
        "name": "wall",
        "position": [0.0, 0.0, 4.0],
        "size": [10.0, 1.0, 8.0],
        "top_z": 8.0,
        "no_fly": False,
    }])
    route = planner.plan((-8.0, -4.0, 8.0), (8.0, 4.0, 8.0),
                         allow_altitude=True, nominal_altitude=8.0)
    assert len(route) >= 2
    assert route[0] == (-8.0, -4.0, 8.0)
    assert route[-1] == (8.0, 4.0, 8.0)
    assert max(p[2] for p in route) <= 20.0

    print("ARIS Open Core self-test: PASS")
    print(f"Mission IR actions: {len(mission.actions)}")
    print(f"3D route points: {len(route)}")


if __name__ == "__main__":
    main()
