import asyncio
import math
import pytest

from app.adaptive import AdaptiveMissionExecutor, Adaptive3DPlanner


class FakeWorld:
    def __init__(self, position):
        self._position = [position[0], position[1], position[2] + .24]

    def state(self):
        return {"position": list(self._position)}


class FakeDrone:
    def __init__(self, position=(0.0, 0.0, 12.0), move=True):
        self.world = FakeWorld(position)
        self.move = move
        self.commands = 0

    async def sense_obstacles(self, range_m=10.0, altitude_m=0.0):
        return []

    async def go_to_local(self, x, y, z, heading, speed, label):
        self.commands += 1
        if self.move:
            self.world._position = [float(x), float(y), float(z) + .24]


class LongRoutePlanner:
    """A deliberately long winding route with >40 execution legs."""
    def __init__(self):
        self.obstacles = {}

    def add_obstacles(self, obstacles):
        return []

    def segment_clear(self, a, b):
        return True

    def plan(self, start, goal, allow_altitude=True, nominal_altitude=None):
        pts = [tuple(start)]
        # 55 waypoints, each about 1m apart, ending at the requested goal.
        for i in range(1, 56):
            t = i / 56.0
            x = start[0] + (goal[0] - start[0]) * t
            y = 2.0 * math.sin(i * 0.5)
            z = start[2]
            pts.append((x, y, z))
        pts.append(tuple(goal))
        return pts


class StraightPlanner:
    def __init__(self):
        self.obstacles = {}

    def add_obstacles(self, obstacles):
        return []

    def segment_clear(self, a, b):
        return True

    def plan(self, start, goal, allow_altitude=True, nominal_altitude=None):
        return [tuple(start), tuple(goal)]


def test_navigation_does_not_fail_after_40_valid_motion_cycles():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    ex.planner3d = LongRoutePlanner()
    drone = FakeDrone((0, 0, 12), move=True)

    asyncio.run(ex._navigate_3d(
        drone, (20, 0, 12), "long valid route",
        speed=2.0, allow_altitude=True, nominal_altitude=12
    ))

    # This route intentionally takes more than the old hardcoded 40 iterations.
    assert drone.commands > 40
    pos = ex._current_xyz(drone)
    assert math.dist(pos, (20, 0, 12)) < .75


def test_true_no_motion_is_detected_as_stall_not_attempt_budget():
    ex = AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)
    ex.planner3d = StraightPlanner()
    drone = FakeDrone((0, 0, 12), move=False)

    with pytest.raises(RuntimeError, match="stopped making physical progress"):
        asyncio.run(ex._navigate_3d(
            drone, (10, 0, 12), "stalled vehicle",
            speed=2.0, allow_altitude=True, nominal_altitude=12
        ))
    assert drone.commands == 4


def test_same_quantized_3d_cell_still_returns_real_motion_leg():
    p = Adaptive3DPlanner(
        bounds=(-10,10,-10,10),
        min_altitude=5,
        max_altitude=20,
        xy_resolution=2,
        z_resolution=2
    )
    route = p.plan((0.1, 0.1, 12.0), (0.8, 0.8, 12.0), allow_altitude=True)
    assert len(route) == 2
    assert route[0] == (0.1, 0.1, 12.0)
    assert route[-1] == (0.8, 0.8, 12.0)


def test_old_recovery_budget_error_string_is_gone():
    import inspect
    source = inspect.getsource(AdaptiveMissionExecutor._navigate_3d)
    assert "recovery budget" not in source.lower()
    assert "range(40)" not in source
