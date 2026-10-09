import cv2
import math
import numpy as np
from app.adaptive import AdaptiveMissionExecutor, SurfaceEvidence


def executor():
    return AdaptiveMissionExecutor(lambda *args, **kwargs: None, memory=None)


def add_north_wall(ex):
    ex.planner.add_obstacles([{
        "name":"north_wall", "position":[0.0,8.2,6.0],
        "size":[8.5,1.2,12.0], "top_z":12.0, "no_fly":False,
    }])


def test_north_wall_blocks_straight_view_but_oblique_view_exists_at_9m():
    ex = executor(); add_north_wall(ex)
    look = ex.SURFACE_LOOK_AT["north_facade"]
    assert ex._visual_line_clear((0.0,10.58,9.0), look) is False
    viable = [
        c for c in ex._surface_candidates("north_facade", 9.0, 7.5)
        if ex._visual_line_clear(c, look)
    ]
    assert viable, "planner must generate a side-looking view around the wall"
    assert any(abs(c[0]) >= 8.0 for c in viable)


def test_best_north_candidate_avoids_known_wall():
    ex = executor(); add_north_wall(ex)
    c = ex._best_surface_candidate((-12.0,-12.0), "north_facade", 9.0, 7.5)
    assert c is not None
    assert ex._visual_line_clear(c, ex.SURFACE_LOOK_AT["north_facade"])


def test_route_visibility_scoring_rejects_blocked_candidate():
    ex = executor(); add_north_wall(ex)
    blocked = (0.0, 10.58, 9.0)
    score = ex._surface_candidate_score((-12,-12), "north_facade", blocked, 9.0, 7.5)
    assert math.isinf(score)


def marker(marker_id, size=360):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.generateImageMarker(d, marker_id, size)


def test_surface_evidence_robust_to_scaled_blurred_render_like_image(tmp_path):
    canvas = np.full((720, 960), 235, dtype=np.uint8)
    ids = [0,1,2]
    for i, marker_id in enumerate(ids):
        m = marker(marker_id, 220)
        m = cv2.resize(m, (145,145), interpolation=cv2.INTER_AREA)
        x = 120 + i*250
        canvas[270:415, x:x+145] = m
    canvas = cv2.GaussianBlur(canvas, (3,3), .6)
    path = tmp_path/'north.png'
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    result = SurfaceEvidence().assess(str(path), 'north_facade')
    assert result['accepted'] is True
    assert len(result['matched_target_markers']) >= 3


def test_stable_navigation_source_does_not_replan_unconditionally_each_step():
    import inspect
    src = inspect.getsource(AdaptiveMissionExecutor._navigate)
    assert 'new_obstacle_blocks_active_route' in src
    assert 'route = None' in src
