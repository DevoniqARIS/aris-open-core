from __future__ import annotations

import asyncio
import cv2
import hashlib
import heapq
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ai_client import ResponsesJSON
from .change_detection import ChangeDetector
from .adapters.capabilities import DroneCapabilities


Corner = Literal["corner_ne", "corner_nw", "corner_se", "corner_sw"]
ALL_CORNERS: list[Corner] = ["corner_ne", "corner_se", "corner_sw", "corner_nw"]
Surface = Literal["north_facade", "east_facade", "south_facade", "west_facade", "roof"]
ALL_SIDES: list[Surface] = ["north_facade", "east_facade", "south_facade", "west_facade"]

from .maneuver import (
    NaturalLanguageMission, MissionAction, DEFAULT_SPEED_MPS,
    MAX_ALTITUDE_M, MAX_SPEED_MPS, MAX_DURATION_S, MAX_MOVE_DISTANCE_M,
    MAX_ORBIT_RADIUS_M, MAX_LAPS,
)


class GridReplanner:
    """2D A* planner over obstacles ARIS has actually discovered."""

    def __init__(self, bounds=(-26.0, 26.0, -26.0, 26.0), resolution=1.5, clearance=1.1):
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.resolution = float(resolution)
        self.clearance = float(clearance)
        self.obstacles: dict[str, dict] = {}
        # Clean slate: no site obstacle or asset location is pre-seeded.
        # Geometry enters this map only through runtime local sensing.

    def add_obstacles(self, obstacles: list[dict]) -> list[str]:
        added = []
        for obstacle in obstacles:
            name = obstacle["name"]
            if name not in self.obstacles:
                self.obstacles[name] = dict(obstacle)
                added.append(name)
        return added

    def _blocked(self, x: float, y: float, altitude_m: float) -> bool:
        for o in self.obstacles.values():
            if not o.get("no_fly") and float(o.get("top_z", 100)) < altitude_m + 0.5:
                continue
            px, py = o["position"][:2]
            sx, sy = o["size"][:2]
            if abs(x - px) <= sx / 2 + self.clearance and abs(y - py) <= sy / 2 + self.clearance:
                return True
        return False

    def _node(self, point):
        return (
            round((point[0] - self.min_x) / self.resolution),
            round((point[1] - self.min_y) / self.resolution),
        )

    def _point(self, node):
        return (
            self.min_x + node[0] * self.resolution,
            self.min_y + node[1] * self.resolution,
        )

    def _valid_node(self, node, altitude_m):
        x, y = self._point(node)
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and not self._blocked(x, y, altitude_m)
        )

    def _segment_clear(self, a, b, altitude_m):
        distance = math.dist(a, b)
        steps = max(2, math.ceil(distance / 0.4))
        for i in range(steps + 1):
            t = i / steps
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if self._blocked(x, y, altitude_m):
                return False
        return True

    def plan(self, start, goal, altitude_m):
        start_n, goal_n = self._node(start), self._node(goal)
        if not self._valid_node(goal_n, altitude_m):
            raise ValueError("Requested viewpoint is inside a discovered obstacle")

        moves = [
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]
        frontier = [(0.0, start_n)]
        came = {start_n: None}
        cost = {start_n: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_n:
                break
            for dx, dy, move_cost in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._valid_node(nxt, altitude_m):
                    continue
                # Prevent diagonal corner cutting.
                if dx and dy:
                    if not self._valid_node((current[0] + dx, current[1]), altitude_m):
                        continue
                    if not self._valid_node((current[0], current[1] + dy), altitude_m):
                        continue
                new_cost = cost[current] + move_cost
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    gx, gy = goal_n
                    heuristic = math.hypot(nxt[0] - gx, nxt[1] - gy)
                    heapq.heappush(frontier, (new_cost + heuristic, nxt))
                    came[nxt] = current

        if goal_n not in came:
            raise ValueError("No safe route exists through the currently known obstacle map")

        nodes = []
        cur = goal_n
        while cur is not None:
            nodes.append(cur)
            cur = came[cur]
        points = [self._point(n) for n in reversed(nodes)]
        points[0] = tuple(start)
        points[-1] = tuple(goal)

        # Greedily simplify while respecting every discovered obstacle.
        simplified = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._segment_clear(points[i], points[j], altitude_m):
                j -= 1
            simplified.append(points[j])
            i = j
        return simplified


class Adaptive3DPlanner:
    """3D A* over obstacles ARIS has actually discovered.

    This planner is intentionally additive: legacy 2D inspection routes continue to
    use GridReplanner. Clean-slate discovery missions can use this planner when the
    semantic mission permits altitude adaptation.
    """

    def __init__(self, bounds=(-26.0, 26.0, -26.0, 26.0), min_altitude=5.0,
                 max_altitude=20.0, xy_resolution=2.0, z_resolution=2.0,
                 horizontal_clearance=1.15, vertical_clearance=1.0):
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.min_altitude = float(min_altitude)
        self.max_altitude = float(max_altitude)
        self.xy_resolution = float(xy_resolution)
        self.z_resolution = float(z_resolution)
        self.horizontal_clearance = float(horizontal_clearance)
        self.vertical_clearance = float(vertical_clearance)
        self.obstacles: dict[str, dict] = {}

    def add_obstacles(self, obstacles):
        added=[]
        for obstacle in obstacles:
            name=obstacle['name']
            if name not in self.obstacles:
                self.obstacles[name]=dict(obstacle); added.append(name)
            else:
                self.obstacles[name].update(dict(obstacle))
        return added

    def _blocked(self, x, y, z):
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return True
        if not (self.min_altitude <= z <= self.max_altitude):
            return True
        for o in self.obstacles.values():
            px,py=o['position'][:2]; sx,sy=o['size'][:2]
            horizontal = (abs(x-px) <= sx/2 + self.horizontal_clearance and
                          abs(y-py) <= sy/2 + self.horizontal_clearance)
            if not horizontal:
                continue
            if o.get('no_fly'):
                return True
            if z <= float(o.get('top_z', 100.0)) + self.vertical_clearance:
                return True
        return False

    def _physical_blocked(self, x, y, z):
        """True only for actual obstacle volume / true no-fly regions.

        This deliberately omits the planner's clearance inflation. It is used only
        to let an aircraft that has just discovered an obstacle while already inside
        the *clearance margin* safely move outward to a valid planning cell.
        """
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return True
        if not (self.min_altitude <= z <= self.max_altitude):
            return True
        for o in self.obstacles.values():
            px, py = o['position'][:2]
            sx, sy = o['size'][:2]
            horizontal = (
                abs(x - px) <= sx / 2
                and abs(y - py) <= sy / 2
            )
            if not horizontal:
                continue
            if o.get('no_fly'):
                return True
            if z <= float(o.get('top_z', 100.0)):
                return True
        return False

    def _segment_physical_clear(self, a, b):
        distance = math.dist(a, b)
        steps = max(2, math.ceil(distance / 0.3))
        for i in range(steps + 1):
            t = i / steps
            p = (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            )
            if self._physical_blocked(*p):
                return False
        return True

    def _nearest_valid_escape(self, start):
        """Find a valid planning node reachable from a clearance-buffer violation.

        The aircraft is allowed to EXIT a clearance buffer it already occupies, but
        never to pass through actual geometry or a true no-fly region.
        """
        if self._physical_blocked(*start):
            return None

        sn = self._node(start)
        candidates = []
        # Up to ~10m horizontal/vertical search in this small simulation world.
        for radius in range(1, 7):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        n = (sn[0] + dx, sn[1] + dy, sn[2] + dz)
                        if not self._valid(n):
                            continue
                        p = self._point(n)
                        if self._segment_physical_clear(start, p):
                            candidates.append((math.dist(start, p), n, p))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1], candidates[0][2]
        return None

    def _node(self, p):
        x,y,z=p
        return (round((x-self.min_x)/self.xy_resolution),
                round((y-self.min_y)/self.xy_resolution),
                round((z-self.min_altitude)/self.z_resolution))

    def _point(self, n):
        return (self.min_x+n[0]*self.xy_resolution,
                self.min_y+n[1]*self.xy_resolution,
                self.min_altitude+n[2]*self.z_resolution)

    def _valid(self, n):
        x,y,z=self._point(n)
        return not self._blocked(x,y,z)

    def segment_clear(self, a, b):
        d=math.dist(a,b); steps=max(2, math.ceil(d/0.45))
        for i in range(steps+1):
            t=i/steps
            p=(a[0]+(b[0]-a[0])*t,
               a[1]+(b[1]-a[1])*t,
               a[2]+(b[2]-a[2])*t)
            if self._blocked(*p): return False
        return True

    def plan(self, start, goal, allow_altitude=True, nominal_altitude=None):
        start=tuple(float(v) for v in start); goal=tuple(float(v) for v in goal)
        if not allow_altitude:
            z=float(nominal_altitude if nominal_altitude is not None else goal[2])
            start=(start[0],start[1],z); goal=(goal[0],goal[1],z)
        start_n=self._node(start); goal_n=self._node(goal)
        if not self._valid(goal_n):
            raise ValueError('Requested 3D goal lies inside discovered obstacle volume')

        # A newly discovered obstacle can place the current aircraft pose inside the
        # planner's *inflated safety buffer* even though it is not inside the actual
        # obstacle. Previously that made every outgoing edge invalid and caused every
        # orbit candidate to look unreachable. Escape the margin first, then resume A*.
        escape_prefix = []
        if not self._valid(start_n):
            escape = self._nearest_valid_escape(start)
            if escape is None:
                if self._physical_blocked(*start):
                    raise ValueError(
                        'Current aircraft position intersects physical obstacle/no-fly geometry'
                    )
                raise ValueError(
                    'Unable to exit newly discovered obstacle clearance buffer safely'
                )
            start_n, escape_point = escape
            escape_prefix = [start, escape_point]
            start = escape_point

        # A start and goal can quantize to the same grid cell even while still being
        # far enough apart to require a real movement command. Returning a one-point
        # route here used to make navigation think the goal had been reached.
        if start_n == goal_n:
            if math.dist(start, goal) < 0.05:
                return [start]
            if self.segment_clear(start, goal):
                return [start, goal]

        # 8 horizontal moves + vertical moves. Vertical transitions are deliberately
        # explicit; this avoids cutting diagonally through roof/wall corners.
        moves=[(-1,0,0,1),(1,0,0,1),(0,-1,0,1),(0,1,0,1),
               (-1,-1,0,1.414),(-1,1,0,1.414),(1,-1,0,1.414),(1,1,0,1.414)]
        if allow_altitude:
            moves += [(0,0,-1,1.15),(0,0,1,1.15)]

        frontier=[(0.0,start_n)]; came={start_n:None}; cost={start_n:0.0}
        while frontier:
            _,cur=heapq.heappop(frontier)
            if cur==goal_n: break
            curp=self._point(cur)
            for dx,dy,dz,mc in moves:
                nxt=(cur[0]+dx,cur[1]+dy,cur[2]+dz)
                if not self._valid(nxt): continue
                nxtp=self._point(nxt)
                if not self.segment_clear(curp,nxtp): continue
                nc=cost[cur]+mc
                if nxt not in cost or nc<cost[nxt]:
                    cost[nxt]=nc
                    gp=self._point(goal_n)
                    h=math.dist(nxtp,gp)/self.xy_resolution
                    heapq.heappush(frontier,(nc+h,nxt)); came[nxt]=cur
        if goal_n not in came:
            raise ValueError('No safe 3D route exists through the currently known obstacle map')

        nodes=[]; cur=goal_n
        while cur is not None:
            nodes.append(cur); cur=came[cur]
        pts=[self._point(n) for n in reversed(nodes)]
        pts[0]=start; pts[-1]=goal

        if escape_prefix:
            # Preserve the actual current aircraft pose as the first point. The first
            # leg only exits a clearance margin and has already been checked against
            # non-inflated physical geometry.
            if math.dist(escape_prefix[-1], pts[0]) < 0.05:
                pts = [escape_prefix[0]] + pts
            else:
                pts = escape_prefix + pts

        # Greedy 3D line-of-sight simplification produces smooth climb/cruise/descent legs.
        if escape_prefix and len(pts) >= 2:
            simplified = [pts[0], pts[1]]
            i = 1
        else:
            simplified = [pts[0]]
            i = 0
        while i<len(pts)-1:
            j=len(pts)-1
            while j>i+1 and not self.segment_clear(pts[i],pts[j]):
                j-=1
            simplified.append(pts[j])
            i=j
        return simplified


def _aruco_detect_ids(gray):
    """Robust image-only ArUco decoding across supported OpenCV versions."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = None
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.012
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 8.0
        if hasattr(params, "adaptiveThreshWinSizeMax"):
            params.adaptiveThreshWinSizeMax = 53

    detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params) if params is not None else cv2.aruco.ArucoDetector(dictionary)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    up2 = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up2_enhanced = clahe.apply(up2)
    passes = [gray, enhanced, up2, up2_enhanced]
    for base in (enhanced, up2_enhanced):
        blur = cv2.GaussianBlur(base, (3, 3), 0)
        passes.append(cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        ))

    found = set()
    for image in passes:
        if detector is not None:
            _, ids, _ = detector.detectMarkers(image)
        elif hasattr(cv2.aruco, "detectMarkers"):
            _, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
        else:
            raise RuntimeError("OpenCV ArUco detector API unavailable")
        if ids is not None:
            found.update(int(x) for x in ids.flatten())
    return sorted(found)


class CornerEvidence:
    """Optical corner evidence from actual rendered pixels (ArUco marker detection)."""

    EXPECTED = {
        "corner_ne": [20, 21],
        "corner_se": [22, 23],
        "corner_sw": [24, 25],
        "corner_nw": [26, 27],
    }

    def assess(self, image_path: str, corner: Corner):
        image = cv2.imread(str(image_path))
        if image is None:
            return {"accepted": False, "detected_ids": [], "reason": "image_unreadable"}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = _aruco_detect_ids(gray)
        target = self.EXPECTED[corner]
        accepted = all(marker_id in detected for marker_id in target)
        return {
            "accepted": accepted,
            "detected_ids": detected,
            "target_markers": target,
            "reason": "both_adjacent_corner_panels_detected" if accepted else "corner_evidence_occluded_or_out_of_frame",
        }


class SurfaceEvidence:
    """Optical surface evidence using the existing facade/roof calibration panels."""

    EXPECTED = {
        "north_facade": [0, 1, 2, 3],
        "east_facade": [4, 5, 6, 7],
        "south_facade": [8, 9, 10, 11],
        "west_facade": [12, 13, 14, 15],
        "roof": [16, 17, 18, 19],
    }

    def assess(self, image_path: str, target: Surface):
        image = cv2.imread(str(image_path))
        if image is None:
            return {"accepted": False, "detected_ids": [], "reason": "image_unreadable"}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = _aruco_detect_ids(gray)
        expected = self.EXPECTED[target]
        found = [marker_id for marker_id in expected if marker_id in detected]
        accepted = len(found) >= 3
        return {
            "accepted": accepted,
            "detected_ids": detected,
            "target_markers": expected,
            "matched_target_markers": found,
            "reason": "surface_coverage_verified" if accepted else "surface_evidence_occluded_or_out_of_frame",
        }


class PersistentAssetMemory:
    """Cross-run asset evidence memory stored outside individual run directories."""

    def __init__(self, root: Path, asset_id="simulation-building"):
        self.root = Path(root) / asset_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "memory.json"
        if not self.index.exists():
            self.index.write_text(json.dumps({"asset_id": asset_id, "runs": []}, indent=2))
        self.detector = ChangeDetector()

    def load(self):
        try:
            return json.loads(self.index.read_text())
        except Exception:
            return {"asset_id": self.root.name, "runs": []}

    def record(self, run_id: str, captures: list[dict], scene_variant: str):
        memory = self.load()
        previous = memory["runs"][-1] if memory["runs"] else None
        dest = self.root / run_id
        dest.mkdir(parents=True, exist_ok=True)
        stored = []
        changes = []

        def evidence_key(item):
            return item.get("target") or item.get("corner")

        previous_by_target = {evidence_key(x): x for x in (previous or {}).get("captures", [])}
        for capture in captures:
            source = Path(capture["image_path"])
            key = evidence_key(capture)
            target = dest / f'{key}.png'
            shutil.copy2(source, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            item = {"target": key, "image_path": str(target), "sha256": digest}
            stored.append(item)
            if key in previous_by_target:
                prior = previous_by_target[key]
                change = self.detector.compare_pair(prior["image_path"], str(target), key)
                changes.append(change.model_dump())

        entry = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scene_variant": scene_variant,
            "captures": stored,
            "changes_vs_previous": changes,
        }
        memory["runs"].append(entry)
        self.index.write_text(json.dumps(memory, indent=2))
        return {
            "baseline_created": previous is None,
            "previous_run_id": previous.get("run_id") if previous else None,
            "changes": changes,
            "inspection_count": len(memory["runs"]),
        }


class AdaptiveMissionExecutor:
    """Unified Mission IR executor with discovery, replanning and evidence verification."""

    CORNERS = {
        "corner_ne": (4.0, 3.0),
        "corner_se": (4.0, -3.0),
        "corner_sw": (-4.0, -3.0),
        "corner_nw": (-4.0, 3.0),
    }
    NAMED_TARGETS = {
        "home": (-12.0, -12.0),
        "asset_center": (0.0, 0.0),
        "asset_north": (0.0, 8.0),
        "asset_south": (0.0, -8.0),
        "asset_east": (9.0, 0.0),
        "asset_west": (-9.0, 0.0),
    }
    SURFACE_LOOK_AT = {
        "north_facade": (0.0, 3.08, 2.0),
        "east_facade": (4.08, 0.0, 2.0),
        "south_facade": (0.0, -3.08, 2.0),
        "west_facade": (-4.08, 0.0, 2.0),
        "roof": (0.0, 0.0, 4.04),
    }
    SURFACE_NORMAL = {
        "north_facade": (0.0, 1.0),
        "east_facade": (1.0, 0.0),
        "south_facade": (0.0, -1.0),
        "west_facade": (-1.0, 0.0),
    }

    def __init__(self, journal, memory: PersistentAssetMemory | None, stop_event=None):
        self.journal = journal
        self.memory = memory
        self.stop_event = stop_event or asyncio.Event()
        self.planner = GridReplanner()
        self.planner3d = Adaptive3DPlanner()
        self.used_clean_slate_3d = False
        self.evidence = CornerEvidence()
        self.surface_evidence = SurfaceEvidence()
        self.replans = 0
        self.discovered = set()
        self.discovered_objects: dict[str, dict] = {}
        self.heading_deg = 90.0

    @staticmethod
    def _heading(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        return math.degrees(math.atan2(dx, dy)) % 360

    @staticmethod
    def _pitch(horizontal, drone_alt, target_alt=2.3):
        return -math.degrees(math.atan2(max(0.1, drone_alt - target_alt), max(0.5, horizontal)))

    @staticmethod
    def _xy_from_heading(origin, heading_deg, distance_m):
        r = math.radians(heading_deg)
        return (origin[0] + math.sin(r) * distance_m,
                origin[1] + math.cos(r) * distance_m)

    @staticmethod
    def _direction_heading(current_heading, direction):
        if direction in (None, "forward"):
            return current_heading % 360
        if direction == "backward":
            return (current_heading + 180) % 360
        if direction == "right":
            return (current_heading + 90) % 360
        if direction == "left":
            return (current_heading - 90) % 360
        return {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}[direction]

    def _target_xy(self, target, current):
        return current if target == "current_position" else self.NAMED_TARGETS[target]

    async def _discover_3d(self, drone, range_m=10.0):
        # Query at ground-reference altitude so nearby low obstacles are not hidden
        # merely because the aircraft is currently above them. The 3D planner then
        # decides whether to route around or above each discovered volume.
        sensed = await drone.sense_obstacles(range_m=range_m, altitude_m=0.0)
        new2d = self.planner.add_obstacles(sensed)
        new3d = self.planner3d.add_obstacles(sensed)
        for name in new3d:
            self.discovered.add(name)
            obstacle = next(o for o in sensed if o['name']==name)
            self.journal('obstacle_discovered', obstacle=name, obstacle_geometry=obstacle,
                         planning_dimension='3D')
        return list(dict.fromkeys(new2d + new3d))

    def _current_xyz(self, drone):
        p = drone.world.state()['position']
        return (float(p[0]), float(p[1]), max(5.0, float(p[2]) - .24))

    def _route3d_still_clear(self, current, route):
        if not route: return False
        pts=[current]+list(route)
        return all(self.planner3d.segment_clear(x,y) for x,y in zip(pts,pts[1:]))

    async def _navigate_3d(self, drone, goal_xyz, label, speed=2.0,
                           allow_altitude=True, nominal_altitude=None):
        """Progress-aware receding-horizon 3D navigation.

        There is deliberately no fixed "N recovery attempts" limit. A long but
        valid route must not fail merely because it contains many short safe legs.

        Safety termination is based on real behavior instead:
        - repeated commands that produce no physical movement;
        - excessive cumulative travel relative to the size of this bounded scene;
        - repeated route invalidation without convergence.

        Newly sensed geometry can still invalidate the active route, in which case
        ARIS replans from the aircraft's actual current pose.
        """
        self.used_clean_slate_3d = True
        route = None
        cursor = 0
        max_step = 4.0
        goal_xyz = tuple(float(v) for v in goal_xyz)

        initial = self._current_xyz(drone)
        direct_distance = math.dist(initial, goal_xyz)

        # This is an infinite-loop safety envelope, not an attempt counter.
        # It is intentionally far above any reasonable route inside the 52m x 52m
        # simulation site, including detours and altitude changes.
        max_travel_m = max(300.0, direct_distance * 10.0 + 100.0)
        traveled_m = 0.0
        consecutive_stalls = 0
        route_invalidations = 0
        last_reported_distance = direct_distance

        while True:
            if self.stop_event.is_set():
                raise RuntimeError('Operator requested abort')

            current = self._current_xyz(drone)
            distance_to_goal = math.dist(current, goal_xyz)

            if distance_to_goal < .75:
                await self._discover_3d(drone)
                self.journal(
                    'navigation_goal_reached',
                    label=label,
                    goal=list(goal_xyz),
                    traveled_m=round(traveled_m, 2),
                    remaining_m=round(distance_to_goal, 2),
                    planning_dimension='3D',
                )
                return

            newly = await self._discover_3d(drone)
            needs_plan = route is None or cursor >= len(route)

            if newly and route is not None and not self._route3d_still_clear(current, route[cursor:]):
                needs_plan = True
                route_invalidations += 1

            if needs_plan:
                try:
                    planned = await asyncio.to_thread(
                        self.planner3d.plan, current, goal_xyz,
                        allow_altitude, nominal_altitude
                    )
                except ValueError as exc:
                    raise ValueError(
                        f'3D route to the requested goal is not currently reachable: {exc}'
                    ) from exc

                route = planned[1:]
                cursor = 0

                # A valid plan must contain a movement leg unless already at goal.
                if not route:
                    if self.planner3d.segment_clear(current, goal_xyz):
                        route = [goal_xyz]
                    else:
                        raise ValueError(
                            '3D planner produced no executable movement toward the requested goal'
                        )

                if newly or route_invalidations:
                    self.replans += 1
                    self.journal(
                        'route_replanned',
                        reason='new_geometry_invalidated_active_3d_route',
                        replan_number=self.replans,
                        route_points=len(planned),
                        goal=list(goal_xyz),
                        known_obstacles=sorted(self.planner3d.obstacles),
                        label=label,
                        altitude_adaptation=bool(allow_altitude),
                    )
                else:
                    self.journal(
                        'route_planned',
                        route_points=len(planned),
                        goal=list(goal_xyz),
                        label=label,
                        planning_dimension='3D',
                    )

            wp = route[cursor]
            d = math.dist(current, wp)

            if d <= .7:
                cursor += 1
                continue

            if d > max_step:
                t = max_step / d
                target = (
                    current[0] + (wp[0] - current[0]) * t,
                    current[1] + (wp[1] - current[1]) * t,
                    current[2] + (wp[2] - current[2]) * t,
                )
            else:
                target = wp

            # Sense immediately before committing to this short leg.
            new_before_commit = await self._discover_3d(drone)
            if new_before_commit and not self.planner3d.segment_clear(current, target):
                route = None
                cursor = 0
                route_invalidations += 1
                self.journal(
                    'route_invalidated',
                    reason='new_obstacle_blocks_next_3d_leg',
                    label=label,
                    current=list(current),
                    blocked_target=list(target),
                    route_invalidations=route_invalidations,
                )
                continue

            if not self.planner3d.segment_clear(current, target):
                route = None
                cursor = 0
                route_invalidations += 1
                continue

            heading = (
                self._heading(current[:2], target[:2])
                if math.dist(current[:2], target[:2]) > .05
                else self.heading_deg
            )
            self.heading_deg = heading

            before = current
            await drone.go_to_local(
                target[0], target[1], target[2], heading, speed, label
            )
            after = self._current_xyz(drone)
            moved = math.dist(before, after)
            traveled_m += moved

            # Detect an actuator/control stall, not "too many loops".
            if moved < .18:
                consecutive_stalls += 1
                route = None
                cursor = 0
                self.journal(
                    'navigation_stall_detected',
                    label=label,
                    stall_count=consecutive_stalls,
                    position=list(after),
                    requested_target=list(target),
                )
                if consecutive_stalls >= 4:
                    raise RuntimeError(
                        '3D navigation stopped making physical progress after repeated command retries'
                    )
            else:
                consecutive_stalls = 0

            if math.dist(after, wp) < .3:
                cursor += 1

            if traveled_m > max_travel_m:
                raise RuntimeError(
                    '3D navigation exceeded the safe travel-distance envelope without reaching the goal'
                )

            # If obstacle discoveries keep invalidating routes without the aircraft
            # getting materially closer over a long period, surface a meaningful
            # convergence failure instead of a mysterious attempt-budget error.
            new_distance = math.dist(after, goal_xyz)
            if new_distance < last_reported_distance - 2.0:
                last_reported_distance = new_distance
                route_invalidations = 0
            elif route_invalidations >= 20:
                raise ValueError(
                    'Repeated obstacle discoveries prevented the 3D route from converging; '
                    'the caller should choose another goal region'
                )

    def _goal_region_candidates(self, obj, nominal_altitude, standoff, count=16):
        cx,cy,_=obj['position']; sx,sy,sz=obj['size']
        base=max(float(standoff), max(sx,sy)/2 + 3.5)
        alt0=float(nominal_altitude)
        altitudes=sorted(set(max(5.0,min(20.0,z)) for z in
            (alt0, alt0+3.0, alt0+6.0, max(5.0,alt0-2.0), float(obj.get('top_z',sz))+2.0)))
        radii=sorted(set((base, base+2.5, base+5.0)))
        out=[]
        for z in altitudes:
            for r in radii:
                for i in range(count):
                    theta=2*math.pi*i/count
                    out.append((cx+r*math.cos(theta),cy+r*math.sin(theta),z))
        return out

    def _best_3d_goal(self, start_xyz, candidates, allow_altitude=True, nominal_altitude=None):
        # First prefer candidates reachable by a direct safe segment. This avoids
        # running a full 3D A* once per candidate during every orbit sector. On an
        # older demo laptop that synchronous N×A* burst could starve the web server
        # long enough for the browser to report a false backend disconnect.
        direct=[]
        eligible=[]
        for c in candidates:
            if not allow_altitude and nominal_altitude is not None and abs(c[2]-nominal_altitude)>.25:
                continue
            eligible.append(c)
            if self.planner3d.segment_clear(start_xyz, c):
                vertical=abs(c[2]-start_xyz[2])*.15
                direct.append((math.dist(start_xyz,c)+vertical,c))
        if direct:
            direct.sort(key=lambda item:item[0])
            return direct[0][1]

        best=None; best_cost=float('inf')
        for c in eligible:
            try:
                route=self.planner3d.plan(start_xyz,c,allow_altitude=allow_altitude,
                                          nominal_altitude=nominal_altitude)
                cost=sum(math.dist(x,y) for x,y in zip(route,route[1:]))
                cost += sum(abs(y[2]-x[2]) for x,y in zip(route,route[1:]))*.15
                if cost<best_cost: best_cost=cost; best=c
            except ValueError:
                pass
        return best

    async def _discover_scene_objects(self, drone, range_m=14.0):
        if not hasattr(drone, "sense_objects"):
            raise RuntimeError("Connected adapter does not expose object discovery")
        seen = await drone.sense_objects(range_m=range_m)
        new = []
        for obj in seen:
            if obj["id"] not in self.discovered_objects:
                self.discovered_objects[obj["id"]] = dict(obj)
                new.append(obj["id"])
                self.journal(
                    "object_discovered", object_id=obj["id"], object_class=obj["class"],
                    estimated_position=obj["position"], estimated_size=obj["size"],
                    range_m=obj.get("range_m"), source="simulated onboard object detector",
                )
        return new

    def _matching_objects(self, object_class):
        values = list(self.discovered_objects.values())
        if object_class == "object":
            return values
        return [o for o in values if o.get("class") == object_class]

    async def _search_objects(self, drone, object_class, selection, altitude, speed,
                              allow_altitude=True):
        await self._discover_3d(drone)
        await self._discover_scene_objects(drone)
        if selection == 'nearest' and self._matching_objects(object_class):
            matches=self._matching_objects(object_class); current=self._current_xyz(drone)
            return [min(matches,key=lambda o:math.dist(current[:2],o['position'][:2]))]

        # Generate a boundary-aware serpentine exploration pattern. It is independent
        # of target locations and therefore remains a genuine clean-slate search.
        margin=5.0
        xs=[self.planner3d.min_x+margin, -8.0, 8.0, self.planner3d.max_x-margin]
        ys=[self.planner3d.min_y+margin, -8.0, 8.0, self.planner3d.max_y-margin]
        search_points=[]
        for i,y in enumerate(ys):
            row=xs if i%2==0 else list(reversed(xs))
            search_points.extend((x,y,float(altitude)) for x in row)

        for idx,point in enumerate(search_points):
            if self.stop_event.is_set(): raise RuntimeError('Operator requested abort')
            try:
                await self._navigate_3d(drone,point,f'clean-slate search {idx+1}',speed=speed,
                                        allow_altitude=allow_altitude, nominal_altitude=altitude)
            except ValueError:
                continue
            await self._discover_scene_objects(drone)
            matches=self._matching_objects(object_class)
            if selection=='nearest' and matches:
                current=self._current_xyz(drone)
                return [min(matches,key=lambda o:math.dist(current[:2],o['position'][:2]))]

        matches=self._matching_objects(object_class)
        if not matches: raise RuntimeError(f'Search completed but no {object_class} was discovered')
        if selection=='all': return matches
        if selection=='largest':
            return [max(matches,key=lambda o:o['size'][0]*o['size'][1]*o['size'][2])]
        home=self.NAMED_TARGETS['home']
        return [min(matches,key=lambda o:math.dist(home,o['position'][:2]))]

    def _object_viewpoints(self, obj, altitude, standoff, style):
        cx, cy, cz = obj["position"]
        sx, sy, sz = obj["size"]
        clearance = max(float(standoff), max(sx, sy)/2 + 3.0)
        z = max(float(altitude), min(18.0, cz + min(sz*.25, 3.0)))
        if style == "photograph":
            return [(cx-clearance, cy-clearance*.35, z, "photo")]
        # Four generic cardinal views around the discovered bounding box.
        return [
            (cx, cy + sy/2 + clearance, z, "north"),
            (cx + sx/2 + clearance, cy, z, "east"),
            (cx, cy - sy/2 - clearance, z, "south"),
            (cx - sx/2 - clearance, cy, z, "west"),
        ]

    async def _capture_discovered_object(self, drone, obj, viewpoint, run_id, attempt, label):
        x, y, z, side = viewpoint
        center = obj["position"]
        heading = self._heading((x,y), center[:2])
        horizontal = math.dist((x,y), center[:2])
        target_z = min(float(obj.get("top_z", center[2])), max(1.5, center[2]))
        pitch = -math.degrees(math.atan2(max(.1, z-target_z), max(.5, horizontal)))
        await drone.orient_camera_local(heading, pitch, f"frame discovered {obj['id']} {side}")
        capture = await drone.capture_discovered(run_id, obj["id"], label, attempt)
        visible = await drone.object_visible(obj["id"])
        self.journal(
            "discovered_object_capture", object_id=obj["id"], object_class=obj["class"],
            view=side, accepted=bool(visible), file=Path(capture["image_path"]).name,
            evidence_source="simulated object detector visibility + rendered RGB capture",
        )
        return capture, bool(visible)

    async def _inspect_discovered_object(self, action, obj, drone, run_id, captures, required_targets):
        nominal=float(action.altitude_m or 9.0); speed=float(action.speed_mps or DEFAULT_SPEED_MPS)
        standoff=float(action.standoff_m or 7.5); style=action.discovered_inspection_style
        allow_altitude=(action.altitude_policy != 'fixed')
        await self._discover_3d(drone)

        if style == 'orbit':
            center=tuple(obj['position'][:2]); radius=max(standoff,max(obj['size'][:2])/2+4.0)
            current=self._current_xyz(drone)
            # Start from nearest angular sector, then maintain direction around object.
            start_theta=math.atan2(current[1]-center[1],current[0]-center[0])
            for i in range(16):
                theta=start_theta + 2*math.pi*i/16

                # Goal REGION: preserve the requested angular sector while allowing
                # ARIS to choose radius and altitude. A point that becomes invalid
                # after new obstacle discovery is discarded and another candidate
                # from the same semantic sector is selected.
                sector=[]
                for dr in (0.0,2.5,5.0,7.0,10.0):
                    for dz in ((0.0,) if not allow_altitude else (0.0,3.0,6.0,-2.0,8.0)):
                        z=max(5.0,min(20.0,nominal+dz))
                        r=radius+dr
                        sector.append((center[0]+r*math.cos(theta),center[1]+r*math.sin(theta),z))

                reached_sector = False
                rejected_goals = []
                while sector:
                    await self._discover_3d(drone)
                    current=self._current_xyz(drone)
                    goal=await asyncio.to_thread(
                        self._best_3d_goal, current, list(sector),
                        allow_altitude, nominal
                    )
                    if goal is None:
                        break
                    sector.remove(goal)
                    try:
                        await self._navigate_3d(
                            drone, goal,
                            f'discovered orbit {obj["id"]} {i+1}/16',
                            speed=speed,
                            allow_altitude=allow_altitude,
                            nominal_altitude=nominal
                        )
                        reached_sector = True
                        break
                    except ValueError as exc:
                        rejected_goals.append({"goal": list(goal), "reason": str(exc)})
                        self.replans += 1
                        self.journal(
                            'orbit_sector_goal_reselected',
                            object_id=obj['id'],
                            sector=i+1,
                            rejected_goal=list(goal),
                            reason=str(exc),
                            replan_number=self.replans,
                        )
                        continue

                if not reached_sector:
                    current = self._current_xyz(drone)
                    raise RuntimeError(
                        f'Unable to find a reachable 3D orbit region for discovered target {obj["id"]} '
                        f'from current position {tuple(round(v,2) for v in current)}; '
                        f'{len(sector) + len(rejected_goals)} candidate region points were evaluated'
                    )
            # One documented view proves the discovered target was actually observed.
            viewpoints=self._object_viewpoints(obj,nominal,standoff,'photograph')
        else:
            viewpoints=self._object_viewpoints(obj,nominal,standoff,style)

        for idx,viewpoint in enumerate(viewpoints,start=1):
            x,y,z,side=viewpoint; label=f'{obj["id"]}:{side}'; required_targets.append(label)
            accepted=False
            # Treat a requested view as a REGION, not one brittle coordinate.
            base_angle=math.atan2(y-obj['position'][1],x-obj['position'][0])
            candidates=[]
            for angle_offset in (0,.22,-.22,.45,-.45,.72,-.72):
                for r_extra in (0,2.5,5.0):
                    r=max(standoff,max(obj['size'][:2])/2+3.5)+r_extra
                    for dz in (0,3,6,-2):
                        zz=max(5.0,min(20.0,nominal+dz))
                        th=base_angle+angle_offset
                        candidates.append((obj['position'][0]+r*math.cos(th),
                                           obj['position'][1]+r*math.sin(th),zz,side))
            attempt=0
            while candidates and attempt<18:
                await self._discover_3d(drone)
                cur=self._current_xyz(drone)
                xyz=[c[:3] for c in candidates]
                goal=await asyncio.to_thread(
                    self._best_3d_goal, cur, xyz, allow_altitude, nominal
                )
                if goal is None: break
                candidate=next(c for c in candidates if c[:3]==goal); candidates.remove(candidate); attempt+=1
                try:
                    await self._navigate_3d(drone,goal,f'discovered inspection {label}',speed=speed,
                                            allow_altitude=allow_altitude,nominal_altitude=nominal)
                except ValueError:
                    continue
                capture,visible=await self._capture_discovered_object(drone,obj,candidate,run_id,attempt,label)
                if visible:
                    captures.append({'target':label,'image_path':capture['image_path'],'attempt':attempt})
                    accepted=True; break
                self.replans+=1
                self.journal('evidence_replan',target=label,replan_number=self.replans,
                             reason='discovered object not sufficiently visible',
                             action='rescore 3D goal region')
            if not accepted:
                raise RuntimeError(f'Unable to obtain clear evidence for discovered target {label}')

    async def _find_and_inspect(self, action, drone, run_id, captures, required_targets):
        altitude=float(action.altitude_m or 9.0)
        speed=float(action.speed_mps or DEFAULT_SPEED_MPS)
        self.journal(
            "clean_slate_search_started", object_class=action.object_class,
            selection=action.object_selection, inspection_style=action.discovered_inspection_style,
            note="No target coordinates were supplied to ARIS",
        )
        selected = await self._search_objects(
            drone, action.object_class, action.object_selection, altitude, speed,
            allow_altitude=(action.altitude_policy != "fixed")
        )
        self.journal(
            "clean_slate_targets_selected",
            targets=[{"id":o["id"],"class":o["class"],"position":o["position"]} for o in selected],
            selection=action.object_selection,
        )
        for obj in selected:
            await self._inspect_discovered_object(action,obj,drone,run_id,captures,required_targets)

    def _candidates(self, corner: Corner, altitude=9.0, standoff=7.5):
        cx, cy = self.CORNERS[corner]
        norm = math.hypot(cx, cy)
        outward = (cx / norm, cy / norm)
        candidates = []
        for degrees in (0, 20, -20, 40, -40, 62, -62, 82, -82):
            ang = math.radians(degrees)
            ux = outward[0] * math.cos(ang) - outward[1] * math.sin(ang)
            uy = outward[0] * math.sin(ang) + outward[1] * math.cos(ang)
            candidates.append((cx + ux * standoff, cy + uy * standoff, altitude))
        return candidates

    async def _discover(self, drone, altitude):
        sensed = await drone.sense_obstacles(range_m=10.0, altitude_m=altitude)
        new = self.planner.add_obstacles(sensed)
        for name in new:
            if name == "primary_asset_keepout":
                continue
            self.discovered.add(name)
            obstacle = next(o for o in sensed if o["name"] == name)
            self.journal("obstacle_discovered", obstacle=name, obstacle_geometry=obstacle)
        return new

    def _route_still_clear(self, current, route, altitude):
        if not route:
            return False
        pts = [current] + list(route)
        return all(self.planner._segment_clear(x, y, altitude) for x, y in zip(pts, pts[1:]))

    async def _navigate(self, drone, goal_xy, altitude, label, speed=2.0):
        """
        Follow one stable route and replan only when newly discovered geometry
        actually invalidates it. The old implementation recomputed A* every few
        metres, which made the aircraft visibly zig-zag as grid quantization changed.
        """
        route = None
        route_cursor = 0
        sensor_range = 10.0
        max_step = 5.5  # stays comfortably inside the sensor horizon

        while True:
            if self.stop_event.is_set():
                raise RuntimeError("Operator requested abort")

            current = tuple(drone.world.state()["position"][:2])
            if math.dist(current, goal_xy) < 0.65:
                # Final scan makes subsequent visibility scoring aware of nearby obstacles.
                await self._discover(drone, altitude)
                return

            newly_discovered = await self._discover(drone, altitude)

            needs_plan = route is None or route_cursor >= len(route)
            if newly_discovered and route is not None:
                remaining = route[route_cursor:]
                if not self._route_still_clear(current, remaining, altitude):
                    needs_plan = True

            if needs_plan:
                planned = await asyncio.to_thread(self.planner.plan, current, goal_xy, altitude)
                route = planned[1:]  # first point is current
                route_cursor = 0
                if newly_discovered:
                    self.replans += 1
                    self.journal(
                        "route_replanned", reason="new_obstacle_blocks_active_route",
                        replan_number=self.replans, route_points=len(planned), goal=goal_xy,
                        known_obstacles=sorted(self.planner.obstacles), label=label,
                    )
                else:
                    self.journal("route_planned", route_points=len(planned), goal=goal_xy, label=label)

            if not route:
                return

            waypoint = route[route_cursor]
            d = math.dist(current, waypoint)
            if d <= 0.65:
                route_cursor += 1
                continue

            # Travel along the existing segment rather than asking A* for a new grid
            # direction every step. This keeps the visible path smooth.
            if d > max_step:
                ratio = max_step / d
                target = (
                    current[0] + (waypoint[0] - current[0]) * ratio,
                    current[1] + (waypoint[1] - current[1]) * ratio,
                )
            else:
                target = waypoint

            heading = self._heading(current, target)
            self.heading_deg = heading
            await drone.go_to_local(target[0], target[1], altitude, heading, speed, label)

            if math.dist(target, waypoint) < 0.25:
                route_cursor += 1

    def _route_cost(self, start, goal, altitude):
        try:
            route = self.planner.plan(start, goal, altitude)
            return sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        except ValueError:
            return float("inf")

    def _ordered_corners(self, requested, strategy, current, altitude):
        requested = list(requested)
        if strategy == "given_order":
            return requested
        ordered, pos, remaining = [], current, set(requested)
        while remaining:
            best = min(
                remaining,
                key=lambda c: min(
                    self._route_cost(pos, candidate[:2], altitude)
                    for candidate in self._candidates(c, altitude)
                ),
            )
            ordered.append(best)
            pos = self.CORNERS[best]
            remaining.remove(best)
        return ordered

    def _best_candidates(self, corner, current, altitude, standoff):
        scored = []
        for c in self._candidates(corner, altitude, standoff):
            cost = self._route_cost(current, c[:2], altitude)
            if math.isfinite(cost):
                scored.append((cost, c))
        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored]

    async def _inspect(self, action: MissionAction, drone, run_id, captures):
        altitude = float(action.altitude_m or 9.0)
        standoff = float(action.standoff_m or 7.5)
        speed = float(action.speed_mps or DEFAULT_SPEED_MPS)
        current = tuple(drone.world.state()["position"][:2])
        corners = await asyncio.to_thread(
            self._ordered_corners, action.corners, action.corner_strategy, current, altitude
        )
        self.journal(
            "inspection_sequence_resolved",
            requested_corners=action.corners,
            strategy=action.corner_strategy,
            resolved_order=corners,
            altitude_m=altitude,
            standoff_m=standoff,
        )

        for corner in corners:
            accepted = False
            current = tuple(drone.world.state()["position"][:2])
            candidates = await asyncio.to_thread(
                self._best_candidates, corner, current, altitude, standoff
            )
            if not candidates:
                raise RuntimeError(f"No currently safe generated viewpoint for {corner}")

            for attempt, candidate in enumerate(candidates, start=1):
                self.journal(
                    "viewpoint_generated", corner=corner, attempt=attempt,
                    viewpoint=candidate,
                    source="ARIS geometry + current discovered world model",
                )
                try:
                    await self._navigate(drone, candidate[:2], altitude,
                                         f"adaptive route to {corner}", speed=speed)
                except ValueError as exc:
                    self.replans += 1
                    self.journal("viewpoint_rejected", corner=corner, attempt=attempt, reason=str(exc))
                    continue

                corner_xy = self.CORNERS[corner]
                current = tuple(drone.world.state()["position"][:2])
                heading = self._heading(current, corner_xy)
                self.heading_deg = heading
                pitch = self._pitch(math.dist(current, corner_xy), altitude)
                await drone.orient_camera_local(heading, pitch, f"frame {corner}")
                capture = await drone.capture_corner(run_id, corner, attempt)
                assessment = await asyncio.to_thread(
                    self.evidence.assess, capture["image_path"], corner
                )
                self.journal(
                    "corner_capture", corner=corner, attempt=attempt,
                    accepted=assessment["accepted"], file=Path(capture["image_path"]).name,
                    assessment=assessment,
                )
                if assessment["accepted"]:
                    captures.append({"corner": corner, "image_path": capture["image_path"], "attempt": attempt})
                    accepted = True
                    break

                self.replans += 1
                self.journal(
                    "evidence_replan", corner=corner, replan_number=self.replans,
                    reason=assessment["reason"], action="generate_and_score_new_viewpoint",
                )
            if not accepted:
                raise RuntimeError(f"Unable to obtain verified clear evidence for {corner}")

    @staticmethod
    def _segment_rect_interval(a, b, min_x, max_x, min_y, max_y):
        """Return XY segment parameter interval inside an axis-aligned rectangle."""
        t0, t1 = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        for p, q0, q1 in ((dx, min_x - a[0], max_x - a[0]),
                          (dy, min_y - a[1], max_y - a[1])):
            if abs(p) < 1e-9:
                if q0 > 0 or q1 < 0:
                    return None
                continue
            lo, hi = q0 / p, q1 / p
            if lo > hi:
                lo, hi = hi, lo
            t0, t1 = max(t0, lo), min(t1, hi)
            if t0 > t1:
                return None
        return (max(0.0, t0), min(1.0, t1))

    def _visual_line_clear(self, camera_xyz, look_xyz):
        """Visibility against ARIS's currently discovered obstacle map only."""
        a = camera_xyz
        b = look_xyz
        for name, obstacle in self.planner.obstacles.items():
            if name == "primary_asset_keepout":
                continue  # the target surface belongs to the inspected asset itself
            px, py = obstacle["position"][:2]
            sx, sy = obstacle["size"][:2]
            interval = self._segment_rect_interval(
                a, b, px - sx/2, px + sx/2, py - sy/2, py + sy/2
            )
            if interval is None:
                continue
            # Test the highest line-of-sight altitude while crossing the obstacle.
            z0 = a[2] + (b[2] - a[2]) * interval[0]
            z1 = a[2] + (b[2] - a[2]) * interval[1]
            if min(z0, z1) <= float(obstacle.get("top_z", 999.0)) + 0.25:
                return False
        return True

    def _surface_candidates(self, target: Surface, altitude=9.0, standoff=7.5):
        if target == "roof":
            z = max(float(altitude), 8.0)
            return [(-7.0, -7.0, z), (7.0, -7.0, z), (7.0, 7.0, z), (-7.0, 7.0, z)]

        look_x, look_y, _ = self.SURFACE_LOOK_AT[target]
        nx, ny = self.SURFACE_NORMAL[target]
        tx, ty = -ny, nx

        base = float(standoff)
        radii = []
        for r in (base, max(5.5, base - 1.5), base + 1.5, base + 3.0, base + 4.5):
            if 5.0 <= r <= 14.0 and all(abs(r - x) > 0.2 for x in radii):
                radii.append(r)

        # Broad lateral fan is essential in the complex site. The north/south walls
        # intentionally make straight-on facade views impossible at 9 m; oblique views
        # around the wall ends are the valid solution.
        laterals = (0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0, -10.0, 10.0)
        candidates = []
        for radius in radii:
            for lateral in laterals:
                candidates.append((
                    look_x + nx * radius + tx * lateral,
                    look_y + ny * radius + ty * lateral,
                    float(altitude),
                ))
        return candidates

    def _surface_candidate_score(self, start, target, candidate, altitude, standoff):
        route_cost = self._route_cost(start, candidate[:2], candidate[2])
        if not math.isfinite(route_cost):
            return float("inf")

        look = self.SURFACE_LOOK_AT[target]
        if not self._visual_line_clear(candidate, look):
            return float("inf")

        nx, ny = self.SURFACE_NORMAL[target] if target != "roof" else (0.0, 0.0)
        if target == "roof":
            angle_penalty = 0.0
            standoff_penalty = 0.0
        else:
            vx, vy = candidate[0] - look[0], candidate[1] - look[1]
            dist = max(1e-6, math.hypot(vx, vy))
            cosang = max(-1.0, min(1.0, (vx * nx + vy * ny) / dist))
            angle_deg = math.degrees(math.acos(cosang))
            if angle_deg > 67.0:
                return float("inf")
            angle_penalty = angle_deg * 0.055
            normal_distance = abs(vx * nx + vy * ny)
            standoff_penalty = abs(normal_distance - float(standoff)) * 0.20

        return route_cost + angle_penalty + standoff_penalty

    def _best_surface_candidate(self, start, target, altitude, standoff, rejected=None):
        rejected = rejected or set()
        scored = []
        for candidate in self._surface_candidates(target, altitude, standoff):
            key = (round(candidate[0], 2), round(candidate[1], 2), round(candidate[2], 2))
            if key in rejected:
                continue
            score = self._surface_candidate_score(start, target, candidate, altitude, standoff)
            if math.isfinite(score):
                scored.append((score, candidate))
        scored.sort(key=lambda item: item[0])
        return scored[0][1] if scored else None

    def _surface_route_cost(self, start, target, altitude, standoff):
        candidate = self._best_surface_candidate(start, target, altitude, standoff)
        if candidate is None:
            return float("inf")
        return self._surface_candidate_score(start, target, candidate, altitude, standoff)

    def _ordered_surfaces(self, requested, strategy, current, altitude, standoff):
        # given_order is an explicit operator request. nearest_safe is intentionally
        # resolved one target at a time during _inspect_asset as the world model grows.
        return list(requested)

    def _next_surface(self, remaining, strategy, current, altitude, standoff):
        if strategy == "given_order":
            return remaining[0]
        scored = []
        for target in remaining:
            cost = self._surface_route_cost(current, target, altitude, standoff)
            if math.isfinite(cost):
                scored.append((cost, target))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    async def _inspect_asset(self, action: MissionAction, drone, run_id, captures):
        altitude = float(action.altitude_m or 9.0)
        standoff = float(action.standoff_m or 7.5)
        speed = float(action.speed_mps or DEFAULT_SPEED_MPS)
        remaining = list(action.inspection_targets or ALL_SIDES)

        self.journal(
            "asset_coverage_sequence_resolved",
            coverage_mode=action.coverage_mode,
            requested_targets=action.inspection_targets,
            strategy=action.coverage_strategy,
            altitude_m=altitude,
            standoff_m=standoff,
            note="nearest_safe order is resolved at runtime from the growing world model",
        )

        while remaining:
            current = tuple(drone.world.state()["position"][:2])
            target = await asyncio.to_thread(
                self._next_surface, remaining, action.coverage_strategy,
                current, altitude, standoff
            )
            if target is None:
                raise RuntimeError(
                    "No safe, visible viewpoint remains for required surfaces: " + ", ".join(remaining)
                )

            self.journal(
                "coverage_target_selected",
                target=target,
                remaining_targets=list(remaining),
                strategy=action.coverage_strategy,
                known_obstacles=sorted(self.planner.obstacles),
            )

            accepted = False
            rejected = set()
            max_attempts = 14

            for attempt in range(1, max_attempts + 1):
                current = tuple(drone.world.state()["position"][:2])
                candidate = await asyncio.to_thread(
                    self._best_surface_candidate, current, target, altitude,
                    standoff, rejected
                )
                if candidate is None:
                    break

                key = (round(candidate[0], 2), round(candidate[1], 2), round(candidate[2], 2))
                self.journal(
                    "coverage_viewpoint_generated",
                    target=target,
                    attempt=attempt,
                    viewpoint=candidate,
                    source="ARIS route + visibility + discovered-world scoring",
                )

                try:
                    await self._navigate(
                        drone, candidate[:2], candidate[2],
                        f"adaptive coverage route to {target}", speed=speed
                    )
                except ValueError as exc:
                    rejected.add(key)
                    self.replans += 1
                    self.journal(
                        "coverage_viewpoint_rejected",
                        target=target, attempt=attempt,
                        reason=str(exc), replan_number=self.replans,
                    )
                    continue

                # The trip may have discovered geometry that was unknown when this
                # viewpoint was scored. Re-evaluate visibility before wasting a capture.
                await self._discover(drone, candidate[2])
                look = self.SURFACE_LOOK_AT[target]
                actual = tuple(drone.world.state()["position"][:2]) + (candidate[2],)
                if not self._visual_line_clear(actual, look):
                    rejected.add(key)
                    self.replans += 1
                    self.journal(
                        "coverage_viewpoint_rejected",
                        target=target, attempt=attempt,
                        reason="discovered obstacle blocks camera line of sight",
                        replan_number=self.replans,
                    )
                    continue

                current = tuple(drone.world.state()["position"][:2])
                look_x, look_y, look_z = look
                heading = self._heading(current, (look_x, look_y))
                self.heading_deg = heading
                horizontal = math.dist(current, (look_x, look_y))
                pitch = -math.degrees(
                    math.atan2(max(.1, candidate[2] - look_z), max(.5, horizontal))
                )
                if target == "roof":
                    pitch = min(pitch, -55.0)

                await drone.orient_camera_local(heading, pitch, f"frame {target}")
                capture = await drone.capture_corner(run_id, target, attempt)
                assessment = await asyncio.to_thread(
                    self.surface_evidence.assess, capture["image_path"], target
                )
                self.journal(
                    "surface_capture", target=target, attempt=attempt,
                    accepted=assessment["accepted"], file=Path(capture["image_path"]).name,
                    detected_ids=assessment.get("detected_ids", []),
                    target_markers=assessment.get("target_markers", []),
                    matched_target_markers=assessment.get("matched_target_markers", []),
                    assessment=assessment,
                )

                if assessment["accepted"]:
                    captures.append({
                        "target": target,
                        "image_path": capture["image_path"],
                        "attempt": attempt,
                    })
                    accepted = True
                    break

                rejected.add(key)
                self.replans += 1
                self.journal(
                    "evidence_replan", target=target, replan_number=self.replans,
                    reason=assessment["reason"],
                    detected_ids=assessment.get("detected_ids", []),
                    action="rescore wider surface viewpoint set",
                )

            if not accepted:
                raise RuntimeError(
                    f"Unable to obtain verified clear evidence for {target} after {len(rejected)} distinct viewpoints"
                )

            remaining.remove(target)

    def _validate_intent(self, intent: NaturalLanguageMission):
        for action in intent.actions:
            if action.altitude_m is not None and action.altitude_m > MAX_ALTITUDE_M:
                raise ValueError(f"SafetyGate: altitude {action.altitude_m:g} m exceeds {MAX_ALTITUDE_M:g} m")
            if action.speed_mps is not None and action.speed_mps > MAX_SPEED_MPS:
                raise ValueError(f"SafetyGate: speed {action.speed_mps:g} m/s exceeds {MAX_SPEED_MPS:g} m/s")
            if action.duration_s is not None and action.duration_s > MAX_DURATION_S:
                raise ValueError(f"SafetyGate: duration {action.duration_s:g} s exceeds {MAX_DURATION_S:g} s")
            if action.kind == "move" and action.distance_m is not None and action.distance_m > MAX_MOVE_DISTANCE_M:
                raise ValueError(f"SafetyGate: move distance {action.distance_m:g} m exceeds {MAX_MOVE_DISTANCE_M:g} m")
            if action.kind == "orbit":
                if action.radius_m is not None and action.radius_m > MAX_ORBIT_RADIUS_M:
                    raise ValueError(f"SafetyGate: orbit radius {action.radius_m:g} m exceeds {MAX_ORBIT_RADIUS_M:g} m")
                if action.laps is not None and action.laps > MAX_LAPS:
                    raise ValueError(f"SafetyGate: orbit laps {action.laps} exceeds {MAX_LAPS}")
        return True

    def _orbit_goal_candidates(self, center, angle_rad, requested_radius):
        base = max(
            float(requested_radius),
            7.5 if center == self.NAMED_TARGETS["asset_center"] else float(requested_radius),
        )
        radii = [base, base + 1.5, base + 3.0, base + 4.5, base + 6.0]
        return [
            (center[0] + r * math.cos(angle_rad), center[1] + r * math.sin(angle_rad), r)
            for r in radii
        ]

    async def _navigate_orbit_waypoint(self, drone, center, theta, requested_radius, altitude, speed, label):
        last_error = None
        for candidate_index, (gx, gy, actual_radius) in enumerate(
            self._orbit_goal_candidates(center, theta, requested_radius), start=1
        ):
            try:
                await asyncio.to_thread(
                    self.planner.plan, tuple(drone.world.state()["position"][:2]),
                    (gx, gy), altitude
                )
                await self._navigate(
                    drone, (gx, gy), altitude,
                    f"{label} · r={actual_radius:.1f}m",
                    speed=speed,
                )
                if candidate_index > 1:
                    self.replans += 1
                    self.journal(
                        "orbit_waypoint_replanned",
                        reason="nominal_orbit_point_blocked",
                        replan_number=self.replans,
                        replacement_goal=[gx, gy],
                        replacement_radius_m=actual_radius,
                        label=label,
                    )
                return (gx, gy)
            except ValueError as exc:
                last_error = exc
                self.journal(
                    "orbit_waypoint_blocked",
                    attempted_goal=[gx, gy],
                    attempted_radius_m=actual_radius,
                    reason=str(exc),
                    label=label,
                )
                continue
        raise ValueError(f"No safe adaptive orbit waypoint found: {last_error}")

    async def run(self, intent: NaturalLanguageMission, drone, run_id: str, scene_variant: str):
        self._validate_intent(intent)
        captures, requested_corners, requested_targets = [], [], []
        self.journal("mission_status", status="VALIDATED", mission_kind="adaptive_inspection")
        await drone.connect()
        caps: DroneCapabilities = drone.capabilities()
        self.journal("adapter_capabilities", profile=caps.model_dump(mode="json"))
        required = ["local_xy_motion", "heading_control", "gimbal_control", "still_capture", "obstacle_sensing"]
        if any(a.kind == "find_and_inspect" for a in intent.actions):
            required.append("object_detection")
        missing = [name for name in required if not getattr(caps, name)]
        if missing:
            raise RuntimeError("Connected adapter lacks adaptive-inspection capabilities: " + ", ".join(missing))

        try:
            airborne, current_alt = False, 0.0
            self.journal("mission_status", status="RUNNING", mission_kind="adaptive_inspection")

            for action_index, action in enumerate(intent.actions):
                for repetition in range(action.repeat):
                    self.journal(
                        "mission_ir_action", action_index=action_index,
                        repeat_index=repetition, action=action.model_dump(mode="json"),
                    )

                    if action.kind == "takeoff":
                        await drone.takeoff(float(action.altitude_m))
                        airborne, current_alt = True, float(action.altitude_m)

                    elif action.kind == "move":
                        if not airborne:
                            raise RuntimeError("Mission IR requested movement before takeoff")
                        speed = float(action.speed_mps or DEFAULT_SPEED_MPS)
                        heading = (float(action.heading_deg) if action.heading_deg is not None
                                   else self._direction_heading(self.heading_deg, action.direction)) % 360
                        distance = (float(action.distance_m) if action.distance_m is not None
                                    else speed * float(action.duration_s))
                        current = tuple(drone.world.state()["position"][:2])
                        goal = self._xy_from_heading(current, heading, distance)
                        await self._navigate(drone, goal, current_alt,
                                             f"Mission IR move {action_index + 1}", speed=speed)
                        self.heading_deg = heading

                    elif action.kind == "turn":
                        if not airborne:
                            raise RuntimeError("Mission IR requested turn before takeoff")
                        heading = (float(action.heading_deg) if action.heading_deg is not None
                                   else (self.heading_deg + float(action.turn_degrees)) % 360)
                        await drone.turn_to_heading(heading, f"Mission IR turn {action_index + 1}")
                        self.heading_deg = heading

                    elif action.kind == "set_altitude":
                        if not airborne:
                            raise RuntimeError("Mission IR requested altitude change before takeoff")
                        current = tuple(drone.world.state()["position"][:2])
                        current_alt = float(action.altitude_m)
                        await drone.go_to_local(current[0], current[1], current_alt,
                                                self.heading_deg,
                                                float(action.speed_mps or DEFAULT_SPEED_MPS),
                                                f"Mission IR altitude {action_index + 1}")

                    elif action.kind == "hover":
                        if not airborne:
                            raise RuntimeError("Mission IR requested hover before takeoff")
                        await drone.hold()
                        await asyncio.sleep(float(action.duration_s))

                    elif action.kind == "go_to":
                        if not airborne:
                            raise RuntimeError("Mission IR requested destination before takeoff")
                        current = tuple(drone.world.state()["position"][:2])
                        goal = self._target_xy(action.target, current)
                        if action.stop_distance_m:
                            d = math.dist(current, goal)
                            if d > action.stop_distance_m:
                                ratio = (d - action.stop_distance_m) / d
                                goal = (current[0] + (goal[0] - current[0]) * ratio,
                                        current[1] + (goal[1] - current[1]) * ratio)
                        current_alt = float(action.altitude_m or current_alt)
                        await self._navigate(drone, goal, current_alt,
                                             f"Mission IR go_to {action.target}",
                                             speed=float(action.speed_mps or DEFAULT_SPEED_MPS))

                    elif action.kind == "orbit":
                        if not airborne:
                            raise RuntimeError("Mission IR requested orbit before takeoff")
                        center = self._target_xy(action.target, tuple(drone.world.state()["position"][:2]))
                        altitude = float(action.altitude_m or current_alt)
                        radius = float(action.radius_m)
                        sign = -1.0 if action.orbit_direction == "clockwise" else 1.0
                        current = tuple(drone.world.state()["position"][:2])
                        vx, vy = current[0] - center[0], current[1] - center[1]
                        start_angle = math.atan2(vy, vx) if math.hypot(vx, vy) > 0.5 else 0.0
                        total = int(action.laps) * 16
                        for i in range(total + 1):
                            theta = start_angle + sign * 2 * math.pi * (i / 16)
                            await self._navigate_orbit_waypoint(
                                drone, center, theta, radius, altitude,
                                float(action.speed_mps or DEFAULT_SPEED_MPS),
                                f"Mission IR orbit {action.target} {i + 1}/{total + 1}",
                            )
                        current_alt = altitude

                    elif action.kind == "inspect_corners":
                        if not airborne:
                            raise RuntimeError("Mission IR requested inspection before takeoff")
                        requested_corners.extend(action.corners)
                        await self._inspect(action, drone, run_id, captures)
                        current_alt = float(action.altitude_m or current_alt)

                    elif action.kind == "inspect_asset":
                        if not airborne:
                            raise RuntimeError("Mission IR requested asset inspection before takeoff")
                        requested_targets.extend(action.inspection_targets or [])
                        await self._inspect_asset(action, drone, run_id, captures)
                        current_alt = float(action.altitude_m or current_alt)

                    elif action.kind == "find_and_inspect":
                        if not airborne:
                            raise RuntimeError("Mission IR requested clean-slate search before takeoff")
                        await self._find_and_inspect(action, drone, run_id, captures, requested_targets)
                        current_alt = float(action.altitude_m or current_alt)

                    elif action.kind == "return_home":
                        if airborne:
                            if self.used_clean_slate_3d:
                                await self._navigate_3d(
                                    drone,
                                    (self.NAMED_TARGETS["home"][0], self.NAMED_TARGETS["home"][1], current_alt),
                                    "Mission IR return home",
                                    speed=float(action.speed_mps or DEFAULT_SPEED_MPS),
                                    allow_altitude=True,
                                    nominal_altitude=current_alt,
                                )
                            else:
                                await self._navigate(drone, self.NAMED_TARGETS["home"], current_alt,
                                                     "Mission IR return home",
                                                     speed=float(action.speed_mps or DEFAULT_SPEED_MPS))

                    elif action.kind == "land":
                        if airborne:
                            await drone.land()
                            airborne, current_alt = False, 0.0

            if airborne:
                raise RuntimeError("Unified adaptive mission ended while aircraft remained airborne")

            memory = (
                await asyncio.to_thread(self.memory.record, run_id, captures, scene_variant)
                if captures and self.memory else None
            )
            if memory:
                self.journal("asset_memory", **memory)
            self.journal("mission_status", status="COMPLETE", mission_kind="adaptive_inspection")
            try:
                landed_confirmed = not (await drone.telemetry()).airborne
            except Exception:
                landed_confirmed = False
            return {
                "status": "COMPLETE", "captures": captures,
                "required_corners": list(dict.fromkeys(requested_corners)),
                "required_targets": list(dict.fromkeys(requested_targets + requested_corners)),
                "replans": self.replans, "discovered_obstacles": sorted(self.discovered),
                "discovered_objects": list(self.discovered_objects.values()),
                "asset_memory": memory,
                "landed_confirmed": landed_confirmed,
            }
        except Exception as exc:
            self.journal("mission_execution_error", error=type(exc).__name__, message=str(exc)[:300])
            try:
                await drone.abort()
            except Exception:
                pass
            self.journal("mission_status", status="FAILED", mission_kind="adaptive_inspection")
            try:
                landed_confirmed = not (await drone.telemetry()).airborne
            except Exception:
                landed_confirmed = False
            return {
                "status": "FAILED", "error": str(exc), "captures": captures,
                "required_corners": list(dict.fromkeys(requested_corners)),
                "required_targets": list(dict.fromkeys(requested_targets + requested_corners)),
                "replans": self.replans, "discovered_obstacles": sorted(self.discovered),
                "discovered_objects": list(self.discovered_objects.values()),
                "landed_confirmed": landed_confirmed,
            }
        finally:
            await drone.close()
