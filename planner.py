import uuid, math
from .geo import centroid, offset_m, bearing_deg, distance_m
from .models import ObservationTask, MissionPlan, LatLon


class InspectionPlanner:
    def plan_from_spec(self, asset, req, spec):
        c = centroid(asset.boundary)
        north = max(v.lat for v in asset.boundary)
        south = min(v.lat for v in asset.boundary)
        east = max(v.lon for v in asset.boundary)
        west = min(v.lon for v in asset.boundary)
        points = {
            "north_facade": offset_m(LatLon(lat=north, lon=c.lon), 8, 0),
            "east_facade": offset_m(LatLon(lat=c.lat, lon=east), 0, 8),
            "south_facade": offset_m(LatLon(lat=south, lon=c.lon), -8, 0),
            "west_facade": offset_m(LatLon(lat=c.lat, lon=west), 0, -8),
            "roof": c,
        }
        altitude = max(10, asset.height_m + 8)
        tasks = []
        for ev in spec.evidence_requirements:
            if ev.kind != "visual_coverage":
                continue
            if ev.target not in points:
                raise ValueError("Unsupported observation target")
            p = points[ev.target]
            pitch = (
                -90
                if ev.target == "roof"
                else -math.degrees(
                    math.atan2(altitude - asset.height_m / 2, max(1, distance_m(p, c)))
                )
            )
            tasks.append(
                ObservationTask(
                    id=str(uuid.uuid4()),
                    requirement_id=ev.id,
                    lat=p.lat,
                    lon=p.lon,
                    altitude_m=altitude,
                    heading_deg=bearing_deg(p, c),
                    gimbal_pitch_deg=pitch,
                    label=ev.target,
                    required_quality=ev.min_quality,
                )
            )
        return MissionPlan(
            id=str(uuid.uuid4()),
            asset_id=asset.id,
            prompt=req.prompt,
            adapter=req.adapter,
            tasks=tasks,
            ai_summary=spec.summary,
            mission_spec=spec,
            security_presence_logging=spec.security_presence_logging,
        )
