import math
from shapely.geometry import Polygon, Point, LineString
from .models import LatLon

EARTH_M = 6371000.0


def polygon(points):
    p = Polygon([(v.lon, v.lat) for v in points])
    if not p.is_valid or p.is_empty or p.area < 1e-14:
        raise ValueError("Invalid, degenerate or self-intersecting polygon")
    if p.bounds[2] - p.bounds[0] > 0.05 or p.bounds[3] - p.bounds[1] > 0.05:
        raise ValueError("This prototype supports local inspection areas only")
    return p


def centroid(points):
    c = polygon(points).centroid
    return LatLon(lat=c.y, lon=c.x)


def distance_m(a, b):
    a1, b1 = math.radians(a.lat), math.radians(b.lat)
    h = (
        math.sin((b1 - a1) / 2) ** 2
        + math.cos(a1) * math.cos(b1) * math.sin(math.radians(b.lon - a.lon) / 2) ** 2
    )
    return 2 * EARTH_M * math.asin(math.sqrt(min(1, h)))


def offset_m(origin, north_m, east_m):
    return LatLon(
        lat=origin.lat + math.degrees(north_m / EARTH_M),
        lon=origin.lon
        + math.degrees(east_m / (EARTH_M * math.cos(math.radians(origin.lat)))),
    )


def bearing_deg(a, b):
    a1, b1 = math.radians(a.lat), math.radians(b.lat)
    dl = math.radians(b.lon - a.lon)
    return (
        math.degrees(
            math.atan2(
                math.sin(dl) * math.cos(b1),
                math.cos(a1) * math.sin(b1)
                - math.sin(a1) * math.cos(b1) * math.cos(dl),
            )
        )
        + 360
    ) % 360


def point_allowed(asset, p):
    q = Point(p.lon, p.lat)
    return polygon(asset.flight_boundary).covers(q) and not any(
        polygon(z).intersects(q) for z in asset.exclusion_zones
    )


def segment_allowed(asset, a, b, altitude):
    if not point_allowed(asset, a) or not point_allowed(asset, b):
        return False
    if a.lat == b.lat and a.lon == b.lon:
        line = Point(a.lon, a.lat)
    else:
        line = LineString([(a.lon, a.lat), (b.lon, b.lat)])
    if not polygon(asset.flight_boundary).covers(line):
        return False
    if any(polygon(z).intersects(line) for z in asset.exclusion_zones):
        return False
    # Footprint is also an obstacle below the user-declared building height + margin.
    return altitude >= asset.height_m + 3 or not polygon(asset.boundary).intersects(
        line
    )
