from dataclasses import dataclass
from datetime import datetime, timezone
from .geo import point_allowed, segment_allowed, distance_m


@dataclass(frozen=True)
class SafetyConfig:
    min_altitude_m: float = 10
    max_altitude_m: float = 30
    min_start_battery_percent: float = 50
    return_battery_percent: float = 30
    max_speed_mps: float = 4
    max_tasks: int = 30
    max_telemetry_age_s: float = 3
    watchdog_interval_s: float = 0.1
    operation_timeout_s: float = 60
    total_mission_timeout_s: float = 600
    recovery_timeout_s: float = 60


class SafetyViolation(ValueError):
    pass


class SafetyGate:
    def __init__(self, cfg=None):
        self.cfg = cfg or SafetyConfig()

    def validate_task(self, asset, task):
        if not self.cfg.min_altitude_m <= task.altitude_m <= self.cfg.max_altitude_m:
            raise SafetyViolation("Altitude outside configured limits")
        if not point_allowed(asset, task):
            raise SafetyViolation(
                "Task outside flight polygon or inside exclusion zone"
            )
        if not segment_allowed(asset, task, task, task.altitude_m):
            raise SafetyViolation("Task intersects declared building volume")

    def validate_segment(self, asset, a, b, altitude):
        if not segment_allowed(asset, a, b, altitude):
            raise SafetyViolation(
                "Transit crosses geofence, exclusion zone or declared building volume"
            )

    def validate_plan(self, asset, plan):
        if not plan.tasks or len(plan.tasks) > self.cfg.max_tasks:
            raise SafetyViolation("Invalid task count")
        if not 0 < plan.speed_mps <= self.cfg.max_speed_mps:
            raise SafetyViolation("Invalid speed")
        if len({r.id for r in plan.mission_spec.evidence_requirements}) != len(
            plan.mission_spec.evidence_requirements
        ):
            raise SafetyViolation("Duplicate requirement IDs")
        for t in plan.tasks:
            self.validate_task(asset, t)
            if t.altitude_m > plan.max_altitude_m:
                raise SafetyViolation("Task exceeds mission altitude cap")
        for a, b in zip(plan.tasks, plan.tasks[1:]):
            self.validate_segment(asset, a, b, min(a.altitude_m, b.altitude_m))

    def validate_fresh(self, t):
        age = (datetime.now(timezone.utc) - t.observed_at).total_seconds()
        if age > self.cfg.max_telemetry_age_s or age < -1:
            raise SafetyViolation("Stale or future telemetry")

    def validate_preflight(self, asset, t):
        self.validate_fresh(t)
        if t.airborne:
            raise SafetyViolation("Aircraft already airborne")
        if not t.gps_ok or not t.link_ok:
            raise SafetyViolation("GPS or link unhealthy")
        if t.battery_percent < self.cfg.min_start_battery_percent:
            raise SafetyViolation("Battery below launch threshold")
        if not point_allowed(asset, t):
            raise SafetyViolation(
                "Launch point outside flight polygon or in excluded area"
            )
        # A takeoff through the building footprint is invalid even if cruise height clears its roof.
        if not segment_allowed(asset, t, t, 0):
            raise SafetyViolation("Launch point intersects declared building footprint")

    def check_runtime(self, asset, t, altitude_cap=None):
        self.validate_fresh(t)
        if not t.link_ok:
            raise SafetyViolation("Aircraft link lost")
        if not t.gps_ok:
            raise SafetyViolation("GPS unhealthy")
        if t.battery_percent <= self.cfg.return_battery_percent:
            raise SafetyViolation("Battery at return threshold")
        if t.altitude_m > min(
            self.cfg.max_altitude_m, altitude_cap or self.cfg.max_altitude_m
        ):
            raise SafetyViolation("Altitude limit exceeded")
        if not point_allowed(asset, t):
            raise SafetyViolation(
                "Aircraft left flight polygon or entered exclusion zone"
            )
        if t.airborne and not segment_allowed(asset, t, t, t.altitude_m):
            raise SafetyViolation("Aircraft entered declared building volume")

    def validate_arrival(self, t, task):
        if distance_m(t, task) > 2 or abs(t.altitude_m - task.altitude_m) > 1.5:
            raise SafetyViolation("Capture point not reached")
