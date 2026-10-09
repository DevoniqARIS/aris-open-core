import asyncio
from datetime import datetime, timezone, timedelta
import pytest
from app.models import MissionStatus, LatLon, AssetCreate
from app.safety import SafetyViolation, SafetyConfig
from app.adapters.sim import SimDrone
from app.perception import UnavailablePerception
from tests.helpers import (
    setup,
    asset,
    RecordedFixtureDrone,
    LowQualityFirstDrone,
    FixturePerception,
)


def run(ex, p, drone):
    return asyncio.run(ex.run(p, drone))


def test_real_workflow_with_explicit_test_perception(tmp_path):
    s, a, p, ex = setup(tmp_path)
    r = run(ex, p, RecordedFixtureDrone(runtime=tmp_path))
    assert r.status == MissionStatus.COMPLETE and r.landed_confirmed
    assert len(s.observations_for_mission(p.id)) == 5


def test_default_unavailable_perception_leaves_incomplete(tmp_path):
    s, a, p, ex = setup(tmp_path, perception=UnavailablePerception())
    r = run(ex, p, RecordedFixtureDrone(runtime=tmp_path))
    assert r.status == MissionStatus.INCOMPLETE
    assert not any(o.accepted for o in s.observations_for_mission(p.id))


def test_synthetic_cannot_pass_even_if_test_model_says_yes(tmp_path):
    s, a, p, ex = setup(tmp_path, mode="sim")
    r = run(ex, p, SimDrone(runtime=tmp_path))
    assert r.status == MissionStatus.INCOMPLETE
    assert not any(o.accepted for o in s.observations_for_mission(p.id))


def test_retry_improves_actual_pixels_not_filename(tmp_path):
    s, a, p, ex = setup(tmp_path, prompt="Inspect roof only")
    r = run(ex, p, LowQualityFirstDrone(runtime=tmp_path))
    obs = s.observations_for_mission(p.id)
    assert r.status == MissionStatus.COMPLETE and len(obs) == 2
    assert not obs[0].quality_passed and obs[1].quality_passed


def test_permanent_weak_images_stop_at_max_attempts(tmp_path):
    from PIL import Image

    class Weak(RecordedFixtureDrone):
        async def capture_image(self, *args):
            c = await super().capture_image(*args)
            Image.new("RGB", (640, 480), "gray").save(c.image_path)
            return c

    s, a, p, ex = setup(tmp_path, prompt="Inspect roof only")
    r = run(ex, p, Weak(runtime=tmp_path))
    assert (
        r.status == MissionStatus.INCOMPLETE
        and len(s.observations_for_mission(p.id)) == 3
    )


def test_battery_drop_after_initial_pass_stops_retry(tmp_path):
    class Drop(LowQualityFirstDrone):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.captures = 0
            self.unsafe_moves = 0

        async def capture_image(self, *args):
            c = await super().capture_image(*args)
            self.captures += 1
            if self.captures == 5:
                self.state.battery_percent = 5
            return c

        async def go_to(self, *args):
            if self.state.battery_percent <= 30:
                self.unsafe_moves += 1
            await super().go_to(*args)

    s, a, p, ex = setup(tmp_path)
    drone = Drop(runtime=tmp_path)
    r = run(ex, p, drone)
    assert (
        r.status == MissionStatus.ABORTED
        and r.landed_confirmed
        and drone.unsafe_moves == 0
    )


def test_battery_drop_during_slow_analysis_is_monitored(tmp_path):
    class Slow(FixturePerception):
        async def assess(self, *args):
            await asyncio.sleep(1)
            return await super().assess(*args)

    s, a, p, ex = setup(tmp_path, perception=Slow())
    d = RecordedFixtureDrone(runtime=tmp_path)

    async def go():
        async def drop():
            await asyncio.sleep(0.12)
            d.state.battery_percent = 5

        t = asyncio.create_task(drop())
        r = await ex.run(p, d)
        await t
        return r

    r = asyncio.run(go())
    assert r.status == MissionStatus.ABORTED and r.landed_confirmed


def test_early_return_reaches_terminal_state(tmp_path):
    class Drop(SimDrone):
        async def takeoff(self, alt):
            await super().takeoff(alt)
            self.state.battery_percent = 29

    s, a, p, ex = setup(tmp_path)
    r = run(ex, p, Drop(runtime=tmp_path))
    assert r.status == MissionStatus.ABORTED and r.landed_confirmed


def test_connection_failure_terminal_and_no_takeoff(tmp_path):
    class Broken(SimDrone):
        async def connect(self):
            raise RuntimeError("injected")

    s, a, p, ex = setup(tmp_path)
    r = run(ex, p, Broken(runtime=tmp_path))
    assert r.status == MissionStatus.FAILED
    assert not s._one("SELECT * FROM control_lock", ())


def test_stale_telemetry_aborts_before_takeoff(tmp_path):
    class Stale(SimDrone):
        async def telemetry(self):
            return self.state.model_copy(
                update={
                    "observed_at": datetime.now(timezone.utc) - timedelta(seconds=20)
                }
            )

    s, a, p, ex = setup(tmp_path)
    d = Stale(runtime=tmp_path)
    r = run(ex, p, d)
    assert r.status == MissionStatus.ABORTED and not d.state.airborne


@pytest.mark.parametrize("fault", ["gps", "link", "altitude", "position"])
def test_runtime_faults_abort(tmp_path, fault):
    class Fault(SimDrone):
        async def go_to(self, *args):
            await super().go_to(*args)
            if fault == "gps":
                self.state.gps_ok = False
            elif fault == "link":
                self.state.link_ok = False
            elif fault == "altitude":
                self.state.altitude_m = 45
            else:
                self.state.lat = 50

    s, a, p, ex = setup(tmp_path)
    r = run(ex, p, Fault(runtime=tmp_path))
    assert r.status == MissionStatus.ABORTED


def test_operation_timeout_is_terminal(tmp_path):
    class Stuck(SimDrone):
        async def go_to(self, *args):
            await asyncio.sleep(30)

    s, a, p, ex = setup(
        tmp_path,
        config=SafetyConfig(operation_timeout_s=0.05, watchdog_interval_s=0.005),
    )
    r = run(ex, p, Stuck(runtime=tmp_path))
    assert r.status == MissionStatus.FAILED and r.landed_confirmed


def test_failed_live_recovery_retains_aircraft_lock(tmp_path):
    class BrokenRecovery(SimDrone):
        async def go_to(self, *args):
            raise RuntimeError("movement failed")

        async def abort(self):
            raise RuntimeError("unreachable")

    s, a, p, ex = setup(tmp_path, mode="dji")
    r = run(ex, p, BrokenRecovery(runtime=tmp_path))
    assert (
        r.status == MissionStatus.FAILED and r.recovery_error and not r.landed_confirmed
    )
    assert s._one("SELECT * FROM control_lock", ())


def test_operator_abort_monitored(tmp_path):
    class Slow(SimDrone):
        async def go_to(self, *args):
            await asyncio.sleep(10)

    s, a, p, ex = setup(tmp_path)

    async def go():
        t = asyncio.create_task(ex.run(p, Slow(runtime=tmp_path)))
        await asyncio.sleep(0.06)
        assert ex.request_stop(p.id)
        return await t

    r = asyncio.run(go())
    assert r.status == MissionStatus.ABORTED and r.landed_confirmed


def test_outside_polygon_rejected_even_within_old_radius(tmp_path):
    s, a, p, ex = setup(tmp_path)
    a.flight_boundary = [
        LatLon(lat=48.99995, lon=24.99995),
        LatLon(lat=48.99995, lon=25.00025),
        LatLon(lat=49.00025, lon=25.00025),
        LatLon(lat=49.00025, lon=24.99995),
    ]
    t = p.tasks[0].model_copy(update={"lat": 49.0004, "lon": 25.0001})
    with pytest.raises(SafetyViolation):
        ex.safety.validate_task(a, t)


def test_transit_across_exclusion_rejected(tmp_path):
    s, a, p, ex = setup(tmp_path)
    a.exclusion_zones = [
        [
            LatLon(lat=49.00027, lon=25.00005),
            LatLon(lat=49.00027, lon=25.00015),
            LatLon(lat=49.00029, lon=25.00015),
            LatLon(lat=49.00029, lon=25.00005),
        ]
    ]
    with pytest.raises(SafetyViolation):
        ex.safety.validate_segment(
            a, LatLon(lat=49.00024, lon=25.0001), LatLon(lat=49.00032, lon=25.0001), 18
        )


def test_invalid_polygon_and_nonfinite_coordinates_rejected():
    a = asset().model_dump(exclude={"id"})
    a["boundary"] = [
        {"lat": 49, "lon": 25},
        {"lat": 49.0002, "lon": 25.0002},
        {"lat": 49, "lon": 25.0002},
        {"lat": 49.0002, "lon": 25},
    ]
    with pytest.raises(ValueError):
        AssetCreate.model_validate(a)
    with pytest.raises(ValueError):
        LatLon(lat=float("nan"), lon=25)


def test_restart_marks_interrupted_live_flight_failed_and_keeps_lock(tmp_path):
    s, a, p, ex = setup(tmp_path, mode="dji")
    s.claim(p.id)
    s.recover_interrupted()
    assert s.get_mission(p.id).status == MissionStatus.FAILED
    assert s._one("SELECT * FROM control_lock", ())


def test_occlusion_uses_only_configured_viewpoint_and_revalidates(tmp_path):
    from app.models import ApprovedViewpoint
    from app.gap_replanner import GapReplanner

    s, a, p, ex = setup(tmp_path)
    original = p.tasks[0]
    assert GapReplanner().supplemental_task(a, original, 2, ["occluded"]) is None
    a.alternate_views[original.label] = [
        ApprovedViewpoint(
            lat=original.lat, lon=original.lon + 0.00001, altitude_m=18, heading_deg=180
        )
    ]
    retry = GapReplanner().supplemental_task(a, original, 2, ["occluded"])
    assert retry.lon == original.lon + 0.00001
    ex.safety.validate_task(a, retry)
    a.alternate_views[original.label] = [
        ApprovedViewpoint(lat=50, lon=25, altitude_m=18, heading_deg=180)
    ]
    with pytest.raises(SafetyViolation):
        ex.safety.validate_task(
            a, GapReplanner().supplemental_task(a, original, 2, ["occluded"])
        )


def test_second_controller_cannot_recover_active_first_controller(tmp_path):
    from app.process_lock import ControllerLock

    a = ControllerLock(tmp_path / "control.lock")
    b = ControllerLock(tmp_path / "control.lock")
    a.acquire()
    try:
        with pytest.raises(RuntimeError):
            b.acquire()
    finally:
        a.release()
    b.acquire()
    b.release()
