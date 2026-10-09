import asyncio, uuid
from .models import MissionStatus, Observation, SecurityEvent
from .safety import SafetyViolation


class MissionExecutor:
    def __init__(
        self,
        store,
        safety,
        quality,
        presence_detector,
        completion_engine,
        gap_replanner,
        perception,
    ):
        self.store = store
        self.safety = safety
        self.quality = quality
        self.presence_detector = presence_detector
        self.completion_engine = completion_engine
        self.gap_replanner = gap_replanner
        self.perception = perception
        self.stops = {}

    def request_stop(self, id):
        if id in self.stops:
            self.stops[id].set()
            return True
        return False

    def _status(self, plan, status):
        plan.status = status
        self.store.save_mission(plan)

    async def _guarded(self, plan, asset, drone, operation, timeout=None):
        task = asyncio.create_task(operation)
        deadline = asyncio.get_running_loop().time() + (
            timeout or self.safety.cfg.operation_timeout_s
        )
        try:
            while True:
                if self.stops[plan.id].is_set():
                    raise SafetyViolation("Operator requested abort")
                t = await asyncio.wait_for(
                    drone.telemetry(), min(3, self.safety.cfg.operation_timeout_s)
                )
                self.safety.check_runtime(asset, t, plan.max_altitude_m)
                if task.done():
                    return await task
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Operation timed out")
                await asyncio.wait(
                    {task}, timeout=min(self.safety.cfg.watchdog_interval_s, remaining)
                )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _capture_task(self, plan, asset, drone, task, attempt):
        t = await drone.telemetry()
        self.safety.check_runtime(asset, t, plan.max_altitude_m)
        self.safety.validate_task(asset, task)
        self.safety.validate_segment(asset, t, task, min(t.altitude_m, task.altitude_m))
        await self._guarded(plan, asset, drone, drone.go_to(task, plan.speed_mps))
        arrived = await drone.telemetry()
        self.safety.validate_arrival(arrived, task)
        if task.settle_seconds:
            await self._guarded(plan, asset, drone, asyncio.sleep(task.settle_seconds))
        capture = await self._guarded(
            plan, asset, drone, drone.capture_image(plan.id, task, attempt)
        )
        if (capture.source == "simulated_sensor") != (plan.adapter == "physics"):
            raise SafetyViolation("Simulation evidence domain does not match mission")
        self.safety.validate_arrival(capture.telemetry, task)
        if capture.source in {"dji", "parrot", "px4"}:
            self.safety.check_runtime(asset, capture.telemetry, plan.max_altitude_m)
        req = next(
            r
            for r in plan.mission_spec.evidence_requirements
            if r.id == task.requirement_id
        )
        q = await self._guarded(
            plan,
            asset,
            drone,
            asyncio.to_thread(self.quality.assess, capture.image_path),
        )
        # Never spend a cloud call pretending a synthetic simulator image is inspection evidence.
        from .perception import Assessment

        if capture.source == "synthetic":
            a = Assessment(
                status="unavailable",
                provider="synthetic",
                image_sha256=q.sha256,
                issues=["synthetic"],
                explanation="Synthetic image cannot satisfy real inspection evidence.",
            )
        elif not q.passed:
            a = Assessment(
                status="unavailable",
                provider="technical_quality_gate",
                image_sha256=q.sha256,
                issues=q.issues,
                explanation="Technical image quality failed; semantic/presence assessment not performed.",
            )
        else:
            a = await self._guarded(
                plan,
                asset,
                drone,
                self.perception.assess(capture.image_path, req, asset),
            )
        accepted = (
            q.passed
            and q.score >= req.min_quality
            and a.status == "assessed"
            and a.target_visible is True
            and a.criterion_met is True
            and a.image_sha256 == q.sha256
            and capture.source != "synthetic"
        )
        present = (
            self.presence_detector.detect(a) if plan.security_presence_logging else None
        )
        finding = (
            "; ".join(a.anomalies) if a.status == "assessed" and a.anomalies else None
        )
        obs = Observation(
            id=str(uuid.uuid4()),
            mission_id=plan.id,
            task_id=task.id,
            requirement_id=req.id,
            asset_id=plan.asset_id,
            label=task.label,
            lat=capture.telemetry.lat,
            lon=capture.telemetry.lon,
            altitude_m=capture.telemetry.altitude_m,
            captured_at=capture.captured_at.isoformat(),
            image_path=capture.image_path,
            image_sha256=q.sha256,
            source=capture.source,
            quality_score=q.score,
            quality_passed=q.passed,
            quality_issues=q.issues,
            accepted=accepted,
            attempt=attempt,
            assessment=a.model_dump(),
            finding=finding,
            person_present=present,
        )
        self.store.save_observation(obs)
        if present is True:
            self.store.save_security_event(
                SecurityEvent(
                    id=str(uuid.uuid4()),
                    mission_id=plan.id,
                    asset_id=plan.asset_id,
                    observed_at=obs.captured_at,
                    observation_label=task.label,
                    lat=obs.lat,
                    lon=obs.lon,
                    evidence_image_path=obs.image_path,
                    evidence_sha256=obs.image_sha256,
                    provider=a.provider,
                )
            )
        return obs

    def _evaluate(self, plan):
        state = self.completion_engine.evaluate(
            plan.mission_spec,
            [o for o in self.store.observations_for_mission(plan.id)
             if (o.source == "simulated_sensor") == (plan.adapter == "physics")],
            self.store.security_events_for_mission(plan.id),
        )
        self.store.save_completion_state(plan.id, state)
        return state

    async def _recover(self, plan, drone):
        try:
            # Gateway abort must independently apply aircraft-local failsafe and confirm landing.
            await asyncio.wait_for(drone.abort(), self.safety.cfg.recovery_timeout_s)
            t = await asyncio.wait_for(drone.telemetry(), 3)
            self.safety.validate_fresh(t)
            if t.airborne:
                raise RuntimeError("Landing not confirmed")
            plan.landed_confirmed = True
        except Exception as e:
            plan.recovery_error = f"Recovery not confirmed ({type(e).__name__}); local operator action required."

    async def run(self, plan, drone, claimed=False):
        if not claimed:
            plan = self.store.claim(plan.id)
        stop = asyncio.Event()
        self.stops[plan.id] = stop
        takeoff_attempted = False
        try:
            async with asyncio.timeout(self.safety.cfg.total_mission_timeout_s):
                asset = self.store.get_asset(plan.asset_id)
                if not asset:
                    raise ValueError("Missing asset")
                self.safety.validate_plan(asset, plan)
                await asyncio.wait_for(drone.connect(), getattr(drone, "connection_timeout_s", 10))
                home = await asyncio.wait_for(drone.telemetry(), 3)
                self.safety.validate_preflight(asset, home)
                self.safety.validate_segment(
                    asset, home, plan.tasks[0], plan.tasks[0].altitude_m
                )
                # Validate return corridors from every potential observation point before takeoff.
                for task in plan.tasks:
                    self.safety.validate_segment(asset, task, home, task.altitude_m)
                if hasattr(drone, "authorize"):
                    await asyncio.wait_for(drone.authorize(plan, asset), 60)
                self._status(plan, MissionStatus.VALIDATED)
                takeoff_attempted = True
                await self._guarded(
                    plan, asset, drone, drone.takeoff(plan.tasks[0].altitude_m)
                )
                self._status(plan, MissionStatus.RUNNING)
                originals = list(plan.tasks)
                for task in originals:
                    await self._capture_task(plan, asset, drone, task, 1)
                while True:
                    state = self._evaluate(plan)
                    if state.complete:
                        break
                    observations = self.store.observations_for_mission(plan.id)
                    generated = []
                    for req in plan.mission_spec.evidence_requirements:
                        if (
                            req.kind != "visual_coverage"
                            or req.label not in state.unsatisfied_labels
                        ):
                            continue
                        previous = [
                            o for o in observations if o.requirement_id == req.id
                        ]
                        if not previous or len(previous) >= req.max_attempts:
                            continue
                        last = previous[-1]
                        issues = last.quality_issues + last.assessment.get("issues", [])
                        original = next(
                            t for t in originals if t.requirement_id == req.id
                        )
                        retry = self.gap_replanner.supplemental_task(
                            asset, original, len(previous) + 1, issues
                        )
                        if retry:
                            self.safety.validate_task(asset, retry)
                            self.safety.validate_segment(
                                asset, retry, home, retry.altitude_m
                            )
                            generated.append(retry)
                    if not generated:
                        break
                    self._status(plan, MissionStatus.REINSPECTING)
                    for task in generated:
                        if len(plan.tasks) >= self.safety.cfg.max_tasks:
                            raise SafetyViolation("Total retry task limit reached")
                        plan.tasks.append(task)
                        self.store.save_mission(plan)
                        await self._capture_task(
                            plan, asset, drone, task, task.attempts
                        )
                final = self._evaluate(plan)
                self._status(plan, MissionStatus.RETURNING_HOME)
                t = await drone.telemetry()
                self.safety.validate_segment(asset, t, home, t.altitude_m)
                await self._guarded(plan, asset, drone, drone.return_home())
                await self._guarded(plan, asset, drone, drone.land())
                landed = await drone.telemetry()
                self.safety.validate_fresh(landed)
                if landed.airborne:
                    raise RuntimeError("Landing not confirmed")
                plan.landed_confirmed = True
                self._status(
                    plan,
                    MissionStatus.COMPLETE
                    if final.complete
                    else MissionStatus.INCOMPLETE,
                )
        except (Exception, asyncio.CancelledError) as e:
            plan.error = (
                f"{type(e).__name__}: {str(e)[:300]}"
                if isinstance(e, (SafetyViolation, ValueError, TimeoutError))
                else f"{type(e).__name__}: execution failed"
            )
            plan.status = (
                MissionStatus.ABORTED
                if isinstance(e, (SafetyViolation, asyncio.CancelledError))
                else MissionStatus.FAILED
            )
            if takeoff_attempted:
                await self._recover(plan, drone)
            self.store.save_mission(plan)
            self._evaluate(plan)
        finally:
            try:
                await drone.close()
            except Exception:
                pass
            self.stops.pop(plan.id, None)
            # A failed live recovery holds the aircraft lock across process restart.
            if not takeoff_attempted or plan.landed_confirmed or plan.adapter not in {"dji", "parrot", "px4"}:
                self.store.release(plan.id)
        return self.store.get_mission(plan.id)
