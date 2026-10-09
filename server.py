import asyncio, hashlib, io, json, os, secrets, threading, uuid, zipfile, traceback, time
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal

from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.models import MissionPlanRequest, MissionStatus
from app.mission_interpreter import DeterministicMissionInterpreter, GPTMissionInterpreter
from app.planner import InspectionPlanner
from app.executor import MissionExecutor
from app.store import Store
from app.safety import SafetyGate
from app.quality import QualityInspector
from app.security import HumanPresenceDetector
from app.completion_engine import CompletionEngine
from app.evidence_verifier import EvidenceVerifier
from app.gap_replanner import GapReplanner
from app.reporting import build_report
from app.perception import OpenAIPerception
from app.ai_client import ModelUnavailable
from app.adaptive import AdaptiveMissionExecutor, PersistentAssetMemory
from app.maneuver import (
    GPTManeuverInterpreter,
    ManeuverCompiler,
    ManeuverSafetyGate,
    ManeuverExecutor,
    verify_gpt_link,
    ClarificationRequired,
)

from .world import World, scene_asset, HOME, geo
from .adapter import PhysicsDrone
from .perception import CalibrationVision, JournalPerception

ROOT = Path(__file__).resolve().parents[1]


class Start(BaseModel):
    mission_mode: Literal["inspection", "natural_language_flight", "adaptive_inspection"] = "inspection"
    scenario: Literal["clear", "occlusion", "blocked", "battery", "collision", "adaptive_complex", "adaptive_changed"] = "occlusion"
    vision: Literal["calibration", "openai"] = "calibration"
    objective: str = Field(default="Inspect exterior and roof", min_length=1, max_length=2000)
    use_gpt: bool = False
    seed: int = Field(default=7, ge=0, le=100000)
    wind: float = Field(default=0, ge=0, le=1.5)


class Fault(BaseModel):
    kind: Literal["battery", "gps", "stale", "link", "wind"]


class EventStore(Store):
    def __init__(self, path, journal):
        self.journal = journal
        super().__init__(path)

    def save_mission(self, p):
        super().save_mission(p)
        self.journal("mission_status", status=p.status.value, error=p.error, mission_kind="inspection")

    def save_observation(self, o):
        super().save_observation(o)
        self.journal(
            "observation",
            id=o.id,
            label=o.label,
            accepted=o.accepted,
            attempt=o.attempt,
            quality=o.quality_score,
            issues=o.quality_issues + o.assessment.get("issues", []),
            file=Path(o.image_path).name,
            explanation=o.assessment.get("explanation"),
            sha256=o.image_sha256,
        )


class Manager:
    def __init__(self, runtime, rate=1):
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.events = []
        self.event_lock = threading.Lock()
        self.world = World(self.runtime / "preview")
        self.world.start()

        self.run_id = None
        self.plan = None
        self.store = None
        self.executor = None
        self.task = None
        self.run_dir = None
        self.result = None
        self.options = None
        self.journal_file = None
        self.begin_lock = asyncio.Lock()
        self.camera_lock = asyncio.Lock()
        self.camera_cache = None
        self.camera_cache_time = 0.0

        self.mission_kind = "inspection"
        self.maneuver_plan = None
        self.maneuver_intent = None
        self.maneuver_stop = None
        self.adaptive_intent = None
        self.adaptive_stop = None
        self.adaptive_outcome = None
        self.camera_cache = None
        self.camera_cache_time = 0.0
        self.gpt_connection = {
            "configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MISSION_MODEL")),
            "verified": False,
            "model": os.getenv("OPENAI_MISSION_MODEL"),
            "verified_utc": None,
            "response_id": None,
        }

    def journal(self, kind, **fields):
        with self.event_lock:
            event = {
                "seq": len(self.events) + 1,
                "utc": datetime.now(timezone.utc).isoformat(),
                "sim_time": round(self.world.t, 3),
                "kind": kind,
                **fields,
            }
            self.events.append(event)
            if self.journal_file:
                with self.journal_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")

    @property
    def running(self):
        return self.task is not None and not self.task.done()

    async def _fresh_run(self, options):
        # Closing Bullet can wait for an in-flight render/physics step. Keep that wait
        # off the ASGI event loop so the browser does not interpret startup as a crash.
        old_world = self.world
        await asyncio.to_thread(old_world.close)
        if self.store:
            self.store.close()
        self.run_id = str(uuid.uuid4())
        self.run_dir = self.runtime / "runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.world = World(self.run_dir / "images", options.scenario, options.seed, self.rate, options.wind)
        self.events = []
        self.result = None
        self.options = options
        self.journal_file = self.run_dir / "events.jsonl"
        self.plan = None
        self.store = None
        self.executor = None
        self.maneuver_plan = None
        self.maneuver_intent = None
        self.maneuver_stop = None
        self.adaptive_intent = None
        self.adaptive_stop = None
        self.adaptive_outcome = None
        self.camera_cache = None
        self.camera_cache_time = 0.0

    def _source_hashes(self):
        return {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in [ROOT / "app", ROOT / "simulation"]
            for p in sorted(root.rglob("*.py"))
        }

    async def verify_gpt(self):
        self.gpt_connection["configured"] = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MISSION_MODEL"))
        self.gpt_connection["model"] = os.getenv("OPENAI_MISSION_MODEL")
        if not self.gpt_connection["configured"]:
            self.gpt_connection["verified"] = False
            raise ModelUnavailable("OPENAI_API_KEY and OPENAI_MISSION_MODEL are not configured")
        try:
            info = await verify_gpt_link()
        except Exception:
            self.gpt_connection["verified"] = False
            self.gpt_connection["verified_utc"] = None
            self.gpt_connection["response_id"] = None
            raise
        self.gpt_connection.update(info)
        self.gpt_connection["configured"] = True
        return dict(self.gpt_connection)

    async def begin(self, options: Start):
        async with self.begin_lock:
            if self.running:
                raise ValueError("A mission is already running")
            if options.mission_mode == "natural_language_flight":
                return await self._begin_maneuver(options)
            if options.mission_mode == "adaptive_inspection":
                return await self._begin_adaptive(options)
            return await self._begin_inspection(options)

    async def _begin_inspection(self, options: Start):
        self.mission_kind = "inspection"
        if options.vision == "openai" and not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_VISION_MODEL")):
            raise ValueError("Set OPENAI_API_KEY and OPENAI_VISION_MODEL on the server first")
        if options.use_gpt and not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MISSION_MODEL")):
            raise ValueError("Set OPENAI_API_KEY and OPENAI_MISSION_MODEL first")

        asset = scene_asset()
        req = MissionPlanRequest(asset_id=asset.id, prompt=options.objective, adapter="physics", use_gpt=options.use_gpt)
        spec = await GPTMissionInterpreter().interpret(asset, req.prompt) if options.use_gpt else DeterministicMissionInterpreter().interpret(asset, req.prompt)
        if options.vision == "calibration":
            for ev in spec.evidence_requirements:
                if ev.kind == "visual_coverage":
                    ev.criterion = "At least three of the four surveyed calibration panels for this surface must be optically decodable."
        plan = InspectionPlanner().plan_from_spec(asset, req, spec)
        SafetyGate().validate_plan(asset, plan)

        await self._fresh_run(options)
        self.mission_kind = "inspection"
        self.store = EventStore(self.run_dir / "simulation.sqlite", self.journal)
        self.store.save_asset(asset)
        self.store.save_mission(plan)
        self.plan = plan
        inner = CalibrationVision() if options.vision == "calibration" else OpenAIPerception(
            os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_VISION_MODEL"), simulation=True
        )
        self.executor = MissionExecutor(
            self.store,
            SafetyGate(),
            QualityInspector(),
            HumanPresenceDetector(),
            CompletionEngine(EvidenceVerifier()),
            GapReplanner(),
            JournalPerception(inner, self.journal),
        )
        manifest = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mission_kind": "inspection",
            "engine": "PyBullet/Bullet native runtime",
            "plant": "idealized force-controlled rigid body; not manufacturer firmware",
            "source": "simulated_sensor",
            "vision": options.vision,
            "model": os.getenv("OPENAI_VISION_MODEL") if options.vision == "openai" else "OpenCV ArUco DICT_4X4_50",
            "options": options.model_dump(),
            "simulation_rate_requested": self.rate,
            "mission_interpreter": "live GPT" if options.use_gpt else "bounded deterministic interpreter",
            "seed_note": "Scene is deterministic; seed reserved for future variation. Wind is deterministic.",
            "source_hashes": self._source_hashes(),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (self.run_dir / "asset.json").write_text(asset.model_dump_json(indent=2))
        self.world.start()
        self.journal("run_started", **options.model_dump())
        self.task = asyncio.create_task(self._run_inspection())
        return self.run_id

    async def _begin_adaptive(self, options: Start):
        if not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MISSION_MODEL")):
            raise ModelUnavailable("Adaptive inspection requires OPENAI_API_KEY and OPENAI_MISSION_MODEL")
        if options.scenario not in {"adaptive_complex", "adaptive_changed"}:
            # The adaptive demo intentionally uses the complex site; no hidden substitution occurs.
            options = options.model_copy(update={"scenario": "adaptive_complex"})

        intent, gpt_meta = await GPTManeuverInterpreter().interpret(options.objective, adaptive_inspection=True)
        self.gpt_connection.update({
            "configured": True, "verified": True,
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "model": gpt_meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
            "response_id": gpt_meta.get("response_id"),
        })

        await self._fresh_run(options)
        self.mission_kind = "adaptive_inspection"
        self.adaptive_intent = intent
        self.adaptive_stop = asyncio.Event()
        memory = PersistentAssetMemory(self.runtime / "asset_memory")
        self.executor = AdaptiveMissionExecutor(self.journal, memory, self.adaptive_stop)

        record = {
            "prompt": options.objective,
            "gpt": gpt_meta,
            "mission_ir": intent.model_dump(mode="json"),
            "architecture": "free-form language -> live GPT unified ARIS Mission IR -> ARIS discovery/replanner/evidence loop -> PhysicsDrone",
            "route_source": "runtime A* from discovered obstacle map; no recorded alternate route",
        }
        (self.run_dir / "adaptive_interpretation.json").write_text(json.dumps(record, indent=2))
        manifest = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mission_kind": "adaptive_inspection",
            "engine": "PyBullet/Bullet native runtime",
            "plant": "idealized force-controlled rigid body; not manufacturer firmware",
            "scene": options.scenario,
            "source": "live_gpt_to_aris_dynamic_replanner_to_physics",
            "mission_interpreter": "LIVE OpenAI Responses API unified ARIS Mission IR (flight + adaptive inspection)",
            "gpt_model": gpt_meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
            "gpt_response_id": gpt_meta.get("response_id"),
            "obstacle_model": "site obstacles exist in physics but become available to ARIS only through local simulated obstacle sensing",
            "object_discovery_model": "clean-slate simulated onboard object detector; no target object coordinates are supplied at mission start",
            "evidence": "OpenCV ArUco detection on rendered RGB corner imagery",
            "asset_memory": "persistent across runs in demo_runtime/asset_memory",
            "options": options.model_dump(),
            "source_hashes": self._source_hashes(),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self.world.start()
        self.journal("run_started", **options.model_dump())
        self.journal(
            "gpt_interpretation", provider="OpenAI Responses API",
            model=gpt_meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
            response_id=gpt_meta.get("response_id"), original_prompt=options.objective,
            mission_ir=intent.model_dump(mode="json"),
        )
        self.task = asyncio.create_task(self._run_adaptive())
        return self.run_id

    async def _begin_maneuver(self, options: Start):
        # Natural-language flight is intentionally LIVE-GPT-only. There is no canned fallback.
        if not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MISSION_MODEL")):
            raise ModelUnavailable("Natural-language flight requires OPENAI_API_KEY and OPENAI_MISSION_MODEL")

        asset = scene_asset()
        # Real model request happens before the flight world is started. A failure means no motion.
        intent, gpt_meta = await GPTManeuverInterpreter().interpret(options.objective)
        # A successful interpretation is itself a verified live provider round trip.
        self.gpt_connection.update({
            "configured": True,
            "verified": True,
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "model": gpt_meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
            "response_id": gpt_meta.get("response_id"),
        })
        compiled = ManeuverCompiler(home_xy=HOME[:2], asset_center_xy=(0.0, 0.0)).compile(options.objective, intent)
        ManeuverSafetyGate(geo, home_xy=HOME[:2]).validate(asset, compiled)
        SafetyGate().validate_preflight(asset, self.world.telemetry())

        await self._fresh_run(options)
        self.mission_kind = "natural_language_flight"
        self.maneuver_intent = intent
        self.maneuver_plan = compiled
        self.maneuver_stop = asyncio.Event()
        self.executor = ManeuverExecutor(
            self.journal,
            lambda telemetry: SafetyGate().check_runtime(asset, telemetry, compiled.max_altitude_m),
            self.maneuver_stop,
        )

        interpretation_record = {
            "prompt": options.objective,
            "gpt": gpt_meta,
            "mission_ir": intent.model_dump(mode="json"),
            "compiled_by": "ARIS ManeuverCompiler",
            "compiled_plan": compiled.model_dump(mode="json"),
            "safety_gate": "APPROVED",
            "note": "GPT interpreted free-form language into ARIS Mission IR only. ARIS deterministically generated and safety-validated every concrete command.",
        }
        (self.run_dir / "gpt_interpretation.json").write_text(json.dumps(interpretation_record, indent=2))
        (self.run_dir / "asset.json").write_text(asset.model_dump_json(indent=2))
        manifest = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mission_kind": "natural_language_flight",
            "engine": "PyBullet/Bullet native runtime",
            "plant": "idealized force-controlled rigid body; not manufacturer firmware",
            "source": "live_gpt_intent_to_aris_commands_to_physics",
            "mission_interpreter": "LIVE OpenAI Responses API → ARIS semantic Mission IR; fail-closed",
            "gpt_model": gpt_meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
            "gpt_response_id": gpt_meta.get("response_id"),
            "options": options.model_dump(),
            "simulation_rate_requested": self.rate,
            "source_hashes": self._source_hashes(),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self.world.start()
        self.journal("run_started", **options.model_dump())
        self.journal(
            "gpt_interpretation",
            provider="OpenAI Responses API",
            model=gpt_meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
            response_id=gpt_meta.get("response_id"),
            original_prompt=options.objective,
            structured_intent=intent.model_dump(mode="json"),
        )
        self.journal(
            "safety_gate",
            result="APPROVED",
            compiled_by="ARIS ManeuverCompiler",
            command_count=len(compiled.commands),
            max_altitude_m=compiled.max_altitude_m,
        )
        self.task = asyncio.create_task(self._run_maneuver())
        return self.run_id

    async def _run_inspection(self):
        try:
            result = await self.executor.run(self.plan, PhysicsDrone(self.world, self.journal))
            self.plan = result
            report = build_report(self.store, result.id)
            truth = self.world.capture_truth
            obs = self.store.observations_for_mission(result.id)
            comparisons = [
                {
                    "observation": o.id,
                    "label": o.label,
                    "aris_accepted": o.accepted,
                    "oracle_visible": truth.get(o.image_sha256, {}).get("criterion_met"),
                    "sha256": o.image_sha256,
                }
                for o in obs
            ]
            false_accepts = [o for o in comparisons if o["aris_accepted"] and o["oracle_visible"] is not True]
            safe_landing = await self._safe_landed_state()
            observed_labels = {o["label"] for o in comparisons if o["aris_accepted"] and o["oracle_visible"] is True}
            required = {r.target for r in result.mission_spec.evidence_requirements if r.kind == "visual_coverage" and r.required}
            false_complete = result.status == MissionStatus.COMPLETE and (not required.issubset(observed_labels) or not safe_landing)
            expected = {"clear": "COMPLETE", "occlusion": "COMPLETE", "blocked": "INCOMPLETE", "battery": "ABORTED", "collision": "FAILED"}[self.options.scenario]
            expected_ok = result.status.value == expected if not self.world.faults or self.options.scenario == "battery" else result.status.value in {"ABORTED", "FAILED"}
            oracle = {
                "basis": "Independent ray visibility + segmentation pixels of calibration panels; not semantic defect accuracy",
                "false_acceptances": len(false_accepts),
                "false_complete": bool(false_complete),
                "collisions": self.world.collisions,
                "landed_in_physics": safe_landing,
                "expected_status": expected,
                "observed_status": result.status.value,
                "observations": comparisons,
                "verdict": "PASS" if expected_ok and not false_accepts and not false_complete and not self.world.collisions and safe_landing else "FAIL",
            }
            if self.options.vision == "openai":
                oracle["verdict"] = "REVIEW_REQUIRED"
                oracle["basis"] += "; live semantic criteria require separate human-labeled ground truth"
            self.result = {
                "mission": report,
                "oracle": oracle,
                "disclosure": "SIMULATED SENSOR EVIDENCE — not a physical flight or Parrot firmware validation",
            }
            self.journal("run_finished", status=result.status.value, verdict=oracle["verdict"])
            if safe_landing:
                self.world.freeze()
            self._save()
        except Exception as exc:
            self.journal("infrastructure_error", error=type(exc).__name__)
            self.result = {"error": type(exc).__name__, "oracle": {"verdict": "ERROR"}}
            self._save()

    async def _run_adaptive(self):
        try:
            asset = scene_asset()
            SafetyGate().validate_preflight(asset, self.world.telemetry())
            outcome = await self.executor.run(
                self.adaptive_intent,
                PhysicsDrone(self.world, self.journal),
                self.run_id,
                self.options.scenario,
            )
            self.adaptive_outcome = outcome
            safe_landing = await self._safe_landed_state()
            accepted = len(outcome.get("captures", []))
            required_corners = outcome.get("required_corners", [])
            required_targets = outcome.get("required_targets", required_corners)
            required = len(required_targets)
            oracle = {
                "basis": "Rendered RGB evidence + independent Bullet collision state + landing state. Dynamic routes/viewpoints are generated from the progressively discovered obstacle map.",
                "required_evidence_targets": required,
                "accepted_evidence_targets": accepted,
                "required_corner_images": len(required_corners),
                "accepted_corner_images": len([c for c in outcome.get("captures", []) if c.get("corner")]),
                "replans": outcome.get("replans", 0),
                "discovered_obstacles": outcome.get("discovered_obstacles", []),
                "discovered_objects": outcome.get("discovered_objects", []),
                "collisions": list(self.world.collisions),
                "landed_in_physics": safe_landing,
                "verdict": "PASS" if outcome.get("status") == "COMPLETE" and accepted == required and not self.world.collisions and safe_landing else "FAIL",
            }
            self.result = {
                "mission": {
                    "status": outcome.get("status"),
                    "prompt": self.options.objective,
                    "ai_summary": self.adaptive_intent.summary,
                    "mission_ir": self.adaptive_intent.model_dump(mode="json"),
                    "required_corners": required_corners,
                    "required_targets": required_targets,
                    "replans": outcome.get("replans", 0),
                    "discovered_obstacles": outcome.get("discovered_obstacles", []),
                    "discovered_objects": outcome.get("discovered_objects", []),
                    "landed_confirmed": outcome.get("landed_confirmed"),
                    "error": outcome.get("error"),
                },
                "asset_memory": outcome.get("asset_memory"),
                "oracle": oracle,
                "disclosure": "SIMULATED ADAPTIVE INSPECTION — live RS objective, clean-slate runtime object/obstacle discovery, ARIS-generated replanning and rendered-camera evidence. Not physical-aircraft validation.",
            }
            self.journal("run_finished", status=outcome.get("status"), verdict=oracle["verdict"])
            if safe_landing:
                self.world.freeze()
            self._save()
        except Exception as exc:
            tb = traceback.format_exc()
            self.journal(
                "infrastructure_error",
                error=type(exc).__name__,
                message=str(exc)[:500],
            )
            # Preserve the full traceback in the run folder rather than losing the
            # only useful diagnostic if an investor-demo mission hits infrastructure.
            if self.run_dir:
                (self.run_dir / "backend_error.txt").write_text(tb, encoding="utf-8")
            try:
                if self.world.telemetry().airborne:
                    await PhysicsDrone(self.world, self.journal).abort()
            except Exception as recovery_exc:
                self.journal(
                    "infrastructure_recovery_error",
                    error=type(recovery_exc).__name__,
                    message=str(recovery_exc)[:300],
                )
            self.result = {
                "error": type(exc).__name__,
                "message": str(exc),
                "oracle": {"verdict": "ERROR"},
            }
            try:
                self._save()
            except Exception as save_exc:
                self.journal(
                    "result_save_error",
                    error=type(save_exc).__name__,
                    message=str(save_exc)[:300],
                )

    async def _run_maneuver(self):
        try:
            asset = scene_asset()
            SafetyGate().validate_preflight(asset, self.world.telemetry())
            outcome = await self.executor.run(self.maneuver_plan, PhysicsDrone(self.world, self.journal))
            safe_landing = await self._safe_landed_state()
            commands_logged = len([e for e in self.events if e.get("kind") == "aris_command"])
            complete = outcome.get("status") == "COMPLETE"
            oracle = {
                "basis": "ARIS command completion + Bullet collision state + confirmed landing. No prerecorded trajectory is executed.",
                "compiled_commands": len(self.maneuver_plan.commands),
                "commands_executed": commands_logged,
                "collisions": list(self.world.collisions),
                "landed_in_physics": safe_landing,
                "observed_status": outcome.get("status"),
                "verdict": "PASS" if complete and commands_logged == len(self.maneuver_plan.commands) and not self.world.collisions and safe_landing else "FAIL",
            }
            self.result = {
                "mission": {
                    "status": outcome.get("status"),
                    "prompt": self.maneuver_plan.prompt,
                    "ai_explanation": self.maneuver_plan.ai_explanation,
                    "defaults_applied": self.maneuver_plan.defaults_applied,
                    "compiled_commands": len(self.maneuver_plan.commands),
                    "landed_confirmed": outcome.get("landed_confirmed"),
                    "error": outcome.get("error"),
                },
                "oracle": oracle,
                "gpt": {
                    "live": True,
                    "model": os.getenv("OPENAI_MISSION_MODEL"),
                    "fail_closed": True,
                },
                "disclosure": "FREE-FORM LANGUAGE → LIVE GPT semantic Mission IR → ARIS deterministic compiler/safety gate/executor → Bullet simulated aircraft. Not a physical flight.",
            }
            self.journal("run_finished", status=outcome.get("status"), verdict=oracle["verdict"])
            if safe_landing:
                self.world.freeze()
            self._save()
        except Exception as exc:
            self.journal("infrastructure_error", error=type(exc).__name__, message=str(exc)[:240])
            self.result = {"error": type(exc).__name__, "oracle": {"verdict": "ERROR"}}
            self._save()

    async def _safe_landed_state(self):
        try:
            return not self.world.telemetry().airborne
        except Exception as exc:
            self.journal("landing_state_unavailable", error=type(exc).__name__, message=str(exc)[:240])
            return False

    async def camera_jpeg(self):
        # Browser polls frequently. Cache for half a second and serialize generation so
        # a slow TinyRenderer frame cannot pile up native Bullet calls or starve control.
        now = time.monotonic()
        if self.camera_cache is not None and now - self.camera_cache_time < 0.5:
            return self.camera_cache
        async with self.camera_lock:
            now = time.monotonic()
            if self.camera_cache is not None and now - self.camera_cache_time < 0.5:
                return self.camera_cache
            world = self.world
            def render_bytes():
                rgb, _ = world.render(400, 300)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=72)
                return buf.getvalue()
            payload = await asyncio.to_thread(render_bytes)
            if world is self.world:
                self.camera_cache = payload
                self.camera_cache_time = time.monotonic()
            return payload

    def _save(self):
        if not self.run_dir:
            return
        (self.run_dir / "report.json").write_text(json.dumps(self.result, indent=2))
        (self.run_dir / "ground_truth.json").write_text(json.dumps(self.world.capture_truth, indent=2))
        (self.run_dir / "trajectory.json").write_text(json.dumps(self.world.path))
        (self.run_dir / "replay.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "scene": self.scene(),
                    "trajectory": self.world.path,
                    "events": self.events,
                    "result": self.result,
                }
            )
        )

    def scene(self):
        return {
            "objects": self.world.objects,
            "markers": [{k: v for k, v in m.items() if k != "body"} for m in self.world.markers],
        }

    def _maneuver_status(self):
        for event in reversed(self.events):
            if event.get("kind") == "mission_status":
                return event.get("status")
        return "READY"

    def snapshot(self):
        plan = self.store.get_mission(self.plan.id) if self.store and self.plan else None
        observations = []
        mission = None
        if plan:
            mission = plan.model_dump(mode="json")
            observations = [
                {
                    "id": o.id,
                    "label": o.label,
                    "accepted": o.accepted,
                    "attempt": o.attempt,
                    "quality": o.quality_score,
                    "file": Path(o.image_path).name,
                    "explanation": o.assessment.get("explanation", ""),
                    "issues": o.quality_issues + o.assessment.get("issues", []),
                }
                for o in self.store.observations_for_mission(plan.id)
            ]
        elif self.adaptive_intent:
            status = self._maneuver_status()
            mission = {
                "id": self.run_id,
                "status": status,
                "prompt": self.options.objective if self.options else "",
                "ai_summary": self.adaptive_intent.summary,
                "mission_kind": "adaptive_inspection",
                "mission_ir": self.adaptive_intent.model_dump(mode="json"),
                "required_corners": (self.adaptive_outcome or {}).get("required_corners", []),
                "required_targets": (self.adaptive_outcome or {}).get("required_targets", []),
                "replans": (self.adaptive_outcome or {}).get("replans", len([e for e in self.events if e.get("kind") in {"route_replanned", "evidence_replan"}])),
            }
            for event in self.events:
                if event.get("kind") == "corner_capture":
                    observations.append({
                        "id": f"{event.get('corner')}-{event.get('attempt')}",
                        "label": event.get("corner"),
                        "accepted": bool(event.get("accepted")),
                        "attempt": event.get("attempt"),
                        "quality": 1.0 if event.get("accepted") else 0.0,
                        "file": event.get("file"),
                        "explanation": (event.get("assessment") or {}).get("reason", ""),
                        "issues": [] if event.get("accepted") else [(event.get("assessment") or {}).get("reason", "evidence_failed")],
                    })
        elif self.maneuver_plan:
            mission = {
                "id": self.maneuver_plan.id,
                "status": self._maneuver_status(),
                "prompt": self.maneuver_plan.prompt,
                "ai_summary": self.maneuver_plan.ai_explanation,
                "defaults_applied": self.maneuver_plan.defaults_applied,
                "compiled_commands": len(self.maneuver_plan.commands),
                "mission_kind": "natural_language_flight",
            }

        configured = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MISSION_MODEL"))
        self.gpt_connection["configured"] = configured
        self.gpt_connection["model"] = os.getenv("OPENAI_MISSION_MODEL")
        return {
            "run_id": self.run_id,
            "running": self.running,
            "mission_kind": self.mission_kind,
            "state": self.world.state(),
            "mission": mission,
            "maneuver_intent": self.maneuver_intent.model_dump(mode="json") if self.maneuver_intent else None,
            "maneuver_plan": self.maneuver_plan.model_dump(mode="json") if self.maneuver_plan else None,
            "adaptive_intent": self.adaptive_intent.model_dump(mode="json") if self.adaptive_intent else None,
            "adaptive_outcome": self.adaptive_outcome,
            "observations": observations,
            "events": self.events[-160:],
            "result": self.result,
            "options": self.options.model_dump() if self.options else None,
            "gpt_connection": dict(self.gpt_connection),
            "capabilities": {
                "openai_vision": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_VISION_MODEL")),
                "gpt_mission": configured,
                "natural_language_flight": configured,
                "adaptive_inspection": configured,
            },
        }

    async def request_abort(self):
        if not self.running:
            raise ValueError("No active mission")
        if self.mission_kind == "natural_language_flight":
            self.maneuver_stop.set()
        elif self.mission_kind == "adaptive_inspection":
            self.adaptive_stop.set()
        else:
            self.executor.request_stop(self.plan.id)
        self.journal("operator_abort")

    async def close(self):
        if self.running:
            try:
                await self.request_abort()
                await asyncio.wait_for(asyncio.shield(self.task), 10)
            except asyncio.TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        await asyncio.to_thread(self.world.close)
        if self.store:
            self.store.close()


def create_app(runtime=None, token=None, rate=1):
    token = token or secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = Manager(runtime or ROOT / "demo_runtime", rate)
        try:
            yield
        finally:
            await app.state.manager.close()

    app = FastAPI(title="ARIS Simulation Lab", lifespan=lifespan)
    app.state.token = token

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path.startswith("/demo/"):
            provided = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not secrets.compare_digest(provided, token):
                return Response("Unauthorized", status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/demo/state")
    async def state():
        return app.state.manager.snapshot()

    @app.get("/demo/scene")
    async def scene():
        return app.state.manager.scene()

    @app.post("/demo/gpt/verify")
    async def gpt_verify():
        try:
            return await app.state.manager.verify_gpt()
        except ModelUnavailable as e:
            raise HTTPException(503, str(e))
        except Exception:
            raise HTTPException(503, "Live GPT verification failed")

    @app.post("/demo/start")
    async def start(body: Start):
        try:
            return {"run_id": await app.state.manager.begin(body)}
        except ModelUnavailable as e:
            raise HTTPException(503, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/demo/abort")
    async def abort():
        try:
            await app.state.manager.request_abort()
            return {"requested": True}
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/demo/fault")
    async def fault(body: Fault):
        m = app.state.manager
        if not m.running:
            raise HTTPException(409, "Start a mission first")
        m.world.inject(body.kind)
        m.journal("fault_injected", fault=body.kind)
        return {"injected": body.kind}

    @app.get("/demo/camera")
    async def camera():
        try:
            return Response(await app.state.manager.camera_jpeg(), media_type="image/jpeg")
        except Exception as exc:
            raise HTTPException(503, f"Camera not available: {type(exc).__name__}")

    @app.get("/demo/image/{name}")
    async def image(name: str):
        m = app.state.manager
        if Path(name).name != name or not name.endswith(".png"):
            raise HTTPException(404)
        path = m.world.directory / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/demo/export")
    async def export():
        m = app.state.manager
        if not m.result or m.running:
            raise HTTPException(409, "Finish the run before exporting")
        m._save()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(m.run_dir.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".png", ".obj", ".txt", ".log"}:
                    z.write(path, path.relative_to(m.run_dir))
        return Response(
            buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="ARIS-run-{m.run_id}.zip"'},
        )

    app.mount("/", StaticFiles(directory=ROOT / "simulation" / "web", html=True), name="web")
    return app
