from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ai_client import ResponsesJSON, ModelUnavailable
from .geo import point_allowed, segment_allowed


Direction = Literal["forward", "backward", "left", "right", "north", "east", "south", "west"]
TargetRef = Literal[
    "current_position",
    "home",
    "asset_center",
    "asset_north",
    "asset_south",
    "asset_east",
    "asset_west",
]

CornerRef = Literal["corner_ne", "corner_nw", "corner_se", "corner_sw"]
SurfaceRef = Literal["north_facade", "east_facade", "south_facade", "west_facade", "roof"]
ObjectClass = Literal["building", "tree", "structure", "object"]
ObjectSelection = Literal["nearest", "largest", "all"]
DiscoveredInspectionStyle = Literal["photograph", "orbit", "all_sides"]


class MissionAction(BaseModel):
    """
    Semantic ARIS Mission IR action.

    The user never needs to know this schema. GPT maps ordinary language into it;
    ARIS validates and compiles it into deterministic aircraft commands.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal[
        "takeoff",
        "move",
        "turn",
        "set_altitude",
        "hover",
        "orbit",
        "go_to",
        "inspect_corners",
        "inspect_asset",
        "find_and_inspect",
        "return_home",
        "land",
    ]

    # Common motion fields.
    altitude_m: float | None = Field(default=None, ge=0, le=30)
    speed_mps: float | None = Field(default=None, gt=0, le=8)
    duration_s: float | None = Field(default=None, ge=0.5, le=60)
    distance_m: float | None = Field(default=None, ge=0.5, le=60)

    # Relative / directional motion.
    direction: Direction | None = None
    heading_deg: float | None = Field(default=None, ge=0, lt=360)

    # Turn.
    turn_degrees: float | None = Field(default=None, ge=-360, le=360)

    # Targeted motion/orbit.
    target: TargetRef | None = None
    stop_distance_m: float | None = Field(default=None, ge=0, le=30)
    radius_m: float | None = Field(default=None, ge=2, le=20)
    laps: int | None = Field(default=None, ge=1, le=8)
    orbit_direction: Literal["clockwise", "counter_clockwise"] | None = None

    # Adaptive inspection semantics.
    # inspect_corners exists only for explicit corner-specific requests.
    corners: list[CornerRef] | None = None
    corner_strategy: Literal["given_order", "nearest_safe"] | None = None

    # General asset coverage. Ordinary requests such as "all directions", "all sides",
    # "facades", or "full exterior" use inspect_asset rather than corners.
    inspection_targets: list[SurfaceRef] | None = None
    coverage_mode: Literal["specified_surfaces", "all_sides", "full_exterior"] | None = None
    coverage_strategy: Literal["given_order", "nearest_safe"] | None = None

    evidence_requirement: Literal["clear_photo"] | None = None
    standoff_m: float | None = Field(default=None, ge=4, le=14)

    # Clean-slate world discovery. ARIS searches first; coordinates are never supplied by RS.
    object_class: ObjectClass | None = None
    object_selection: ObjectSelection | None = None
    discovered_inspection_style: DiscoveredInspectionStyle | None = None

    # Motion-planning policy for autonomous discovery/inspection. Adaptive permits
    # ARIS to change altitude within the validated mission envelope to route around
    # obstacles. Fixed preserves the requested altitude except for takeoff/landing.
    altitude_policy: Literal["fixed", "adaptive"] | None = None

    # Generic action repetition. ARIS expands this deterministically.
    repeat: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def validate_action(self):
        if self.kind == "takeoff" and self.altitude_m is None:
            raise ValueError("takeoff requires altitude_m")

        if self.kind == "move":
            if self.duration_s is None and self.distance_m is None:
                raise ValueError("move requires duration_s or distance_m")
            if self.duration_s is not None and self.distance_m is not None:
                raise ValueError("move uses duration_s or distance_m, not both")
            if self.heading_deg is not None and self.direction is not None:
                raise ValueError("move uses heading_deg or direction, not both")

        if self.kind == "turn":
            if self.heading_deg is None and self.turn_degrees is None:
                raise ValueError("turn requires heading_deg or turn_degrees")
            if self.heading_deg is not None and self.turn_degrees is not None:
                raise ValueError("turn uses heading_deg or turn_degrees, not both")

        if self.kind == "set_altitude" and self.altitude_m is None:
            raise ValueError("set_altitude requires altitude_m")

        if self.kind == "hover" and self.duration_s is None:
            raise ValueError("hover requires duration_s")

        if self.kind == "go_to" and self.target is None:
            raise ValueError("go_to requires target")

        if self.kind == "orbit":
            missing = [
                name
                for name in ("target", "radius_m", "laps", "orbit_direction")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError("orbit missing: " + ", ".join(missing))

        if self.kind == "inspect_corners":
            if not self.corners:
                raise ValueError("inspect_corners requires at least one corner")
            if self.corner_strategy is None:
                raise ValueError("inspect_corners requires corner_strategy")
            if self.evidence_requirement is None:
                raise ValueError("inspect_corners requires evidence_requirement")

        if self.kind == "inspect_asset":
            if self.coverage_mode is None:
                raise ValueError("inspect_asset requires coverage_mode")
            if self.coverage_strategy is None:
                raise ValueError("inspect_asset requires coverage_strategy")
            if self.evidence_requirement is None:
                raise ValueError("inspect_asset requires evidence_requirement")
            if self.coverage_mode == "specified_surfaces" and not self.inspection_targets:
                raise ValueError("specified_surfaces requires inspection_targets")

        if self.kind == "find_and_inspect":
            if self.object_class is None:
                raise ValueError("find_and_inspect requires object_class")
            if self.object_selection is None:
                raise ValueError("find_and_inspect requires object_selection")
            if self.discovered_inspection_style is None:
                raise ValueError("find_and_inspect requires discovered_inspection_style")
            if self.evidence_requirement is None:
                raise ValueError("find_and_inspect requires evidence_requirement")

        return self


class NaturalLanguageMission(BaseModel):
    """
    GPT-facing semantic mission representation.

    disposition=clarify is important: ARIS should ask rather than fake certainty
    when a reference such as "over there" cannot be resolved from available context.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: Literal["execute", "clarify", "reject"]
    summary: str
    clarification_question: str | None = None
    assumptions: list[str] = Field(default_factory=list, max_length=40)
    actions: list[MissionAction] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def consistency(self):
        if self.disposition == "execute" and not self.actions:
            raise ValueError("Executable mission must contain actions")
        if self.disposition == "clarify" and not self.clarification_question:
            raise ValueError("Clarification mission requires clarification_question")
        return self


# Backward names used by the existing server/UI.
ManeuverMission = NaturalLanguageMission
ManeuverStep = MissionAction


class GPTLinkCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    nonce: str
    message: str


class ClarificationRequired(ValueError):
    pass


class CompiledCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: Literal["takeoff", "goto", "straight", "turn", "hover", "return_home", "land"]
    label: str
    x_m: float | None = None
    y_m: float | None = None
    altitude_m: float | None = Field(default=None, ge=0, le=30)
    heading_deg: float | None = Field(default=None, ge=0, lt=360)
    speed_mps: float | None = Field(default=None, gt=0, le=8)
    duration_s: float | None = Field(default=None, ge=0, le=60)
    distance_m: float | None = Field(default=None, ge=0, le=80)
    source_action_index: int | None = None


class CompiledManeuver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    ai_explanation: str
    defaults_applied: list[str]
    commands: list[CompiledCommand] = Field(min_length=1, max_length=300)
    max_altitude_m: float = Field(default=20, ge=5, le=20)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Conservative demo defaults. These are disclosed in the UI/logs when used.
DEFAULT_TAKEOFF_ALTITUDE_M = 8.0
DEFAULT_HOVER_SECONDS = 5.0
DEFAULT_SPEED_MPS = 2.0
DEFAULT_MOVE_DISTANCE_M = 8.0
DEFAULT_ORBIT_RADIUS_M = 6.0
DEFAULT_ORBIT_ALTITUDE_M = 8.0
DEFAULT_TURN_DEGREES = 90.0

# Demo safety envelope. These are intentionally lower than the permissive
# Pydantic parse bounds so SafetyGate owns the final authority.
MAX_ALTITUDE_M = 20.0
MAX_SPEED_MPS = 5.0
MAX_DURATION_S = 30.0
MAX_MOVE_DISTANCE_M = 40.0
MAX_ORBIT_RADIUS_M = 12.0
MAX_LAPS = 5


def _note(notes: list[str], value: str):
    if value not in notes:
        notes.append(value)


def _null_action(kind: str, **updates) -> dict:
    data = {
        "kind": kind,
        "altitude_m": None,
        "speed_mps": None,
        "duration_s": None,
        "distance_m": None,
        "direction": None,
        "heading_deg": None,
        "turn_degrees": None,
        "target": None,
        "stop_distance_m": None,
        "radius_m": None,
        "laps": None,
        "orbit_direction": None,
        "corners": None,
        "corner_strategy": None,
        "inspection_targets": None,
        "coverage_mode": None,
        "coverage_strategy": None,
        "evidence_requirement": None,
        "standoff_m": None,
        "object_class": None,
        "object_selection": None,
        "discovered_inspection_style": None,
        "altitude_policy": None,
        "repeat": 1,
    }
    data.update(updates)
    return data


def _normalize_intent(raw: dict, adaptive_inspection: bool = False) -> dict:
    """
    Normalize only execution details that are safe, explicit and auditable.
    This is not phrase matching. GPT already produced semantic actions.
    """
    data = dict(raw)
    actions = [dict(a) for a in (data.get("actions") or [])]
    assumptions = list(data.get("assumptions") or [])

    if data.get("disposition") != "execute":
        data["actions"] = actions
        data["assumptions"] = assumptions
        return data

    # Determine whether the mission requires airborne motion.
    flight_kinds = {"move", "turn", "set_altitude", "hover", "orbit", "go_to", "inspect_corners", "inspect_asset", "find_and_inspect", "return_home"}
    requires_flight = any(a.get("kind") in flight_kinds for a in actions)
    has_takeoff = any(a.get("kind") == "takeoff" for a in actions)

    if requires_flight and not has_takeoff:
        actions.insert(0, _null_action("takeoff", altitude_m=DEFAULT_TAKEOFF_ALTITUDE_M))
        _note(assumptions, "takeoff to 8 m inserted because airborne action was requested")

    for action in actions:
        kind = action.get("kind")

        if kind == "takeoff" and action.get("altitude_m") is None:
            action["altitude_m"] = DEFAULT_TAKEOFF_ALTITUDE_M
            _note(assumptions, "takeoff altitude 8 m used where no altitude was stated")

        if kind in {"move", "go_to", "return_home", "set_altitude", "orbit"} and action.get("speed_mps") is None:
            action["speed_mps"] = DEFAULT_SPEED_MPS
            _note(assumptions, "2 m/s used where movement speed was not stated")

        if kind == "move":
            if action.get("direction") is None and action.get("heading_deg") is None:
                action["direction"] = "forward"
                _note(assumptions, "forward/current-heading used for unspecified move direction")
            if action.get("duration_s") is None and action.get("distance_m") is None:
                action["distance_m"] = DEFAULT_MOVE_DISTANCE_M
                _note(assumptions, "8 m used for unspecified move distance")

        if kind == "turn" and action.get("heading_deg") is None and action.get("turn_degrees") is None:
            action["turn_degrees"] = DEFAULT_TURN_DEGREES
            _note(assumptions, "90° right used for unspecified turn")

        if kind == "hover" and action.get("duration_s") is None:
            action["duration_s"] = DEFAULT_HOVER_SECONDS
            _note(assumptions, "5 s used for unspecified hover duration")

        if kind == "orbit":
            # In clean-slate adaptive mode there is no preselected asset center.
            # A bare orbit must be represented as part of find_and_inspect instead
            # of silently inventing a known building location.
            if action.get("target") is None and not adaptive_inspection:
                action["target"] = "asset_center"
                _note(assumptions, "asset center used for unspecified orbit target")
            if action.get("radius_m") is None:
                action["radius_m"] = DEFAULT_ORBIT_RADIUS_M
                _note(assumptions, "6 m radius used for unspecified orbit size")
            if action.get("altitude_m") is None:
                action["altitude_m"] = DEFAULT_ORBIT_ALTITUDE_M
                _note(assumptions, "8 m used for unspecified orbit altitude")
            if action.get("laps") is None:
                action["laps"] = 1
                _note(assumptions, "one lap used where orbit count was not stated")
            if action.get("orbit_direction") is None:
                action["orbit_direction"] = "clockwise"
                _note(assumptions, "clockwise used where orbit direction was not stated")

        if kind == "inspect_corners":
            if action.get("corner_strategy") is None:
                action["corner_strategy"] = "nearest_safe"
                _note(assumptions, "ARIS may choose the safest/nearest requested corner first")
            if action.get("evidence_requirement") is None:
                action["evidence_requirement"] = "clear_photo"
                _note(assumptions, "clear optically verified corner photographs required")
            if action.get("standoff_m") is None:
                action["standoff_m"] = 7.5
                _note(assumptions, "7.5 m preferred inspection standoff used")
            if action.get("altitude_m") is None:
                action["altitude_m"] = 9.0
                _note(assumptions, "9 m inspection altitude used")
            if action.get("speed_mps") is None:
                action["speed_mps"] = DEFAULT_SPEED_MPS

        if kind == "inspect_asset":
            mode = action.get("coverage_mode") or "all_sides"
            action["coverage_mode"] = mode
            if mode == "all_sides":
                action["inspection_targets"] = ["north_facade", "east_facade", "south_facade", "west_facade"]
            elif mode == "full_exterior":
                action["inspection_targets"] = ["north_facade", "east_facade", "south_facade", "west_facade", "roof"]
            if action.get("coverage_strategy") is None:
                action["coverage_strategy"] = "nearest_safe"
                _note(assumptions, "ARIS may choose the safest/nearest requested asset surface first")
            if action.get("evidence_requirement") is None:
                action["evidence_requirement"] = "clear_photo"
                _note(assumptions, "clear optically verified surface coverage required")
            if action.get("standoff_m") is None:
                action["standoff_m"] = 7.5
                _note(assumptions, "7.5 m preferred inspection standoff used")
            if action.get("altitude_m") is None:
                action["altitude_m"] = 9.0
                _note(assumptions, "9 m inspection altitude used")
            if action.get("speed_mps") is None:
                action["speed_mps"] = DEFAULT_SPEED_MPS

        if kind == "find_and_inspect":
            if action.get("object_selection") is None:
                action["object_selection"] = "nearest"
                _note(assumptions, "nearest matching discovered object used where no selection rule was stated")
            if action.get("discovered_inspection_style") is None:
                action["discovered_inspection_style"] = "photograph"
                _note(assumptions, "single clear photograph used where inspection style was not stated")
            if action.get("evidence_requirement") is None:
                action["evidence_requirement"] = "clear_photo"
            if action.get("standoff_m") is None:
                action["standoff_m"] = 7.5
            if action.get("altitude_m") is None:
                action["altitude_m"] = 9.0
            if action.get("speed_mps") is None:
                action["speed_mps"] = DEFAULT_SPEED_MPS
            if action.get("altitude_policy") is None:
                action["altitude_policy"] = "adaptive"
                _note(assumptions, "adaptive altitude recovery enabled for clean-slate mission")

    # Demo missions must terminate in a known safe landed state.
    if actions and actions[-1].get("kind") != "land":
        if not any(a.get("kind") == "return_home" for a in actions[-2:]):
            actions.append(_null_action("return_home", speed_mps=DEFAULT_SPEED_MPS))
        actions.append(_null_action("land"))
        _note(assumptions, "return-home and land appended by ARIS bounded-mission policy")

    data["actions"] = actions
    data["assumptions"] = assumptions
    return data


BASE_MISSION_INTERPRETER_INSTRUCTIONS = """
You are the semantic mission interpreter for Devoniq ARIS, controlling ONLY a
benign civilian simulated drone inside a bounded PyBullet test environment.

The user speaks ordinary natural language. They do NOT know or need to know the
internal action schema. Understand meaning, paraphrases, conversational wording,
relative language, sequencing and common unit expressions. Do not require magic
phrases and do not key behavior to example sentences.

Translate the request into the ARIS Mission IR actions below. You interpret intent;
you do NOT directly fly the aircraft, emit raw x/y coordinates, write source code,
or select prerecorded trajectories.

GENERAL ACTIONS
- takeoff: take off to altitude_m.
- move: travel in a direction or heading for either distance_m OR duration_s.
- turn: rotate by signed turn_degrees (positive=right/clockwise, negative=left)
  OR turn to absolute heading_deg.
- set_altitude: climb/descend to altitude_m.
- hover: remain stationary for duration_s.
- orbit: circle a semantic target using radius_m, laps, orbit_direction and optionally altitude_m.
- go_to: fly to a semantic target. Available targets are home, current_position,
  asset_center, asset_north, asset_south, asset_east and asset_west.
- return_home: return to launch point.
- land: land at the current position.

RELATIVE MEANING
- "straight", "ahead", "keep going" => move direction=forward.
- "back", "backward" => backward.
- "left/right" used as translation => move left/right.
- "turn left/right" => turn action.
- In ordinary known-asset flight mode, "around/circle/orbit the building" may use asset_center.
- In ordinary known-asset flight mode, named building sides may use the corresponding asset target.
- "come back", "go back to where we started", "return" => return_home.
- Qualitative language such as "a little", "slowly", "wide", "small", "briefly"
  may be converted into sensible bounded values, but every such interpretation
  must be disclosed in assumptions.

UNITS
Convert common user units to SI:
- mph * 0.44704 => m/s
- km/h * 0.2777778 => m/s
- feet * 0.3048 => metres
- minutes * 60 => seconds

CURRENT DEMO SAFETY ENVELOPE
- altitude <= 20 m
- speed <= 5 m/s
- one timed move/hover <= 30 s
- one move distance <= 40 m
- orbit radius <= 12 m
- orbit laps <= 5
ARIS independently enforces these after you respond.

AMBIGUITY
Use disposition="clarify" when a material reference cannot be resolved from the
available world context. Ask one short clarification_question. Do not invent a location.

Use disposition="reject" for requests involving pursuit/tracking of a person,
weapon use, targeting, attack, evasion, or anything outside these benign simulated
flight/inspection capabilities.

Use disposition="execute" for representable benign missions. Preserve the user's
sequence. `repeat` may repeat one action 1-5 times.

Every MissionAction field MUST be present in the JSON response. Use null for fields
that do not apply. Every top-level field must also be present.
"""

ADAPTIVE_INSPECTION_EXTENSION = """
This run enables ARIS clean-slate discovery plus adaptive visual inspection in a
complex industrial site.

CRITICAL WORLD-MODEL RULE
At mission start ARIS knows ONLY its launch position, flight boundary, its own
capabilities and the object classes the perception system can recognize. It is NOT
told where any building, tree, tower, container or other site object is located.
Do not assume a preselected building or emit preprogrammed asset coordinates.

ADAPTIVE ACTIONS

CLEAN-SLATE DISCOVERY
- find_and_inspect: search the environment for an object class, select matching
  discovered object(s), then inspect them.
  object_class: building, tree, structure, object.
  object_selection: nearest, largest, all.
  discovered_inspection_style:
    - photograph = obtain a clear documented view.
    - orbit = circle the discovered object and document it.
    - all_sides = collect coverage from four directions around the discovered object.
  evidence_requirement=clear_photo.
  ARIS performs the search, creates the discovered-object world model, generates
  all coordinates/viewpoints and executes obstacle-aware routes.
  altitude_policy:
    - adaptive = ARIS may climb or descend inside the validated 5-20 m envelope when
      doing so creates a safer/reachable route. Use this when the user says to avoid
      obstacles, find a way, adjust altitude, complete autonomously, or does not demand
      a fixed inspection altitude.
    - fixed = maintain the requested altitude for the autonomous leg. Use only when the
      user explicitly says to stay/hold/remain at that altitude.

AUTONOMY / GOAL-REGION RULE
For discovery missions RS describes the semantic goal, not a single exact waypoint.
"Fly to it" means reach any safe approach region around the discovered object.
"Circle it" means produce safe orbital coverage around its discovered geometry.
ARIS is allowed to choose the actual approach point, radius, route and (with adaptive
altitude_policy) altitude. Do not encode a rigid point merely because the user used
ordinary language such as "fly to it".

SEMANTIC RULES FOR UNKNOWN ENVIRONMENTS
- "find a tree", "locate a tree", "find the nearest tree and photograph it" =>
  find_and_inspect(object_class=tree, object_selection=nearest, style=photograph).
- "find the largest building and inspect all sides" => find_and_inspect(building,
  largest, all_sides).
- "find all buildings" => find_and_inspect(building, all, photograph) unless the
  user requests a different inspection style.
- Generic "inspect the building" in this clean-slate mode MUST NOT assume a known
  building position. Interpret it as discover/select a building first, normally the
  nearest matching building unless the user supplies another selection rule.
- "circle the nearest building" => find_and_inspect(building, nearest, orbit).
- "inspect the building from all directions" => if the asset is already selected, use full facade/surface coverage; in clean-slate mode discover/select the building first and then inspect all sides.
- "find a tree and orbit it" => find_and_inspect(tree, nearest, orbit).
- Compose multi-step requests semantically. A request that combines takeoff, discovery of an unknown object class, approach, obstacle avoidance, altitude freedom, an orbit/circle instruction, return-home and landing should be represented as takeoff + find_and_inspect with the requested discovery/inspection semantics + return_home + land. Do not depend on an exact phrase or sentence template.

LEGACY EXPLICIT-ASSET ACTIONS
inspect_asset and inspect_corners remain supported so existing missions are not
broken, but use them only when the user explicitly refers to an already selected /
known asset or explicitly requests its named surfaces/corners. They are NOT the
fallback for generic discovery requests.
For an already selected/known asset, never translate "all directions" into corners;
use facade/surface coverage. In clean-slate generic wording, discover/select the
asset first, then apply the requested coverage style.

Do not reject a benign mission merely because the target object has not been named
in advance. If the requested class is building/tree/structure/object, use discovery.
Do not emit object coordinates. RS describes semantics only; ARIS discovers the
objects and creates the world model at runtime.

ADAPTIVE TARGET RESOLUTION
- In clean-slate mode, generic references such as "a building", "the building", "a tree",
  "the object", or "it" after a discovery request refer to the runtime-discovered object,
  not asset_center or any other preprogrammed coordinate.
- Do not emit orbit/go_to actions aimed at asset_center for an object that must first be
  discovered. Fold approach/orbit/coverage into find_and_inspect.
- Preserve user ordering and constraints compositionally; do not pattern-match a whole
  command to a canned mission.
"""



class GPTManeuverInterpreter:
    """Free-form natural language -> semantic ARIS Mission IR. No canned fallback."""

    def __init__(self, client: ResponsesJSON | None = None):
        self.client = client or ResponsesJSON(
            os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MISSION_MODEL")
        )

    async def interpret(self, prompt: str, adaptive_inspection: bool = False) -> tuple[NaturalLanguageMission, dict]:
        instructions = BASE_MISSION_INTERPRETER_INSTRUCTIONS
        if adaptive_inspection:
            instructions += "\n" + ADAPTIVE_INSPECTION_EXTENSION
        raw, meta = await self.client.generate_with_meta(
            instructions,
            [{"type": "input_text", "text": json.dumps({
                "objective": prompt,
                "world_context": {
                    "available_named_targets": (["home"] if adaptive_inspection else [
                        "home", "asset_center", "asset_north", "asset_south", "asset_east", "asset_west"
                    ]),
                    "discoverable_object_classes": (["building", "tree", "structure", "object"] if adaptive_inspection else []),
                    "preknown_object_locations": [] if adaptive_inspection else ["asset_center"],
                    "adaptive_obstacle_replanning_available": adaptive_inspection,
                    "runtime_object_discovery_available": adaptive_inspection,
                    "rendered_camera_evidence_available": adaptive_inspection,
                    "initial_state": "on ground at home; no site objects selected or located" if adaptive_inspection else "on ground at home",
                },
            })}],
            NaturalLanguageMission.model_json_schema(),
            schema_name="aris_unified_mission_ir",
        )

        raw = _normalize_intent(raw, adaptive_inspection=adaptive_inspection)
        mission = NaturalLanguageMission.model_validate(raw)

        if mission.disposition == "clarify":
            raise ClarificationRequired(
                "ARIS needs clarification: " + str(mission.clarification_question)
            )
        if mission.disposition == "reject":
            raise ValueError("Unsupported mission: " + mission.summary)

        return mission, meta


async def verify_gpt_link(client: ResponsesJSON | None = None) -> dict:
    c = client or ResponsesJSON(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MISSION_MODEL"))
    nonce = "ARIS-LINK-" + uuid.uuid4().hex
    raw, meta = await c.generate_with_meta(
        "This is a connectivity challenge. Return ok=true, echo the nonce exactly, and a short message saying the ARIS GPT mission link is live. Do nothing else.",
        [{"type": "input_text", "text": nonce}],
        GPTLinkCheck.model_json_schema(),
        schema_name="aris_gpt_link_check",
    )
    check = GPTLinkCheck.model_validate(raw)
    if check.ok is not True:
        raise ModelUnavailable("GPT link check did not return ok=true")
    if check.nonce != nonce:
        raise ModelUnavailable("GPT link check returned the wrong challenge nonce")
    return {
        "verified": True,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "model": meta.get("model") or os.getenv("OPENAI_MISSION_MODEL"),
        "response_id": meta.get("response_id"),
        "message": check.message,
    }


class ManeuverCompiler:
    """Semantic Mission IR -> deterministic ARIS aircraft commands."""

    def __init__(
        self,
        home_xy=(-12.0, -12.0),
        asset_center_xy=(0.0, 0.0),
        points_per_lap=24,
        initial_heading_deg=90.0,
    ):
        self.home_xy = tuple(float(v) for v in home_xy)
        self.asset_center_xy = tuple(float(v) for v in asset_center_xy)
        self.points_per_lap = int(points_per_lap)
        self.initial_heading_deg = float(initial_heading_deg) % 360

    @staticmethod
    def _heading(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        return math.degrees(math.atan2(dx, dy)) % 360

    @staticmethod
    def _xy_from_heading(origin, heading_deg, distance_m):
        r = math.radians(heading_deg)
        return (
            origin[0] + math.sin(r) * distance_m,
            origin[1] + math.cos(r) * distance_m,
        )

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
        return {
            "north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0
        }[direction]

    def _target_xy(self, target: TargetRef, current_xy):
        # Asset extents are approximately +/-4m x +/-3m. Named side targets are
        # deliberately outside the building with clearance for this demo scene.
        return {
            "current_position": current_xy,
            "home": self.home_xy,
            "asset_center": self.asset_center_xy,
            "asset_north": (0.0, 8.0),
            "asset_south": (0.0, -8.0),
            "asset_east": (9.0, 0.0),
            "asset_west": (-9.0, 0.0),
        }[target]

    @staticmethod
    def _stop_short(start, target, stop_distance):
        if not stop_distance:
            return target
        dx, dy = target[0] - start[0], target[1] - start[1]
        d = math.hypot(dx, dy)
        if d <= stop_distance or d == 0:
            return start
        ratio = (d - stop_distance) / d
        return (start[0] + dx * ratio, start[1] + dy * ratio)

    def compile(self, prompt: str, mission: NaturalLanguageMission) -> CompiledManeuver:
        commands = []
        current_xy = self.home_xy
        current_alt = 0.0
        current_heading = self.initial_heading_deg

        for action_index, action in enumerate(mission.actions):
            for repetition in range(action.repeat):
                speed = action.speed_mps or DEFAULT_SPEED_MPS
                suffix = f" · repeat {repetition + 1}/{action.repeat}" if action.repeat > 1 else ""

                if action.kind == "takeoff":
                    current_alt = float(action.altitude_m)
                    commands.append(CompiledCommand(
                        kind="takeoff",
                        label=f"Takeoff to {current_alt:g} m{suffix}",
                        altitude_m=current_alt,
                        speed_mps=speed,
                        source_action_index=action_index,
                    ))

                elif action.kind == "move":
                    altitude = float(action.altitude_m) if action.altitude_m is not None else current_alt
                    heading = (
                        float(action.heading_deg)
                        if action.heading_deg is not None
                        else self._direction_heading(current_heading, action.direction)
                    ) % 360
                    distance = (
                        float(action.distance_m)
                        if action.distance_m is not None
                        else float(speed) * float(action.duration_s)
                    )
                    endpoint = self._xy_from_heading(current_xy, heading, distance)
                    descriptor = (
                        f"{action.duration_s:g} s at {speed:g} m/s"
                        if action.duration_s is not None
                        else f"{distance:g} m at {speed:g} m/s"
                    )
                    commands.append(CompiledCommand(
                        kind="straight",
                        label=f"Move {descriptor} · heading {heading:g}°{suffix}",
                        x_m=endpoint[0], y_m=endpoint[1],
                        altitude_m=altitude, heading_deg=heading,
                        speed_mps=speed, duration_s=action.duration_s,
                        distance_m=distance, source_action_index=action_index,
                    ))
                    current_xy = endpoint
                    current_alt = altitude
                    current_heading = heading

                elif action.kind == "turn":
                    if action.heading_deg is not None:
                        current_heading = float(action.heading_deg) % 360
                        label = f"Turn to heading {current_heading:g}°"
                    else:
                        current_heading = (current_heading + float(action.turn_degrees)) % 360
                        side = "right" if action.turn_degrees >= 0 else "left"
                        label = f"Turn {abs(float(action.turn_degrees)):g}° {side}"
                    commands.append(CompiledCommand(
                        kind="turn", label=label + suffix,
                        heading_deg=current_heading,
                        source_action_index=action_index,
                    ))

                elif action.kind == "set_altitude":
                    altitude = float(action.altitude_m)
                    commands.append(CompiledCommand(
                        kind="goto",
                        label=f"Change altitude to {altitude:g} m{suffix}",
                        x_m=current_xy[0], y_m=current_xy[1],
                        altitude_m=altitude, heading_deg=current_heading,
                        speed_mps=speed, distance_m=abs(altitude - current_alt),
                        source_action_index=action_index,
                    ))
                    current_alt = altitude

                elif action.kind == "hover":
                    commands.append(CompiledCommand(
                        kind="hover",
                        label=f"Hover {action.duration_s:g} s{suffix}",
                        duration_s=float(action.duration_s),
                        source_action_index=action_index,
                    ))

                elif action.kind == "go_to":
                    target = self._target_xy(action.target, current_xy)
                    endpoint = self._stop_short(current_xy, target, action.stop_distance_m)
                    heading = self._heading(current_xy, endpoint) if endpoint != current_xy else current_heading
                    altitude = float(action.altitude_m) if action.altitude_m is not None else current_alt
                    commands.append(CompiledCommand(
                        kind="goto",
                        label=f"Go to {action.target.replace('_', ' ')}{suffix}",
                        x_m=endpoint[0], y_m=endpoint[1],
                        altitude_m=altitude, heading_deg=heading,
                        speed_mps=speed, distance_m=math.dist(current_xy, endpoint),
                        source_action_index=action_index,
                    ))
                    current_xy = endpoint
                    current_alt = altitude
                    current_heading = heading

                elif action.kind == "orbit":
                    center = self._target_xy(action.target, current_xy)
                    altitude = float(action.altitude_m) if action.altitude_m is not None else current_alt
                    radius = float(action.radius_m)
                    laps = int(action.laps)
                    sign = -1.0 if action.orbit_direction == "clockwise" else 1.0

                    vx, vy = current_xy[0] - center[0], current_xy[1] - center[1]
                    start_angle = math.atan2(vy, vx) if math.hypot(vx, vy) > 0.25 else 0.0

                    # First point sits on the requested orbit. Remaining points are generated
                    # from parameters at runtime; no prerecorded path exists.
                    prior = current_xy
                    total = laps * self.points_per_lap
                    for i in range(total + 1):
                        theta = start_angle + sign * 2 * math.pi * (i / self.points_per_lap)
                        xy = (
                            center[0] + radius * math.cos(theta),
                            center[1] + radius * math.sin(theta),
                        )
                        heading = self._heading(prior, xy) if xy != prior else current_heading
                        commands.append(CompiledCommand(
                            kind="goto",
                            label=f"Orbit {action.target.replace('_',' ')} · {i + 1}/{total + 1}{suffix}",
                            x_m=xy[0], y_m=xy[1],
                            altitude_m=altitude, heading_deg=heading,
                            speed_mps=speed,
                            source_action_index=action_index,
                        ))
                        prior = xy
                        current_heading = heading
                    current_xy = prior
                    current_alt = altitude

                elif action.kind in {"inspect_corners", "inspect_asset"}:
                    raise ValueError(f"{action.kind} requires adaptive-inspection execution mode")

                elif action.kind == "return_home":
                    heading = self._heading(current_xy, self.home_xy) if current_xy != self.home_xy else current_heading
                    commands.append(CompiledCommand(
                        kind="return_home",
                        label="Return to launch point" + suffix,
                        x_m=self.home_xy[0], y_m=self.home_xy[1],
                        altitude_m=current_alt, heading_deg=heading,
                        speed_mps=speed, distance_m=math.dist(current_xy, self.home_xy),
                        source_action_index=action_index,
                    ))
                    current_xy = self.home_xy
                    current_heading = heading

                elif action.kind == "land":
                    commands.append(CompiledCommand(
                        kind="land",
                        label="Land at current position" + suffix,
                        altitude_m=0,
                        source_action_index=action_index,
                    ))
                    current_alt = 0.0

        return CompiledManeuver(
            prompt=prompt,
            ai_explanation=mission.summary,
            defaults_applied=mission.assumptions,
            commands=commands,
        )


class ManeuverSafetyGate:
    """Final authority over concrete commands. GPT output cannot bypass this."""

    min_altitude_m = 5.0
    max_altitude_m = MAX_ALTITUDE_M
    max_speed_mps = MAX_SPEED_MPS
    max_commands = 300

    def __init__(self, geo_fn, home_xy=(-12.0, -12.0)):
        self.geo = geo_fn
        self.home_xy = tuple(home_xy)

    def validate(self, asset, compiled: CompiledManeuver):
        if not compiled.commands or len(compiled.commands) > self.max_commands:
            raise ValueError("SafetyGate: invalid compiled command count")

        airborne = False
        last_xy = self.home_xy
        last_alt = 0.0
        has_land = False

        for c in compiled.commands:
            if c.speed_mps is not None and not (0 < c.speed_mps <= self.max_speed_mps):
                raise ValueError(
                    f"SafetyGate: requested speed {c.speed_mps:g} m/s exceeds configured {self.max_speed_mps:g} m/s limit"
                )
            if c.duration_s is not None and c.duration_s > MAX_DURATION_S:
                raise ValueError(
                    f"SafetyGate: requested duration {c.duration_s:g} s exceeds configured {MAX_DURATION_S:g} s limit"
                )
            if c.kind == "straight" and c.distance_m is not None and c.distance_m > MAX_MOVE_DISTANCE_M:
                raise ValueError(
                    f"SafetyGate: requested move distance {c.distance_m:g} m exceeds configured {MAX_MOVE_DISTANCE_M:g} m limit"
                )

            if c.kind == "takeoff":
                if airborne:
                    raise ValueError("SafetyGate: duplicate takeoff")
                if c.altitude_m is None or not self.min_altitude_m <= c.altitude_m <= self.max_altitude_m:
                    raise ValueError("SafetyGate: takeoff altitude outside 5-20 m")
                airborne = True
                last_alt = c.altitude_m
                continue

            if c.kind in {"goto", "straight", "return_home"}:
                if not airborne:
                    raise ValueError("SafetyGate: movement requested before takeoff")
                if c.x_m is None or c.y_m is None or c.altitude_m is None:
                    raise ValueError("SafetyGate: movement missing concrete coordinates")
                if not self.min_altitude_m <= c.altitude_m <= self.max_altitude_m:
                    raise ValueError("SafetyGate: movement altitude outside 5-20 m")
                a = self.geo(last_xy[0], last_xy[1])
                b = self.geo(c.x_m, c.y_m)
                if not point_allowed(asset, b):
                    raise ValueError("SafetyGate: requested movement would leave the configured flight boundary")
                if not segment_allowed(asset, a, b, min(last_alt or c.altitude_m, c.altitude_m)):
                    raise ValueError("SafetyGate: requested movement intersects the building, exclusion zone, or geofence")
                last_xy = (c.x_m, c.y_m)
                last_alt = c.altitude_m

            elif c.kind == "turn":
                if not airborne:
                    raise ValueError("SafetyGate: turn requested before takeoff")
                if c.heading_deg is None:
                    raise ValueError("SafetyGate: turn missing concrete heading")

            elif c.kind == "hover":
                if not airborne:
                    raise ValueError("SafetyGate: hover requested before takeoff")

            elif c.kind == "land":
                if not airborne:
                    raise ValueError("SafetyGate: land requested while not airborne")
                airborne = False
                has_land = True
                last_alt = 0.0

        if airborne or not has_land:
            raise ValueError("SafetyGate: mission must end with a landing")
        return True


class ManeuverExecutor:
    """Executes only ARIS-compiled commands after SafetyGate approval."""

    def __init__(self, journal, runtime_check, stop_event: asyncio.Event | None = None):
        self.journal = journal
        self.runtime_check = runtime_check
        self.stop_event = stop_event or asyncio.Event()

    async def _guarded(self, drone, operation, timeout=90):
        task = asyncio.create_task(operation)
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            while True:
                if self.stop_event.is_set():
                    raise RuntimeError("Operator requested abort")
                telemetry = await drone.telemetry()
                self.runtime_check(telemetry)
                if task.done():
                    return await task
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Maneuver operation timed out")
                await asyncio.wait({task}, timeout=min(0.1, remaining))
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run(self, compiled: CompiledManeuver, drone):
        self.journal("mission_status", status="VALIDATED", mission_kind="natural_language_flight")
        await drone.connect()
        try:
            self.journal("mission_status", status="RUNNING", mission_kind="natural_language_flight")

            for index, command in enumerate(compiled.commands, start=1):
                if self.stop_event.is_set():
                    raise RuntimeError("Operator requested abort")

                self.journal(
                    "aris_command",
                    command_index=index,
                    command_total=len(compiled.commands),
                    command=command.model_dump(mode="json"),
                    executor="ManeuverExecutor",
                    adapter="PhysicsDrone",
                )

                if command.kind == "takeoff":
                    await self._guarded(drone, drone.takeoff(command.altitude_m))

                elif command.kind in {"goto", "return_home"}:
                    await self._guarded(
                        drone,
                        drone.go_to_local(
                            command.x_m, command.y_m, command.altitude_m,
                            command.heading_deg, command.speed_mps, command.label,
                        ),
                    )

                elif command.kind == "straight":
                    if command.duration_s is not None:
                        await self._guarded(
                            drone,
                            drone.fly_straight_local(
                                command.x_m, command.y_m, command.altitude_m,
                                command.heading_deg, command.speed_mps,
                                command.duration_s, command.label,
                            ),
                            timeout=command.duration_s + 20,
                        )
                    else:
                        await self._guarded(
                            drone,
                            drone.go_to_local(
                                command.x_m, command.y_m, command.altitude_m,
                                command.heading_deg, command.speed_mps, command.label,
                            ),
                        )

                elif command.kind == "turn":
                    await self._guarded(
                        drone,
                        drone.turn_to_heading(command.heading_deg, command.label),
                        timeout=20,
                    )

                elif command.kind == "hover":
                    await drone.hold()
                    await self._guarded(
                        drone, asyncio.sleep(command.duration_s),
                        timeout=command.duration_s + 10,
                    )

                elif command.kind == "land":
                    await self._guarded(drone, drone.land())

            telemetry = await drone.telemetry()
            if telemetry.airborne:
                raise RuntimeError("Mission ended but aircraft is still airborne")

            self.journal("mission_status", status="COMPLETE", mission_kind="natural_language_flight")
            return {"status": "COMPLETE", "landed_confirmed": True}

        except Exception as exc:
            self.journal("mission_execution_error", error=type(exc).__name__, message=str(exc)[:240])
            try:
                await drone.abort()
            except Exception as recovery:
                self.journal("recovery_error", error=type(recovery).__name__)

            status = "ABORTED" if "abort" in str(exc).lower() else "FAILED"
            self.journal("mission_status", status=status, mission_kind="natural_language_flight")
            try:
                landed_confirmed = not (await drone.telemetry()).airborne
            except Exception:
                landed_confirmed = False
            return {
                "status": status,
                "landed_confirmed": landed_confirmed,
                "error": str(exc),
            }
        finally:
            await drone.close()
