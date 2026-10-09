from __future__ import annotations
from enum import Enum
from typing import Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .mission_schema import MissionSpec


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LatLon(StrictModel):
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)


class ApprovedViewpoint(LatLon):
    altitude_m: float = Field(ge=10, le=30)
    heading_deg: float = Field(ge=0, lt=360)
    gimbal_pitch_deg: float = Field(default=-35, ge=-90, le=0)


class AssetCreate(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = "building"
    boundary: list[LatLon] = Field(min_length=3, max_length=100)
    flight_boundary: list[LatLon] = Field(min_length=3, max_length=100)
    exclusion_zones: list[list[LatLon]] = Field(default_factory=list, max_length=30)
    height_m: float = Field(gt=0, le=100)
    alternate_views: dict[str, list[ApprovedViewpoint]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def polygons(self):
        from .geo import polygon

        footprint, flight = polygon(self.boundary), polygon(self.flight_boundary)
        if not flight.covers(footprint):
            raise ValueError("Flight boundary must contain the asset footprint")
        for zone in self.exclusion_zones:
            polygon(zone)
        if any(
            k
            not in {
                "north_facade",
                "east_facade",
                "south_facade",
                "west_facade",
                "roof",
            }
            or len(v) > 4
            for k, v in self.alternate_views.items()
        ):
            raise ValueError(
                "Invalid alternate view label or more than four alternates"
            )
        return self


class Asset(AssetCreate):
    id: str


class MissionPlanRequest(StrictModel):
    asset_id: str
    prompt: str = Field(min_length=1, max_length=4000)
    adapter: Literal["sim", "replay", "dji", "parrot", "px4", "physics"] = "sim"
    use_gpt: bool = False


class MissionStatus(str, Enum):
    PLANNED = "PLANNED"
    QUEUED = "QUEUED"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    REINSPECTING = "REINSPECTING"
    RETURNING_HOME = "RETURNING_HOME"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


TERMINAL = {
    MissionStatus.COMPLETE,
    MissionStatus.INCOMPLETE,
    MissionStatus.FAILED,
    MissionStatus.ABORTED,
}


class ObservationTask(StrictModel):
    id: str
    requirement_id: str
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    altitude_m: float = Field(gt=0, le=120)
    heading_deg: float = Field(ge=0, lt=360)
    gimbal_pitch_deg: float = Field(default=-35, ge=-90, le=0)
    label: str = Field(
        pattern=r"^(north_facade|east_facade|south_facade|west_facade|roof)$"
    )
    required_quality: float = Field(default=0.65, ge=0, le=1)
    attempts: int = Field(default=0, ge=0, le=5)
    supplemental: bool = False
    retry_reason: str | None = None
    settle_seconds: float = Field(default=0, ge=0, le=5)


class MissionPlan(StrictModel):
    id: str
    asset_id: str
    prompt: str
    adapter: Literal["sim", "replay", "dji", "parrot", "px4", "physics"]
    speed_mps: float = Field(default=2.0, gt=0, le=4)
    max_altitude_m: float = Field(default=30, gt=0, le=30)
    return_home: Literal[True] = True
    tasks: list[ObservationTask] = Field(min_length=1, max_length=30)
    status: MissionStatus = MissionStatus.PLANNED
    ai_summary: str | None = None
    security_presence_logging: bool = False
    mission_spec: MissionSpec
    error: str | None = None
    recovery_error: str | None = None
    landed_confirmed: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Telemetry(StrictModel):
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    altitude_m: float = Field(ge=0, le=1000)
    battery_percent: float = Field(ge=0, le=100)
    gps_ok: bool
    link_ok: bool
    airborne: bool
    observed_at: datetime

    @model_validator(mode="after")
    def timestamp(self):
        if self.observed_at.tzinfo is None:
            raise ValueError("Telemetry timestamp must have a timezone")
        return self


class Capture(StrictModel):
    image_path: str
    source: Literal["synthetic", "recorded", "dji", "parrot", "px4", "simulated_sensor"]
    captured_at: datetime
    telemetry: Telemetry


class Observation(StrictModel):
    id: str
    mission_id: str
    task_id: str
    requirement_id: str
    asset_id: str
    label: str
    lat: float
    lon: float
    altitude_m: float
    captured_at: str
    image_path: str
    image_sha256: str
    source: Literal["synthetic", "recorded", "dji", "parrot", "px4", "simulated_sensor"]
    quality_score: float
    quality_passed: bool
    quality_issues: list[str] = Field(default_factory=list)
    accepted: bool
    attempt: int
    assessment: dict
    finding: str | None = None
    finding_confidence: float | None = None
    person_present: bool | None = None


class SecurityEvent(StrictModel):
    id: str
    mission_id: str
    asset_id: str
    observed_at: str
    observation_label: str
    lat: float
    lon: float
    event_type: Literal["PERSON_PRESENT"] = "PERSON_PRESENT"
    evidence_image_path: str
    evidence_sha256: str
    provider: str
    note: str = "Non-identifying presence; coordinates describe the camera, not a person location."


class MissionView(BaseModel):
    plan: MissionPlan
    observations: list[Observation]
