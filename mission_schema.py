from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    label: str
    kind: Literal["visual_coverage", "security_presence_log", "finding_traceability"]
    target: str
    criterion: str = (
        "Requested surface is visible and sufficiently unobscured to review."
    )
    min_quality: float = Field(default=0.65, ge=0, le=1)
    required: bool = True
    max_attempts: int = Field(default=3, ge=1, le=5)


class MissionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    asset_type: str
    objectives: list[str] = Field(min_length=1)
    evidence_requirements: list[EvidenceRequirement] = Field(min_length=1)
    security_presence_logging: bool = False
    completion_rule: Literal["all_required_evidence"] = "all_required_evidence"
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Visual evidence triage only; no structural certification or defect-free guarantee.",
            "View direction derives from capture metadata; visual assessment cannot authenticate an asset identity.",
        ]
    )
