import os, json, uuid
from typing import Literal
from pydantic import BaseModel, ConfigDict
from .mission_schema import EvidenceRequirement, MissionSpec
from .ai_client import ResponsesJSON

VIEWS = ["north_facade", "east_facade", "south_facade", "west_facade", "roof"]


class ViewIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal[
        "north_facade", "east_facade", "south_facade", "west_facade", "roof"
    ]
    criterion: str


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool
    explanation: str
    views: list[ViewIntent]
    security_presence_logging: bool


def build_spec(asset, prompt, views, security, summary, criteria=None):
    if not views:
        raise ValueError("Mission requires at least one supported view")
    reqs = [
        EvidenceRequirement(
            id=str(uuid.uuid4()),
            label=v,
            kind="visual_coverage",
            target=v,
            criterion=(criteria or {}).get(
                v,
                "Photograph shows the requested roof or exterior facade, with enough visible surface detail for human visual review.",
            ),
        )
        for v in dict.fromkeys(views)
    ]
    reqs.append(
        EvidenceRequirement(
            id=str(uuid.uuid4()),
            label="finding_traceability",
            kind="finding_traceability",
            target="all_findings",
        )
    )
    if security:
        reqs.append(
            EvidenceRequirement(
                id=str(uuid.uuid4()),
                label="security_presence_log",
                kind="security_presence_log",
                target="observations",
            )
        )
    return MissionSpec(
        summary=summary,
        asset_type=asset.kind,
        objectives=[prompt],
        evidence_requirements=reqs,
        security_presence_logging=security,
    )


class DeterministicMissionInterpreter:
    def interpret(self, asset, prompt):
        text = prompt.lower()
        if not any(k in text for k in ["roof", "exterior", "facade", "perimeter"]):
            raise ValueError(
                "Offline interpreter supports exterior/facade/roof visual inspection only"
            )
        if any(
            k in text
            for k in [
                "track",
                "pursue",
                "follow",
                "identify people",
                "weapon",
                "structural integrity",
                "certify",
                "thermal",
                "measure crack",
                "crack",
                "damage",
                "defect",
                "anomaly",
                "corrosion",
            ]
        ):
            raise ValueError("Requested capability is outside this prototype")
        if "roof only" in text:
            views = ["roof"]
        elif "perimeter" in text and "roof" not in text:
            views = VIEWS[:-1]
        else:
            views = VIEWS if "roof" in text else VIEWS[:-1]
        return build_spec(
            asset,
            prompt,
            views,
            any(k in text for k in ["human", "person", "people", "presence"]),
            "Bounded exterior visual-evidence collection. Offline keyword interpretation; no general mission understanding.",
        )


class GPTMissionInterpreter:
    def __init__(self):
        self.client = ResponsesJSON(
            os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MISSION_MODEL")
        )

    async def interpret(self, asset, prompt):
        raw = await self.client.generate(
            "Translate a civilian inspection request into supported facade/roof photographs and optional non-identifying person presence. "
            "supported must be false for any requested capability beyond visual-evidence collection: defect measurement, structural certification, "
            "thermal analysis, identity, following, pursuit, tactical or weapon functions. Do not silently discard unsupported requirements. "
            "For each view, preserve the requested visible-evidence criteria explicitly. Do not reduce a specific inspection objective to a generic photo. No flight coordinates or instructions. The caller validates the result.",
            [
                {
                    "type": "input_text",
                    "text": json.dumps({"asset_kind": asset.kind, "objective": prompt}),
                }
            ],
            Intent.model_json_schema(),
        )
        intent = Intent.model_validate(raw)
        if not intent.supported:
            raise ValueError("Unsupported mission: " + intent.explanation)
        if len({v.target for v in intent.views}) != len(intent.views):
            raise ValueError(
                "Duplicate view criteria; consolidate into a single criterion per view"
            )
        return build_spec(
            asset,
            prompt,
            [v.target for v in intent.views],
            intent.security_presence_logging,
            intent.explanation,
            {v.target: v.criterion for v in intent.views},
        )
