import hashlib
from pathlib import Path
from pydantic import BaseModel


class EvidenceResult(BaseModel):
    requirement_id: str
    label: str
    satisfied: bool
    score: float
    reason: str
    best_observation_id: str | None = None


def intact(o):
    try:
        return (
            hashlib.sha256(Path(o.image_path).read_bytes()).hexdigest()
            == o.image_sha256
        )
    except OSError:
        return False


def supported(o):
    a = o.assessment
    return (
        o.accepted
        and o.quality_passed
        and o.source != "synthetic"
        and a.get("status") == "assessed"
        and a.get("target_visible") is True
        and a.get("criterion_met") is True
        and a.get("image_sha256") == o.image_sha256
        and intact(o)
    )


class EvidenceVerifier:
    def verify_requirement(self, req, observations, security_events):
        def result(ok, reason, best=None):
            return EvidenceResult(
                requirement_id=req.id,
                label=req.label,
                satisfied=ok,
                score=best.quality_score if best else float(ok),
                reason=reason,
                best_observation_id=best.id if best else None,
            )

        if req.kind == "visual_coverage":
            candidates = [
                o
                for o in observations
                if o.requirement_id == req.id and o.label == req.target
            ]
            valid = [
                o
                for o in candidates
                if supported(o) and o.quality_score >= req.min_quality
            ]
            if valid:
                return result(
                    True,
                    "Technical and semantic evidence checks passed; this is not measured surface-area coverage.",
                    max(valid, key=lambda o: o.quality_score),
                )
            return result(
                False,
                "Missing, synthetic, altered, low-quality, uncertain or semantically insufficient evidence.",
            )
        if req.kind == "security_presence_log":
            # Latest capture for each view needs a determinate assessment; all positive events remain logged.
            latest = {}
            for o in observations:
                latest[o.requirement_id] = o
            ok = bool(latest) and all(
                o.source != "synthetic"
                and o.assessment.get("status") == "assessed"
                and o.person_present is not None
                and intact(o)
                for o in latest.values()
            )
            logged = {e.evidence_sha256 for e in security_events}
            ok = ok and all(
                not o.person_present or o.image_sha256 in logged for o in observations
            )
            return result(
                ok,
                "Presence assessment and positive-event traceability checked."
                if ok
                else "Presence assessment missing, uncertain or not logged.",
            )
        if req.kind == "finding_traceability":
            ok = bool(observations) and all(
                intact(o) and o.assessment.get("image_sha256") == o.image_sha256
                for o in observations
                if o.finding
            )
            return result(
                ok,
                "Finding source hashes checked."
                if ok
                else "Missing or altered evidence.",
            )
        return result(False, "Unsupported requirement")
