from __future__ import annotations

from pydantic import BaseModel

from .evidence_verifier import EvidenceResult, EvidenceVerifier
from .mission_schema import MissionSpec
from .models import Observation, SecurityEvent


class CompletionState(BaseModel):
    complete: bool
    requirements_satisfied_percent: float
    results: list[EvidenceResult]
    unsatisfied_labels: list[str]


class CompletionEngine:
    def __init__(self, verifier: EvidenceVerifier):
        self.verifier = verifier

    def evaluate(
        self,
        spec: MissionSpec,
        observations: list[Observation],
        security_events: list[SecurityEvent],
    ) -> CompletionState:
        results = [
            self.verifier.verify_requirement(req, observations, security_events)
            for req in spec.evidence_requirements
            if req.required
        ]
        complete = bool(results) and all(r.satisfied for r in results)
        coverage = (
            100.0 * sum(r.satisfied for r in results) / len(results) if results else 0.0
        )
        return CompletionState(
            complete=complete,
            requirements_satisfied_percent=round(coverage, 1),
            results=results,
            unsatisfied_labels=[r.label for r in results if not r.satisfied],
        )
