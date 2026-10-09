from .asset_intelligence import AssetIntelligence


def build_report(store, mission_id):
    plan = store.get_mission(mission_id)
    if not plan:
        raise KeyError(mission_id)
    obs = store.observations_for_mission(mission_id)
    events = store.security_events_for_mission(mission_id)
    state = store.completion_state(mission_id)
    return {
        "mission_id": plan.id,
        "asset_id": plan.asset_id,
        "status": plan.status.value,
        "mode": plan.adapter,
        "evidence_domain": "simulation" if plan.adapter == "physics" else "physical_or_recorded",
        "objective": plan.prompt,
        "error": plan.error,
        "recovery_error": plan.recovery_error,
        "landed_confirmed": plan.landed_confirmed,
        "requirements_satisfied_percent": state.requirements_satisfied_percent
        if state
        else 0,
        "evidence_requirements": [r.model_dump() for r in state.results]
        if state
        else [],
        "observations_total": len(obs),
        "images_accepted": sum(o.accepted for o in obs),
        "images_rejected": sum(not o.accepted for o in obs),
        "reinspection_observations": sum(o.attempt > 1 for o in obs),
        "findings_requiring_human_review": [
            {
                "observation_id": o.id,
                "finding": o.finding,
                "image_path": o.image_path,
                "image_sha256": o.image_sha256,
                "provider": o.assessment.get("provider"),
            }
            for o in obs
            if o.accepted and o.finding
        ],
        "security_events": [e.model_dump() for e in events],
        "perception_states": [
            {
                "observation_id": o.id,
                "status": o.assessment.get("status"),
                "provider": o.assessment.get("provider"),
                "issues": o.quality_issues + o.assessment.get("issues", []),
            }
            for o in obs
        ],
        "limitations": plan.mission_spec.limitations
        + [
            "Percentage counts satisfied requirements; it is not measured physical surface coverage.",
            "Replay uses operator-supplied capture metadata; neither replay nor cloud vision authenticates asset identity.",
            "No field-calibrated accuracy claim or flight-readiness certification.",
        ],
    }


def build_asset_history(store, asset_id):
    return AssetIntelligence(store).snapshot(asset_id)
