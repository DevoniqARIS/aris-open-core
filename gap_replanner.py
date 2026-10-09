import uuid


class GapReplanner:
    """Conservative diagnosed retries. No guessed lateral movement around unknown obstacles."""

    def supplemental_task(self, asset, original, attempt, issues):
        if any(
            i in issues
            for i in [
                "perception_unavailable",
                "perception_error",
                "wrong_subject",
                "uncertain",
                "synthetic",
            ]
        ):
            return None
        if not any(
            i in issues
            for i in [
                "blur",
                "low_detail",
                "low_resolution",
                "underexposed",
                "overexposed",
                "occluded",
            ]
        ):
            return None
        if "occluded" in issues:
            approved = asset.alternate_views.get(original.label, [])
            index = attempt - 2
            if index < 0 or index >= len(approved):
                return None
            view = approved[index]
            return original.model_copy(
                update={
                    **view.model_dump(),
                    "id": str(uuid.uuid4()),
                    "attempts": attempt,
                    "supplemental": True,
                    "retry_reason": "occluded: operator-configured alternative",
                    "settle_seconds": 1.0,
                }
            )
        return original.model_copy(
            update={
                "id": str(uuid.uuid4()),
                "attempts": attempt,
                "supplemental": True,
                "retry_reason": ",".join(issues),
                "settle_seconds": 1.0,
            }
        )
