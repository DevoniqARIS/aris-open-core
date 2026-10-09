from .change_detection import ChangeDetector
from .models import MissionStatus


class AssetIntelligence:
    def __init__(self, store):
        self.store = store
        self.change_detector = ChangeDetector()

    def snapshot(self, asset_id):
        missions = self.store.missions_for_asset(asset_id)
        summaries = []
        previous = None
        for mission in missions:
            obs = self.store.observations_for_mission(mission.id)
            changes = []
            if mission.status in {MissionStatus.COMPLETE, MissionStatus.INCOMPLETE}:
                if previous is not None:
                    changes = [
                        c.model_dump()
                        for c in self.change_detector.compare(previous, obs)
                    ]
                previous = obs
            summaries.append(
                {
                    "mission_id": mission.id,
                    "status": mission.status.value,
                    "mode": mission.adapter,
                    "accepted_observations": sum(o.accepted for o in obs),
                    "changes_vs_previous": changes,
                }
            )
        return {
            "asset_id": asset_id,
            "inspection_count": sum(
                m.status in {MissionStatus.COMPLETE, MissionStatus.INCOMPLETE}
                for m in missions
            ),
            "missions": summaries,
        }
