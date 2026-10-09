"""Presence is taken only from an actual image assessment, never a filename."""


class HumanPresenceDetector:
    def detect(self, assessment):
        if assessment.status != "assessed":
            return None
        return {"present": True, "absent": False, "unknown": None}[assessment.presence]
