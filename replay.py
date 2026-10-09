"""Recorded-photo testing. Does not connect to or control an aircraft."""

import json, shutil
from pathlib import Path
from datetime import datetime, timezone
from .sim import SimDrone
from ..models import Capture, Telemetry
from ..quality import read_image


class ReplayDrone(SimDrone):
    def __init__(self, manifest, **kwargs):
        super().__init__(**kwargs)
        self.manifest = Path(manifest).resolve()
        self.records = json.loads(self.manifest.read_text())
        if not isinstance(self.records, dict):
            raise ValueError("Replay manifest must map view labels to capture records")

    async def capture_image(self, mission_id, task, attempt):
        items = self.records.get(task.label, [])
        if attempt > len(items):
            raise ValueError("No recorded image available for requested attempt")
        item = items[attempt - 1]
        source = (self.manifest.parent / item["file"]).resolve()
        if not source.is_relative_to(self.manifest.parent):
            raise ValueError("Replay image escapes manifest directory")
        read_image(source)
        observed = Telemetry.model_validate(item["telemetry"])
        captured = datetime.fromisoformat(item["captured_at"])
        if captured.tzinfo is None or captured > datetime.now(timezone.utc):
            raise ValueError("Invalid recorded capture time")
        out = self.runtime / "images" / mission_id
        out.mkdir(parents=True, exist_ok=True)
        dst = out / (task.id + source.suffix.lower())
        shutil.copyfile(source, dst)
        return Capture(
            image_path=str(dst.resolve()),
            source="recorded",
            captured_at=captured,
            telemetry=observed,
        )
