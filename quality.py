"""Pixel-based technical quality screening, not a semantic evidence verifier."""

import hashlib
from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000


def read_image(path):
    data = Path(path).read_bytes()
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image missing, empty or larger than 20 MiB")
    with Image.open(BytesIO(data)) as raw:
        if raw.width * raw.height > MAX_PIXELS:
            raise ValueError("Image exceeds pixel limit")
        if raw.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Unsupported image format")
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.load()
    return image, hashlib.sha256(data).hexdigest()


class QualityResult(BaseModel):
    passed: bool
    score: float
    sha256: str
    metrics: dict[str, float]
    issues: list[str] = Field(default_factory=list)


class QualityInspector:
    def assess(self, image_path):
        image, digest = read_image(image_path)
        width, height = image.size
        image.thumbnail((1024, 1024))
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        dark = float(np.mean(gray < 8))
        bright = float(np.mean(gray > 247))
        issues = []
        if min(width, height) < 240:
            issues.append("low_resolution")
        if contrast < 10:
            issues.append("low_detail")
        if sharpness < 40:
            issues.append("blur")
        if dark > 0.6:
            issues.append("underexposed")
        if bright > 0.6:
            issues.append("overexposed")
        score = min(1.0, sharpness / 100, contrast / 30, max(0, 1 - max(dark, bright)))
        return QualityResult(
            passed=not issues,
            score=round(score, 4),
            sha256=digest,
            metrics={
                "width": width,
                "height": height,
                "laplacian_variance": sharpness,
                "contrast_std": contrast,
                "dark_fraction": dark,
                "bright_fraction": bright,
            },
            issues=issues,
        )

    def score(self, image_path):
        return self.assess(image_path).score
