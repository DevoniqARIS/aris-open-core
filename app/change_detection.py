"""Image-based candidate change screening; never a structural-change certification."""

import cv2, numpy as np
from pydantic import BaseModel
from .quality import read_image


class AssetChange(BaseModel):
    label: str
    status: str
    changed: bool | None = None
    changed_fraction: float | None = None
    comparable_fraction: float | None = None
    matched_features: int = 0
    previous_evidence: str | None = None
    current_evidence: str | None = None
    reason: str


class ChangeDetector:
    def compare_pair(self, previous, current, label):
        base = {
            "label": label,
            "previous_evidence": previous,
            "current_evidence": current,
        }
        try:
            a, ah = read_image(previous)
            b, bh = read_image(current)
            if ah == bh:
                return AssetChange(
                    **base,
                    status="identical_bytes",
                    changed=False,
                    changed_fraction=0,
                    comparable_fraction=1,
                    reason="Identical source bytes; this does not establish a new inspection occurred.",
                )
            a.thumbnail((1024, 1024))
            b.thumbnail((1024, 1024))
            a = cv2.cvtColor(np.asarray(a), cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(np.asarray(b), cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(2500)
            ka, da = orb.detectAndCompute(a, None)
            kb, db = orb.detectAndCompute(b, None)
            if da is None or db is None:
                raise ValueError("Insufficient texture for registration")
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            good = [
                x[0]
                for x in pairs
                if len(x) == 2 and x[0].distance < 0.7 * x[1].distance
            ]
            if len(good) < 20:
                raise ValueError("Insufficient matching features")
            pa = np.float32([ka[m.queryIdx].pt for m in good])
            pb = np.float32([kb[m.trainIdx].pt for m in good])
            h, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3)
            if h is None or mask is None or mask.sum() < 15 or mask.mean() < 0.5:
                raise ValueError("Registration unreliable")
            corners = np.float32(
                [[0, 0], [a.shape[1], 0], [a.shape[1], a.shape[0]], [0, a.shape[0]]]
            ).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, h)
            area = abs(cv2.contourArea(warped_corners)) / (b.shape[0] * b.shape[1])
            if not 0.25 < area < 4:
                raise ValueError("Implausible registration scale")
            warped = cv2.warpPerspective(a, h, (b.shape[1], b.shape[0]))
            valid = (
                cv2.warpPerspective(
                    np.ones(a.shape, dtype=np.uint8), h, (b.shape[1], b.shape[0])
                )
                > 0
            )
            valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(
                bool
            )
            overlap = float(valid.mean())
            if overlap < 0.65:
                raise ValueError("Insufficient overlapping image area")
            # Compensate a global brightness offset, then compare smoothed registered pixels.
            wa = cv2.GaussianBlur(warped, (5, 5), 0).astype(float)
            bb = cv2.GaussianBlur(b, (5, 5), 0).astype(float)
            delta = np.median(bb[valid] - wa[valid])
            diff = np.abs(bb - wa - delta)
            fraction = float(np.mean(diff[valid] > 35))
            return AssetChange(
                **base,
                status="candidate_change" if fraction > 0.03 else "no_change_detected",
                changed=fraction > 0.03,
                changed_fraction=round(fraction, 4),
                comparable_fraction=round(overlap, 4),
                matched_features=int(mask.sum()),
                reason="Registered pixel difference; lighting, viewpoint and occlusion can cause apparent changes. Human review required.",
            )
        except (OSError, ValueError, cv2.error):
            return AssetChange(
                **base,
                status="not_comparable",
                reason="Images unavailable or registration/overlap insufficient; no change conclusion.",
            )

    def compare(self, previous, current):
        previous_domains = {o.source == "simulated_sensor" for o in previous}
        current_domains = {o.source == "simulated_sensor" for o in current}
        if previous_domains != current_domains or len(previous_domains) > 1:
            return []
        def best(items):
            result = {}
            for o in items:
                if (
                    o.accepted
                    and o.source != "synthetic"
                    and (
                        o.label not in result
                        or o.quality_score > result[o.label].quality_score
                    )
                ):
                    result[o.label] = o
            return result

        a, b = best(previous), best(current)
        return [
            self.compare_pair(a[label].image_path, o.image_path, label)
            for label, o in b.items()
            if label in a
        ]
