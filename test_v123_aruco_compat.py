import cv2
import numpy as np
from pathlib import Path
from app.adaptive import CornerEvidence

def marker(marker_id, size=220):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size, canvas, 1)
    return canvas

def test_modern_opencv_aruco_api_is_supported(tmp_path):
    # Current OpenCV wheels can expose ArucoDetector without module-level detectMarkers.
    assert hasattr(cv2.aruco, "ArucoDetector") or hasattr(cv2.aruco, "detectMarkers")

    m1 = marker(24)
    m2 = marker(25)
    canvas = np.full((300, 560), 255, dtype=np.uint8)
    canvas[40:260, 40:260] = m1
    canvas[40:260, 300:520] = m2
    image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    path = tmp_path / "corner_sw.png"
    assert cv2.imwrite(str(path), image)

    result = CornerEvidence().assess(str(path), "corner_sw")
    assert result["accepted"] is True
    assert 24 in result["detected_ids"]
    assert 25 in result["detected_ids"]

def test_no_unconditional_module_level_detectmarkers_call_remains():
    root = Path(__file__).resolve().parents[1]
    source = (root / "app" / "adaptive.py").read_text(encoding="utf-8")
    assert 'if hasattr(cv2.aruco, "ArucoDetector")' in source
