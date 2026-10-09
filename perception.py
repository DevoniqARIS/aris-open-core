"""Optical calibration demo: decode rendered pixels, never simulator labels.

This is real computer vision, not general AI defect understanding. Marker-free
semantic inspection requires the separately selected live OpenAI mode.
"""
import asyncio
import cv2
import numpy as np
from app.quality import read_image
from app.perception import Assessment, OpenAIPerception
from .world import LABELS

class CalibrationVision:
    async def assess(self,path,requirement,asset):
        return await asyncio.to_thread(self._assess,path,requirement)

    def _assess(self,path,requirement):
        image,digest=read_image(path)
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector=cv2.aruco.ArucoDetector(dictionary)
            corners,ids,_=detector.detectMarkers(gray)
        elif hasattr(cv2.aruco, "detectMarkers"):
            corners,ids,_=cv2.aruco.detectMarkers(gray,dictionary)
        else:
            raise RuntimeError("OpenCV ArUco detector API unavailable")
        decoded=set() if ids is None else set(int(i) for i in ids.flatten())
        start=LABELS.index(requirement.target)*4
        expected=set(range(start,start+4))
        found=sorted(decoded&expected)
        good=len(found)>=3
        return Assessment(status='assessed',provider='opencv_aruco_calibration',model='DICT_4X4_50',
            image_sha256=digest,target_visible=bool(found),criterion_met=good,presence='unknown',
            issues=[] if good else ['occluded'],anomalies=[],
            explanation=f'Optical calibration only: decoded {len(found)}/4 expected panels from pixels; '
                        f'3 required. Visible IDs: {found}. Not a general inspection AI assessment.')

class JournalPerception:
    def __init__(self,inner,journal): self.inner,self.journal=inner,journal
    async def assess(self,path,requirement,asset):
        self.journal('assessment_started',label=requirement.target,provider=type(self.inner).__name__)
        a=await self.inner.assess(path,requirement,asset)
        self.journal('assessment',label=requirement.target,**a.model_dump())
        return a
