"""Detection Tools Package

TOOLs are responsible for:
  - Person Detection (YOLO + Kalman Tracker, ID: 0~9999)
  - Object Detection (OmDet-Turbo + Kalman Tracker, ID: 10000~)
  - Pose Estimation (YOLO11l-Pose)
  - Depth Estimation (Depth Anything V2)

TOOLs do NOT make decisions - they provide structured data.
"""
# modules/tools/__init__.py
from .object_detector import PersonDetector, ObjectDetector
from .pose_estimator import PoseEstimator
from .depth_estimator import DepthEstimator

__all__ = [
    "PersonDetector",
    "ObjectDetector",
    "PoseEstimator",
    "DepthEstimator",
]
