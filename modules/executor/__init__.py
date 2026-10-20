# modules/executor/__init__.py
"""Workflow Executor Package

Exposes the four main executor components used by the safety pipeline:
  StructureLog     — per-run in-memory log with JSON export
  FrameProcessor   — video frame iterator with real-time skip
  EventClassifier  — single-frame VLM inference wrapper
  PipelineExecutor — end-to-end orchestrator (OD → pose → depth → VLM → alert gating)
"""

from .structure_log import StructureLog
from .frame_processor import FrameProcessor
from .event_classifier import EventClassifier
from .pipeline_executor import PipelineExecutor

__all__ = [
    "StructureLog",
    "FrameProcessor",
    "EventClassifier",
    "PipelineExecutor",
]
