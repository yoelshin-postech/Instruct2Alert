# modules/executor/structure_log.py
"""
In-memory structured log that records per-frame pipeline results and flushes to JSON.

Each frame entry stores:
  - Timing metadata (video_time_sec, processing_time_sec)
  - Saved image paths (raw, detection, pose, depth, vlm, roi)
  - Detections: tracker ID, label, confidence, bbox
  - Poses: posture state, body angle, keypoints
  - Depths: per-tracker depth statistics
  - Alerts: OD logic gate result, VLM answer, duration tracking

The log supports:
  - Re-ID history lookup (get_tracker_history)

JSON schema (flushed to db_path on close):
    {
      "metadata": { ... },
      "frames": [
        {
          "frame_id": 1,
          "video_time_sec": 0.033,
          "timestamp": "...",
          "processing_time_sec": 0.12,
          "images": { "raw": "...", "detection": "...", ... },
          "detections": [ { "tracker_id": 1, "label": "person", ... } ],
          "poses": [ { "tracker_id": 1, ... } ],
          "depths": [ { "tracker_id": 1, "mean_depth": 3.5, ... } ],
          "alerts": [ { "rule_index": 0, "OD_logic_triggered": true, "vlm_triggered": true, ... } ]
        },
        ...
      ]
    }
"""

import json
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional
from modules.run_audit import json_default


class StructureLog:
    def __init__(self, db_path: str = ":memory:"):
        """
        Args:
            db_path: File path to write the JSON log on close. Pass ":memory:" to skip disk I/O.
        """
        self.db_path = db_path
        self._in_memory = (db_path == ":memory:")

        # OrderedDict preserves insertion order so frames are serialized in time order.
        self._frames: OrderedDict[int, dict] = OrderedDict()
        self._dirty = False

        print(f"[StructureLog] JSON initialized: {db_path}")

    # ================================================================
    # Internal helpers
    # ================================================================
    def _ensure_frame(self, frame_id: int) -> dict:
        """ Returns the frame entry for frame_id, creating it with default values if missing.

        All log_* methods call this to ensure a frame entry exists before appending
        to its sub-lists, so callers don't need to check for existence themselves.
        """
        if frame_id not in self._frames:
            self._frames[frame_id] = {
                "frame_id": frame_id,
                "video_time_sec": 0.0,
                "delta_t_sec": 0.0,
                "long_gap_detected": False,
                "long_gap_action": "none",
                "video_time_str": "0.000s",
                "timestamp": None,
                "processing_time_sec": 0.0,
                "images": {},
                "frame_summary": {
                    "vlm_called": False,
                    "vlm_results": [],
                },
                "detections": [],
                "poses": [],
                "depths": [],
                "alerts": [],
            }
        return self._frames[frame_id]

    def _save(self):
        """ Flushes the in-memory log to disk if the path is not ":memory:" and data is dirty. """
        if self._in_memory or not self._dirty:
            return
        data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "json_path": self.db_path,
                "total_frames": len(self._frames),
            },
            "frames": list(self._frames.values()),
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)
        self._dirty = False

    # ================================================================
    # Write API
    # ================================================================
    def log_frame(self, frame_id: int, video_time_sec: float,
                  processing_time_sec: float = 0.0, delta_t_sec: float = 0.0,
                  long_gap_detected: bool = False, long_gap_action: str = "none"):
        entry = self._ensure_frame(frame_id)
        entry["video_time_sec"] = round(video_time_sec, 4)
        entry["video_time_str"] = f"{video_time_sec:.3f}s"
        entry["delta_t_sec"] = round(delta_t_sec, 6)
        entry["long_gap_detected"] = bool(long_gap_detected)
        entry["long_gap_action"] = long_gap_action
        entry["timestamp"] = datetime.now().isoformat()
        entry["processing_time_sec"] = round(processing_time_sec, 4)
        self._dirty = True

    def log_frame_image(self, frame_id: int, image_type: str, file_path: str):
        """ Records the path to a saved frame image.

        Args:
            frame_id:   Frame index.
            image_type: 'raw' | 'detection' | 'pose' | 'depth' | 'vlm'
            file_path:  Absolute or relative path to the saved image file.
        """
        entry = self._ensure_frame(frame_id)
        entry["images"][image_type] = file_path
        self._dirty = True

    def log_detection(self, frame_id: int, tracker_id: int, label: str,
                      confidence: float, bbox: List[int]):
        entry = self._ensure_frame(frame_id)
        x1, y1, x2, y2 = bbox
        entry["detections"].append({
            "tracker_id": tracker_id,
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": [x1, y1, x2, y2],
            "video_time_sec": entry["video_time_sec"],
        })
        self._dirty = True

    def log_pose(self, frame_id: int, tracker_id: int, posture: Optional[str],
                 body_angle: Optional[float], confidence: Optional[float],
                 keypoints: Optional[dict], pose_detail: Optional[dict] = None):
        """ Records pose estimation results for a tracked person. """
        entry = self._ensure_frame(frame_id)
        pose_record = {
            "tracker_id": tracker_id,
            "posture": posture,
            "body_angle": round(body_angle, 2) if body_angle is not None else None,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "keypoints": keypoints,
            "video_time_sec": entry["video_time_sec"],
        }
        if pose_detail:
            pose_record["pose_detail"] = pose_detail
        entry["poses"].append(pose_record)
        self._dirty = True

    def log_depth(self, frame_id: int, tracker_id: int, label: Optional[str],
                  mean_depth: Optional[float], median_depth: Optional[float],
                  min_depth: Optional[float] = None, max_depth: Optional[float] = None):
        """ Records depth estimation statistics for a tracked object. """
        entry = self._ensure_frame(frame_id)
        entry["depths"].append({
            "tracker_id": tracker_id,
            "label": label,
            "mean_depth": round(mean_depth, 4) if mean_depth is not None else None,
            "median_depth": round(median_depth, 4) if median_depth is not None else None,
            "min_depth": round(min_depth, 4) if min_depth is not None else None,
            "max_depth": round(max_depth, 4) if max_depth is not None else None,
            "video_time_sec": entry["video_time_sec"],
        })
        self._dirty = True

    def log_alert_evaluation(self, frame_id: int, rule_index: int,
                             condition: str, od_logic_triggered: bool,
                             vlm_triggered: Optional[bool] = None,
                             vlm_raw_response: Optional[str] = None,
                             vlm_latency_sec: Optional[float] = None,
                             duration_met_sec: Optional[float] = None,
                             negative_duration_sec: Optional[float] = None,
                             alert_state: Optional[bool] = None,
                             parser_status: Optional[str] = None,
                             parser_failure: bool = False,
                             binary_evidence: Optional[int] = None,
                             api_success: Optional[bool] = None,
                             api_error: Optional[str] = None,
                             grounding: Optional[dict] = None,
                             details: str = ""):
        """ Records the full evaluation result for one alert rule on one frame.

        Separates the OD logic gate from the VLM decision so each can be
        analyzed independently:
          - OD_logic_triggered: True when the OD/geometry condition was met.
          - vlm_triggered:      True/False when VLM was called; None when skipped.
          - vlm_yes_duration_sec: cumulative duration the VLM has answered "Yes".

        Args:
            frame_id:           Frame index.
            rule_index:         Index of the alert rule (0-based).
            condition:          Condition string from the workflow JSON.
            od_logic_triggered: Whether the OD precondition was satisfied.
            vlm_triggered:      VLM binary answer (True=Yes, False=No, None=not called).
            vlm_raw_response:   Raw text response from the VLM.
            vlm_latency_sec:    VLM inference latency in seconds.
            duration_met_sec:   Cumulative "Yes" duration used for temporal gating.
            details:            Freeform diagnostic string.
        """
        entry = self._ensure_frame(frame_id)

        # vlm_called is True whenever vlm_triggered has been set (even if False).
        vlm_called = (vlm_triggered is not None)

        alert_record = {
            "rule_index": rule_index,
            "condition": condition,
            "OD_logic_triggered": od_logic_triggered,
            "vlm_called": vlm_called,
            # VLM binary answer; None means the VLM was not invoked this frame.
            "vlm_triggered": vlm_triggered,
            "vlm_raw_response": vlm_raw_response,
            "vlm_latency_sec": round(vlm_latency_sec, 4) if vlm_latency_sec is not None else None,
            "vlm_yes_duration_sec": round(duration_met_sec, 4) if duration_met_sec is not None else None,
            "vlm_no_duration_sec": round(negative_duration_sec, 4) if negative_duration_sec is not None else None,
            "alert_state": alert_state,
            "parser_status": parser_status,
            "parser_failure": bool(parser_failure),
            "binary_evidence": binary_evidence,
            "api_success": api_success,
            "api_error": api_error,
            "grounding": grounding or {},
            "details": details,
            "video_time_sec": entry["video_time_sec"],
            "duration_met_sec": round(duration_met_sec, 4) if duration_met_sec is not None else None,
        }
        entry["alerts"].append(alert_record)

        # Update the frame-level VLM summary for quick access.
        if vlm_called:
            entry["frame_summary"]["vlm_called"] = True
            entry["frame_summary"]["vlm_results"].append({
                "rule_index": rule_index,
                "condition": condition,
                "vlm_answer": "Yes" if vlm_triggered else "No",
                "vlm_latency_sec": round(vlm_latency_sec, 4) if vlm_latency_sec is not None else None,
                "vlm_yes_duration_sec": round(duration_met_sec, 4) if duration_met_sec is not None else None,
            })
        self._dirty = True

    def update_frame_processing_time(self, frame_id: int, processing_time_sec: float):
        entry = self._ensure_frame(frame_id)
        entry["processing_time_sec"] = round(processing_time_sec, 4)
        self._dirty = True
        # Flush to disk after every frame so the log is recoverable on crash.
        self._save()

    # ================================================================
    # Read API
    # ================================================================
    def get_tracker_history(self, tracker_id: int, last_n_frames: Optional[int] = None) -> List[dict]:
        """ Returns detection records for a tracker, ordered from most recent to oldest. """
        results = []
        for fid in reversed(self._frames):
            frame = self._frames[fid]
            for det in frame["detections"]:
                if det["tracker_id"] == tracker_id:
                    results.append({
                        "frame_id": frame["frame_id"],
                        "tracker_id": det["tracker_id"],
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "x1": det["bbox"][0],
                        "y1": det["bbox"][1],
                        "x2": det["bbox"][2],
                        "y2": det["bbox"][3],
                        "video_time_sec": frame["video_time_sec"],
                    })
            if last_n_frames and len(results) >= last_n_frames:
                results = results[:last_n_frames]
                break
        return results

    # ================================================================
    # Export
    # ================================================================
    def get_summary(self) -> dict:
        """ Returns aggregate counts across the entire log. """
        all_alerts = [a for f in self._frames.values() for a in f["alerts"]]
        tracker_ids = {
            d["tracker_id"]
            for f in self._frames.values()
            for d in f["detections"]
        }
        return {
            "total_frames": len(self._frames),
            "total_images": sum(len(f["images"]) for f in self._frames.values()),
            "total_detections": sum(len(f["detections"]) for f in self._frames.values()),
            "total_poses": sum(len(f["poses"]) for f in self._frames.values()),
            "total_depths": sum(len(f["depths"]) for f in self._frames.values()),
            "total_alerts": len(all_alerts),
            "OD_logic_triggered_count": sum(1 for a in all_alerts if a.get("OD_logic_triggered", a.get("logic_triggered", False))),
            # Frames where the VLM was actually invoked.
            "vlm_called_count": sum(1 for a in all_alerts if a.get("vlm_called", a["vlm_triggered"] is not None)),
            # Frames where the VLM answered "Yes".
            "vlm_yes_count": sum(1 for a in all_alerts if a["vlm_triggered"] is True),
            "unique_trackers": len(tracker_ids),
            "vlm_triggered_count": sum(1 for a in all_alerts if a["vlm_triggered"]),
        }

    # ================================================================
    # Lifecycle
    # ================================================================
    def close(self):
        self._dirty = True
        self._save()
        print(f"[StructureLog] JSON closed: {self.db_path}")

    def __del__(self):
        try:
            self._dirty = True
            self._save()
        except Exception:
            pass
