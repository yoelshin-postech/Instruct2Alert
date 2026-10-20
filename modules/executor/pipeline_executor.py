# modules/executor/pipeline_executor.py
"""
Frame-by-frame pipeline executor for the safety monitoring system.

Runs the full pipeline on each sampled video frame in sequence:
  1. 2-Step Object Detection: PersonDetector (YOLO) + ObjectDetector (OmDet)
  2. Kalman tracking for the main target
  3. Optional Pose estimation (YOLO-Pose)

Per-alert-rule processing (one iteration per rule per frame):
  4. Metadata gathering: OD conditions, depth/pose context
  5. VLM classification: sends frame + CV context to EventClassifier
  6. Temporal alert gating: ON after yes_streak >= duration_sec, OFF after no_streak >= 1s

Outputs:
  - Structured JSON log (StructureLog)
  - Per-frame images: raw / detection / pose / depth / vlm / roi
  - VLM call records (CSV)
  - Alert intervals (start/end timestamps)

Usage:
    executor = PipelineExecutor(workflow_json, pipeline_df, video_path, provider)
    result = await executor.run()
    executor.close()
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import os
import cv2
from datetime import datetime
from .frame_processor import FrameProcessor
from .structure_log import StructureLog
from .event_classifier import EventClassifier
from modules.tools import (
    PersonDetector, ObjectDetector, PoseEstimator, DepthEstimator,
)
from modules.run_audit import append_jsonl, json_default, summarize_latencies, write_csv, write_json


class PipelineExecutor:
    def __init__(
        self,
        workflow_json: dict,
        pipeline_df: pd.DataFrame,
        video_path: str,
        provider,
        db_path: str = ":memory:",
        device: str = "cuda:1",
        realtime_skip: bool = True,
        target_fps: float = 10.0,
        scenario_id: str = None,
        video_id: str = None,
        image_output_dir: str = None,
        audit_dir: str = None,
        run_seed: Optional[int] = None,
        variant: str = "full",
        long_gap_threshold_sec: float = 2.0,
        gt_intervals: Optional[List[List[int]]] = None,
    ):
        self.workflow = workflow_json
        self.pipeline_df = pipeline_df
        self.video_path = video_path
        self.device = device
        self.run_seed = run_seed
        self.variant = variant
        self.video_id = video_id
        self.long_gap_threshold_sec = float(long_gap_threshold_sec)
        self.gt_intervals = gt_intervals or []

        # Parse workflow targets and module flags.
        targets = workflow_json.get("targets", {})
        self.main_target = targets.get("main", "person")
        self.other_targets = targets.get("other_targets", [])
        self.all_targets = [self.main_target] + self.other_targets
        self.additional_modules = workflow_json.get("additional_modules", [])
        self.alert_rules = workflow_json.get("alert_rules", [])

        self.use_pose = "Pose" in self.additional_modules
        self.use_depth = "Depth" in self.additional_modules

        # Determine which detector types are needed.
        self.has_person_target = (
            self.main_target.lower() == "person" or
            any(t.lower() == "person" for t in self.other_targets)
        )
        # Non-person targets are routed to OmDet (open-vocabulary detector).
        self.non_person_targets = [
            t for t in self.all_targets if t.lower() != "person"
        ]

        # Initialize all pipeline components.
        print(f"[Executor] Initializing modules (device={device})...")
        self.frame_proc = FrameProcessor(
            video_path, realtime_skip=realtime_skip, target_fps=target_fps)
        self.log = StructureLog(db_path)

        # 2-Step OD: PersonDetector(YOLO) + ObjectDetector(OmDet)
        if self.has_person_target:
            self.person_detector = PersonDetector(
                device=device, structure_log=self.log)
        else:
            self.person_detector = None

        if self.non_person_targets or not self.has_person_target:
            self.object_detector = ObjectDetector(
                device=device, structure_log=self.log)
        else:
            self.object_detector = None

        if self.use_pose:
            self.pose_estimator = PoseEstimator(device=device)
        else:
            self.pose_estimator = None

        if self.use_depth:
            self.depth_estimator = DepthEstimator(device=device)
        else:
            self.depth_estimator = None

        self.classifier = EventClassifier(provider)

        # Create per-scenario output directories for frame images.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if scenario_id:
            folder_name = f"{scenario_id}_{timestamp}"
        else:
            folder_name = timestamp
        image_output_dir = image_output_dir or os.path.join("data", "images_result")
        self.output_root = os.path.join(image_output_dir, folder_name)
        os.makedirs(self.output_root, exist_ok=True)
        for sub_dir in ["raw", "detection", "pose", "depth", "vlm", "roi"]:
            os.makedirs(os.path.join(self.output_root, sub_dir), exist_ok=True)

        # Per-module timing accumulators for performance reporting.
        self.step_times: Dict[str, List[float]] = {}
        self.total_vlm_calls = 0
        self.total_vlm_yes_count = 0
        self.total_alerts_triggered = 0

        # Flat list of VLM call records exported to CSV by the caller.
        self.vlm_call_records: List[Dict[str, Any]] = []
        self.scenario_id = scenario_id
        self.audit_dir = audit_dir
        if self.audit_dir:
            os.makedirs(self.audit_dir, exist_ok=True)
        self.frame_audit_records: List[Dict[str, Any]] = []
        self.grounding_events: List[Dict[str, Any]] = []
        self.tracking_events: List[Dict[str, Any]] = []
        self.latency_events: List[Dict[str, Any]] = []
        self._current_frame_id: Optional[int] = None
        self._current_video_time: Optional[float] = None
        self._current_rule_index: Optional[int] = None

        # Previous-frame context passed to the classifier for temporal reference.
        self._prev_frame_context: Optional[Dict[str, Any]] = None

        # Per-rule alert state (True = alert currently ON).
        self._alert_active: Dict[int, bool] = {}

        # Consecutive Yes/No streak start times for temporal gating.
        self._yes_streak_start: Dict[int, Optional[float]] = {}  # rule_idx → start time of current Yes streak
        self._no_streak_start: Dict[int, Optional[float]] = {}   # rule_idx → start time of current No streak

        # Alert interval tracking for output reporting.
        self._alert_intervals: Dict[int, List[List[float]]] = {}
        self._alert_start_time: Dict[int, float] = {}

        det_info = []
        if self.person_detector:
            det_info.append("Person(YOLO)")
        if self.object_detector:
            det_info.append(f"Object(OmDet): {self.non_person_targets if self.has_person_target else self.all_targets}")
        print(f"[Executor] Ready. targets={self.all_targets} | "
              f"Detectors={', '.join(det_info)} | "
              f"Pose={self.use_pose} | Depth={self.use_depth} | "
              f"rules={len(self.alert_rules)} | "
              f"seed={self.run_seed} | long_gap={self.long_gap_threshold_sec:.1f}s")

    # ================================================================
    # Main Run
    # ================================================================
    async def run(self) -> Dict[str, Any]:
        print(f"\n[Executor] Starting pipeline on {self.video_path}")
        run_started_at = time.perf_counter()

        for frame_data in self.frame_proc:
            t0_frame = time.time()

            frame = frame_data["frame"]
            frame_id = frame_data["frame_id"]
            video_time = frame_data["video_time_sec"]
            delta_t_sec = float(frame_data.get("delta_t_sec", 0.0))
            long_gap_detected = delta_t_sec > self.long_gap_threshold_sec
            long_gap_action = "reset_streaks" if long_gap_detected else "none"
            self._current_frame_id = frame_id
            self._current_video_time = video_time

            # Register the frame in the structured log before processing.
            self.log.log_frame(
                frame_id, video_time, delta_t_sec=delta_t_sec,
                long_gap_detected=long_gap_detected,
                long_gap_action=long_gap_action,
            )

            # Run the full per-frame pipeline; log a placeholder alert on exception.
            try:
                await self._process_frame(
                    frame, frame_id, video_time, delta_t_sec,
                    long_gap_detected, long_gap_action,
                )
            except Exception as e:
                print(f"\n  [Executor] ERROR at frame {frame_id}: {e}")
                # Log a failed alert entry so the frame is not silently skipped.
                for rule_idx, rule in enumerate(self.alert_rules):
                    self.log.log_alert_evaluation(
                        frame_id=frame_id,
                        rule_index=rule_idx,
                        condition=rule.get("condition", ""),
                        od_logic_triggered=False,
                        details=f"pipeline_exception: {e}",
                    )

            # Record per-frame wall-clock time and update progress display.
            elapsed = time.time() - t0_frame
            self._record_step_time("FrameExecutor", elapsed, scope="frame")
            self.log.update_frame_processing_time(frame_id, elapsed)
            self.frame_proc.report_processing_time(elapsed)
            self.frame_proc.print_progress(extra=f"proc={elapsed:.3f}s")

        self.frame_proc.release()

        # Close any alerts that were still ON at the end of the video.
        last_video_time = 0.0
        if self.log._frames:
            last_fid = list(self.log._frames.keys())[-1]
            last_video_time = self.log._frames[last_fid]["video_time_sec"]
        for rule_idx, active in self._alert_active.items():
            if active:
                start = self._alert_start_time.get(rule_idx, 0.0)
                if rule_idx not in self._alert_intervals:
                    self._alert_intervals[rule_idx] = []
                self._alert_intervals[rule_idx].append(
                    [round(start, 1), round(last_video_time, 1)])

        self._record_step_time(
            "ScenarioExecutor", time.perf_counter() - run_started_at,
            scope="scenario",
        )

        result = {
            "video_info": self.frame_proc.get_video_info(),
            "run_stats": self.frame_proc.get_run_stats(),
            "db_summary": self.log.get_summary(),
            "step_avg_times": self._compute_avg_step_times(),
            "step_latency_summary": self._compute_step_latency_summary(),
            "total_vlm_calls": self.total_vlm_calls,
            "total_vlm_yes_count": self.total_vlm_yes_count,
            "total_alerts_triggered": self.total_alerts_triggered,
            "alert_intervals": self._get_combined_alert_intervals(),
            "vlm_call_records": self.vlm_call_records,
            "frame_audit_records": self.frame_audit_records,
            "grounding_events": self.grounding_events,
            "tracking_events": self.tracking_events,
            "latency_events": self.latency_events,
            "frame_schedule": self.frame_proc.get_schedule_records(),
            "audit_summary": self._compute_audit_summary(),
        }
        self._write_audit_artifacts(result)

        print(f"\n[Executor] Pipeline complete.")
        print(f"[Executor] Summary: {json.dumps(result['db_summary'], indent=2)}")
        return result

    # ================================================================
    def _is_in_gt_interval(self, frame_index: int) -> bool:
        """Return the scenario GT state for a canonical 1-based frame index."""
        return any(
            len(interval) >= 2 and int(interval[0]) <= frame_index <= int(interval[1])
            for interval in self.gt_intervals
        )

    def _compute_audit_summary(self) -> Dict[str, Any]:
        """Aggregate gap and grounding evidence without discarding raw events."""
        fallback_events = [
            row for row in self.grounding_events if row.get("full_frame_fallback")
        ]
        grounded_events = [
            row for row in self.grounding_events if row.get("grounding_success")
        ]
        return {
            "long_gap_reset_count": sum(
                1 for row in self.frame_audit_records if row.get("long_gap_detected")
            ),
            "fallback_call_count": len(fallback_events),
            "grounded_call_count": len(grounded_events),
            "verifier_call_count": len(self.grounding_events),
        }

    # ================================================================
    # Per-Frame Processing
    # ================================================================
    async def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        video_time: float,
        delta_t_sec: float,
        long_gap_detected: bool,
        long_gap_action: str,
    ):
        """ Runs all pipeline steps for a single frame and logs results. """

        # Shared context dict passed between pipeline steps.
        ctx: Dict[str, Any] = {
            "od_result": None,
            "person_trackers": [],
            "pose_results": {},        # {tracker_id: pose_dict}
            "depth_result": None,
            "logic_triggers": {},
        }

        # Save the raw input frame.
        raw_img_path = os.path.join(self.output_root, "raw", f"{frame_id:06d}.jpg")
        cv2.imwrite(raw_img_path, frame)
        self.log.log_frame_image(frame_id, "raw", raw_img_path)

        # ── Step 1: 2-Step Object Detection ──
        # Run PersonDetector and ObjectDetector separately, then merge their results.
        person_od = {"detections": [], "trackers": []}
        object_od = {"detections": [], "trackers": []}

        if self.has_person_target and self.person_detector:
            t0 = time.time()
            person_od = self.person_detector.detect(frame)
            self._record_step_time("OD_Person", time.time() - t0)

        if self.has_person_target:
            # Person is the main target; run OmDet only for non-person objects.
            if self.object_detector and self.non_person_targets:
                t0 = time.time()
                object_od = self.object_detector.detect(
                    frame, self.non_person_targets)
                self._record_step_time("OD_Object", time.time() - t0)
        else:
            # No person target; use OmDet for all targets.
            if self.object_detector:
                t0 = time.time()
                object_od = self.object_detector.detect(
                    frame, self.all_targets)
                self._record_step_time("OD", time.time() - t0)

        # Merge person and object detections into a single result.
        combined_od = {
            "detections": person_od["detections"] + object_od["detections"],
            "trackers": person_od["trackers"] + object_od["trackers"],
            "tracker_events": person_od.get("tracker_events", []) + object_od.get("tracker_events", []),
        }
        ctx["od_result"] = combined_od

        # Save combined detection visualization.
        det_img_path = self._save_combined_detection(
            frame, combined_od, frame_id)
        self.log.log_frame_image(frame_id, "detection", det_img_path)

        # Draw detections onto a frame copy used for downstream visualization steps.
        det_frame = self._draw_detections(frame, combined_od)

        # Log only active (age==0) trackers to the structured log.
        for trk in combined_od["trackers"]:
            if trk["age"] == 0:
                score = 0.0
                for det in combined_od["detections"]:
                    if det["label"] == trk["label"]:
                        score = det["score"]
                        break
                self.log.log_detection(
                    frame_id=frame_id,
                    tracker_id=trk["tracker_id"],
                    label=trk["label"],
                    confidence=score,
                    bbox=trk["box"],
                )

        # Find trackers for the main target (used by pose, depth, and alert steps).
        t0 = time.time()
        main_target_trackers = self._find_main_target_trackers()
        self._record_step_time("Tracker", time.time() - t0)
        ctx["main_target_trackers"] = main_target_trackers
        ctx["person_trackers"] = main_target_trackers if self.main_target.upper() == "PERSON" else []

        main_label = self.main_target.upper()
        active_main_track_ids = [
            tracker["tracker_id"] for tracker in combined_od["trackers"]
            if tracker["label"] == main_label and tracker["age"] == 0
        ]
        coasting_main_track_ids = [
            tracker["tracker_id"] for tracker in combined_od["trackers"]
            if tracker["label"] == main_label and 0 < tracker["age"] <= 5
        ]
        self.tracking_events.append({
            "run_seed": self.run_seed,
            "variant": self.variant,
            "scenario_id": self.scenario_id,
            "video_id": self.video_id,
            "frame_index": frame_id,
            "video_timestamp_sec": video_time,
            "person_detection_count": len(person_od["detections"]),
            "active_track_ids": active_main_track_ids,
            "coasting_track_ids": coasting_main_track_ids,
            "tracker_lifecycle_events": combined_od.get("tracker_events", []),
        })

        # True when at least one main-target tracker is active or coasting.
        main_target_detected = len(main_target_trackers) > 0

        # ── Step 3 (optional): Pose Estimation ──
        if self.use_pose and self.pose_estimator is not None and self.main_target.upper() == "PERSON":
            t0 = time.time()
            try:
                for trk in main_target_trackers:
                    try:
                        pose_result = self.pose_estimator.estimate_with_tracker(frame, trk)
                        if pose_result:
                            ctx["pose_results"][trk.id] = pose_result
                            self.pose_estimator.save_results(
                                det_frame, pose_result, frame_id,
                                os.path.join(self.output_root, "pose"))
                            pose_img_path = os.path.join(
                                self.output_root, "pose", f"{frame_id:06d}.jpg")
                            self.log.log_frame_image(frame_id, "pose", pose_img_path)
                            self.log.log_pose(
                                frame_id=frame_id,
                                tracker_id=trk.id,
                                posture=pose_result.get("posture", "unknown"),
                                body_angle=pose_result.get("body_angle", 0.0),
                                confidence=pose_result.get("confidence", 0.0),
                                keypoints=pose_result.get("keypoints", {}),
                            )
                        else:
                            self.log.log_pose(
                                frame_id=frame_id,
                                tracker_id=trk.id,
                                posture=None, body_angle=None,
                                confidence=None, keypoints=None,
                            )
                    except Exception as e:
                        print(f"\n  [Pose] WARNING: tracker {trk.id} failed: {e}")
                        self.log.log_pose(
                            frame_id=frame_id,
                            tracker_id=trk.id,
                            posture=None, body_angle=None,
                            confidence=None, keypoints=None,
                        )
            except Exception as e:
                print(f"\n  [Pose] WARNING: Pose estimation skipped: {e}")
            self._record_step_time("Pose", time.time() - t0)

        # ── Depth Estimation (optional) ──
        if self.use_depth and self.depth_estimator is not None:
            t0 = time.time()
            try:
                detected_objects = self._build_depth_objects(combined_od)
                depth_result = self.depth_estimator.analyze(frame, detected_objects)
                ctx["depth_result"] = depth_result
                self.depth_estimator.save_results(
                    det_frame, depth_result, frame_id,
                    os.path.join(self.output_root, "depth"))
                depth_img_path = os.path.join(
                    self.output_root, "depth", f"{frame_id:06d}.jpg")
                self.log.log_frame_image(frame_id, "depth", depth_img_path)

                for od in depth_result.get("object_depths", []):
                    di = od.get("depth_info", {})
                    self.log.log_depth(
                        frame_id=frame_id,
                        tracker_id=od.get("id", 0),
                        label=od.get("type", "unknown"),
                        mean_depth=di.get("mean_depth"),
                        median_depth=di.get("median_depth"),
                        min_depth=di.get("min_depth"),
                        max_depth=di.get("max_depth"),
                    )
            except Exception as e:
                print(f"\n  [Depth] WARNING: Depth estimation skipped: {e}")
            self._record_step_time("Depth", time.time() - t0)

        # ── Steps 4 & 5: Alert Rule Evaluation ──
        # For each rule: gather metadata, run VLM, apply temporal gating.
        frame_vlm_results = []

        # Extract ROI crops for the main target (sent to the VLM alongside the full frame).
        roi_crops = None
        roi_track_ids = []
        if main_target_detected:
            roi_crops, roi_track_ids = self._extract_roi_crops(frame, ctx.get("od_result", {}))
            # Save the first ROI crop for visualization.
            if roi_crops:
                roi_dir = os.path.join(self.output_root, "roi")
                for crop_idx, crop in enumerate(roi_crops):
                    roi_img_path = os.path.join(
                        roi_dir, f"{frame_id:06d}_roi{crop_idx}.jpg")
                    cv2.imwrite(roi_img_path, crop)
                self.log.log_frame_image(frame_id, "roi", os.path.join(
                    roi_dir, f"{frame_id:06d}_roi0.jpg"))

        grounding_attempted = bool(main_target_detected)
        grounding_success = bool(roi_crops)
        coasting_only = bool(main_target_detected and not active_main_track_ids)
        ground_truth_state = "Yes" if self._is_in_gt_interval(frame_id) else "No"
        if grounding_success:
            fallback_reason = None
        elif not main_target_detected:
            fallback_reason = "no_main_target_track"
        elif coasting_only:
            fallback_reason = "coasting_track_not_roi_eligible"
        else:
            fallback_reason = "empty_or_invalid_roi"

        for rule_idx, rule in enumerate(self.alert_rules):
            self._current_rule_index = rule_idx
            # Step 4: Gather OD/depth/pose metadata for this rule.
            t0 = time.time()
            rule_metadata = self._gather_rule_metadata(frame_id, rule, ctx)
            self._record_step_time("Metadata", time.time() - t0)

            vlm_triggered = None
            vlm_raw = None
            vlm_latency = None
            vlm_yes_duration = None

            # Step 5: VLM classification.
            # VLM is always called regardless of OD result (unconditional inference mode).
            t0_vlm = time.time()
            tool_context = self._build_tool_context(ctx, video_time)

            vp = rule.get("verification_prompt", {})
            if not vp.get("query"):
                vp = {
                    "system": "You are a visual safety event verification model.",
                    "instruction": "Based ONLY on what you see in this single frame, "
                                   "answer with 'Yes' or 'No'. Do not explain.",
                    "query": rule.get("condition", "Is there a safety event?"),
                }

            # Include OD supplement text only when the main target was detected.
            od_hit_supplement = None
            if main_target_detected:
                od_hit_supplement = rule.get("od_hit_supplement")

            grounding_record = {
                "run_seed": self.run_seed,
                "variant": self.variant,
                "scenario_id": self.scenario_id,
                "video_id": self.video_id,
                "frame_index": frame_id,
                "rule_index": rule_idx,
                "video_timestamp_sec": video_time,
                "grounding_attempted": grounding_attempted,
                "grounding_success": grounding_success,
                "active_main_target_detected": bool(active_main_track_ids),
                "active_track_ids": active_main_track_ids,
                "coasting_only": coasting_only,
                "coasting_track_ids": coasting_main_track_ids,
                "roi_requested": bool(main_target_detected),
                "roi_count": len(roi_crops or []),
                "roi_track_ids": roi_track_ids,
                "grounding_hint_used": bool(od_hit_supplement and roi_crops),
                "full_frame_fallback": not grounding_success,
                "fallback_reason": fallback_reason,
                "ground_truth_state": ground_truth_state,
            }

            vlm_result = await self.classifier.classify(
                frame=frame,
                verification_prompt=vp,
                tool_context=tool_context,
                roi_crops=roi_crops,
                od_hit_supplement=od_hit_supplement,
            )
            vlm_latency = time.time() - t0_vlm
            self._record_step_time("VLM", vlm_latency)
            self.total_vlm_calls += 1

            vlm_triggered = (vlm_result["answer"] == "Yes")
            vlm_raw = vlm_result["raw_response"]
            grounding_record.update({
                "verifier_answer": vlm_result["answer"],
                "parser_status": vlm_result["parser_status"],
                "binary_evidence": vlm_result["binary_evidence"],
                "parser_failure": vlm_result["parser_failure"],
                "api_success": vlm_result["api_success"],
                "api_error": vlm_result["api_error"],
            })
            self.grounding_events.append(grounding_record)

            if vlm_triggered:
                self.total_vlm_yes_count += 1

            frame_vlm_results.append({
                "rule_index": rule_idx,
                "condition": rule.get("condition", ""),
                "answer": vlm_result["answer"],
            })

            # Save the annotated frame used as VLM input.
            vlm_img_path = os.path.join(
                self.output_root, "vlm", f"{frame_id:06d}.jpg")
            cv2.imwrite(vlm_img_path, det_frame)
            self.log.log_frame_image(frame_id, "vlm", vlm_img_path)

            # Save the full VLM context and answer to a JSON sidecar file.
            vlm_json_path = os.path.join(
                self.output_root, "vlm", f"{frame_id:06d}_r{rule_idx}.json")
            with open(vlm_json_path, 'w') as f:
                json.dump({
                    "vlm_answer": vlm_result["answer"],
                    "vlm_raw": vlm_raw,
                    "vlm_latency_sec": round(vlm_latency, 4),
                    "context": tool_context,
                    "od_detected": main_target_detected,
                    "input_mode": "full+roi" if roi_crops else "full_frame",
                    "od_hit_supplement": od_hit_supplement,
                    "rule_metadata": rule_metadata,
                    "run_seed": self.run_seed,
                    "variant": self.variant,
                    "delta_t_sec": delta_t_sec,
                    "long_gap_detected": long_gap_detected,
                    "long_gap_action": long_gap_action,
                    "grounding": grounding_record,
                    "parser_status": vlm_result["parser_status"],
                    "parser_failure": vlm_result["parser_failure"],
                    "binary_evidence": vlm_result["binary_evidence"],
                    "api_success": vlm_result["api_success"],
                    "api_error": vlm_result["api_error"],
                }, f, indent=2, ensure_ascii=False, default=json_default)

            input_mode = "full+roi" if roi_crops else "full_frame"
            print(f"\n  [VLM] frame={frame_id} rule={rule.get('condition','')} "
                  f"answer={vlm_result['answer']} mode={input_mode} ({vlm_latency:.2f}s)")

            # Append to the flat VLM record list for CSV export.
            self.vlm_call_records.append({
                "Scenario_ID": self.scenario_id,
                "Run_Seed": self.run_seed,
                "Variant": self.variant,
                "Frame_Index": frame_id,
                "Video_Time_sec": round(video_time, 6),
                "Delta_t_sec": round(delta_t_sec, 6),
                "Long_Gap_Detected": long_gap_detected,
                "Long_Gap_Action": long_gap_action,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "Rule_Index": rule_idx,
                "Condition": rule.get("condition", ""),
                "Prompt": vp.get("query", ""),
                "VLM_Latency_sec": round(vlm_latency, 4) if vlm_latency else None,
                "Raw_Response": vlm_raw,
                "Parser_Status": vlm_result["parser_status"],
                "Parser_Failure": vlm_result["parser_failure"],
                "Binary_Evidence": vlm_result["binary_evidence"],
                "Inference_Result": "Yes" if vlm_triggered else "No",
                "API_Success": vlm_result["api_success"],
                "API_Error": vlm_result["api_error"],
                "Request_Seed": vlm_result["request_seed"],
                "OD_Detected": main_target_detected,
                "Input_Mode": "full+roi" if roi_crops else "full_frame",
                "OD_Hit_Supplement": od_hit_supplement,
                "Grounding_Attempted": grounding_attempted,
                "Grounding_Success": grounding_success,
                "Active_Main_Target_Detected": bool(active_main_track_ids),
                "Coasting_Only": coasting_only,
                "ROI_Requested": bool(main_target_detected),
                "ROI_Count": len(roi_crops or []),
                "ROI_Track_IDs": json.dumps(roi_track_ids),
                "Grounding_Hint_Used": bool(od_hit_supplement and roi_crops),
                "Full_Frame_Fallback": not grounding_success,
                "Fallback_Reason": fallback_reason,
                "Ground_Truth_State": ground_truth_state,
            })

            # ── Temporal Alert Gating ──
            condition_str = rule.get("condition", "")
            params = rule.get("parameters", {})
            duration_sec = params.get("duration_sec", 0.0)
            prev_state = self._alert_active.get(rule_idx, False)

            # A long unobserved gap never counts as positive or negative continuity.
            if long_gap_detected:
                self._yes_streak_start.pop(rule_idx, None)
                self._no_streak_start.pop(rule_idx, None)

            # Update Yes/No streak timers.
            if vlm_triggered:
                self._no_streak_start.pop(rule_idx, None)
                if rule_idx not in self._yes_streak_start:
                    self._yes_streak_start[rule_idx] = video_time
                yes_streak = video_time - self._yes_streak_start[rule_idx]
                no_streak = 0.0
            else:
                self._yes_streak_start.pop(rule_idx, None)
                if rule_idx not in self._no_streak_start:
                    self._no_streak_start[rule_idx] = video_time
                no_streak = video_time - self._no_streak_start[rule_idx]
                yes_streak = 0.0

            # Determine the new alert state based on the gating logic.
            if not duration_sec or duration_sec <= 0:
                # No duration gate: alert mirrors the raw VLM answer.
                new_state = vlm_triggered
            else:
                # Hysteresis gate: ON when Yes streak ≥ duration_sec,
                # OFF when No streak ≥ off_duration (fixed at 1s).
                off_duration = 1.0
                if vlm_triggered and yes_streak >= duration_sec:
                    new_state = True
                elif not vlm_triggered and prev_state and no_streak >= off_duration:
                    new_state = False
                else:
                    new_state = prev_state

            # Duration to report in the log (only meaningful when gating is active).
            vlm_yes_duration = yes_streak if (duration_sec and duration_sec > 0) else None

            # Detect alert state transitions and record interval boundaries.
            if new_state and not prev_state:
                # OFF → ON
                self._alert_active[rule_idx] = True
                self._alert_start_time[rule_idx] = video_time
                self.total_alerts_triggered += 1
                print(f"\n  [ALERT] rule[{rule_idx}] '{condition_str}' TRIGGERED | "
                      f"yes_streak={yes_streak:.2f}s")
            elif not new_state and prev_state:
                # ON → OFF
                self._alert_active[rule_idx] = False
                start = self._alert_start_time.get(rule_idx, video_time)
                if rule_idx not in self._alert_intervals:
                    self._alert_intervals[rule_idx] = []
                self._alert_intervals[rule_idx].append(
                    [round(start, 1), round(video_time, 1)])
                print(f"\n  [ALERT] rule[{rule_idx}] '{condition_str}' CLEARED "
                      f"(interval: {start:.1f}s ~ {video_time:.1f}s, no_streak={no_streak:.2f}s)")
            else:
                self._alert_active[rule_idx] = new_state

            # Write the final alert state back to the VLM record for CSV export.
            alert_on = self._alert_active.get(rule_idx, False)
            self.vlm_call_records[-1]["Alert_Result"] = "Yes" if alert_on else "No"
            self.vlm_call_records[-1]["Yes_Streak_sec"] = round(yes_streak, 3)
            self.vlm_call_records[-1]["No_Streak_sec"] = round(no_streak, 3)

            # Build the diagnostic details string and flush to the structured log.
            details_parts = [
                f"od_detected={main_target_detected}",
                f"input_mode={'full+roi' if roi_crops else 'full_frame'}",
                f"yes_streak={yes_streak:.3f}s",
                f"no_streak={no_streak:.3f}s",
                f"S_t={'ON' if alert_on else 'OFF'}",
            ]
            if duration_sec and duration_sec > 0:
                details_parts.append(f"duration_sec={duration_sec}, off_duration=1.0")
            details_parts.append(f"metadata={rule_metadata.get('metadata', '')}")
            self.log.log_alert_evaluation(
                frame_id=frame_id,
                rule_index=rule_idx,
                condition=rule.get("condition", ""),
                od_logic_triggered=main_target_detected,
                vlm_triggered=vlm_triggered,
                vlm_raw_response=vlm_raw,
                vlm_latency_sec=vlm_latency,
                duration_met_sec=vlm_yes_duration,
                negative_duration_sec=no_streak if (duration_sec and duration_sec > 0) else None,
                alert_state=alert_on,
                parser_status=vlm_result["parser_status"],
                parser_failure=vlm_result["parser_failure"],
                binary_evidence=vlm_result["binary_evidence"],
                api_success=vlm_result["api_success"],
                api_error=vlm_result["api_error"],
                grounding=grounding_record,
                details=", ".join(details_parts),
            )

            self.frame_audit_records.append({
                "run_seed": self.run_seed,
                "variant": self.variant,
                "evidence_source": "grounded_verify",
                "scenario_id": self.scenario_id,
                "video_id": self.video_id,
                "frame_index": frame_id,
                "video_timestamp_sec": video_time,
                "delta_t_sec": delta_t_sec,
                "long_gap_detected": long_gap_detected,
                "long_gap_action": long_gap_action,
                "rule_index": rule_idx,
                "raw_response": vlm_raw,
                "parser_status": vlm_result["parser_status"],
                "parser_failure": vlm_result["parser_failure"],
                "binary_evidence": vlm_result["binary_evidence"],
                "api_success": vlm_result["api_success"],
                "api_error": vlm_result["api_error"],
                "positive_streak_sec": round(yes_streak, 6),
                "negative_streak_sec": round(no_streak, 6),
                "final_alert_state": alert_on,
                "ground_truth_state": ground_truth_state,
                "grounding_attempted": grounding_attempted,
                "grounding_success": grounding_success,
                "full_frame_fallback": not grounding_success,
                "fallback_reason": fallback_reason,
                "grounding": grounding_record,
            })

        # Store a summary of this frame's results for the next frame's context.
        self._prev_frame_context = self._build_prev_frame_context(
            video_time, ctx, frame_vlm_results)

    # ================================================================
    # Helpers
    # ================================================================
    def _find_main_target_trackers(self) -> List:
        """ Returns active trackers for the main target, with a short coasting fallback.
        Prefers age==0 (matched this frame), but accepts age<=5 to handle brief occlusions.

        Routing logic:
          - PERSON target   → PersonDetector first, then ObjectDetector as fallback.
          - Non-person target → ObjectDetector first, then PersonDetector as fallback.
        """
        main_label = self.main_target.upper()
        is_person_target = (main_label == "PERSON")

        # Person target: check PersonDetector; fall back to coasting trackers.
        if is_person_target and self.person_detector:
            active = [trk for trk in self.person_detector.get_active_trackers()
                      if trk.label_name == main_label]
            if active:
                return active
            return [trk for trk in self.person_detector.trackers
                    if trk.label_name == main_label and trk.age <= 5]

        # Non-person target: check ObjectDetector first.
        if self.object_detector:
            active = [trk for trk in self.object_detector.get_active_trackers()
                      if trk.label_name == main_label]
            if active:
                return active
            return [trk for trk in self.object_detector.trackers
                    if trk.label_name == main_label and trk.age <= 5]

        # Fallback: try PersonDetector for non-person labels (should not occur normally).
        if self.person_detector:
            active = [trk for trk in self.person_detector.get_active_trackers()
                      if trk.label_name == main_label]
            if active:
                return active
            return [trk for trk in self.person_detector.trackers
                    if trk.label_name == main_label and trk.age <= 5]

        return []

    def _extract_roi_crops(
        self,
        frame: np.ndarray,
        od_result: dict,
        margin_ratio: float = 0.15,
    ) -> tuple[List[np.ndarray], List[int]]:
        """ Crops tight regions around each active main-target tracker with a padding margin.

        Args:
            frame:        Full BGR video frame.
            od_result:    Combined OD result dict with "trackers" list.
            margin_ratio: Fraction of bbox dimensions added as padding on each side.

        Returns:
            BGR crop arrays and the corresponding active tracker IDs.
        """
        crops = []
        tracker_ids = []
        h, w = frame.shape[:2]
        main_label = self.main_target.upper()

        for trk in od_result.get("trackers", []):
            if trk["age"] != 0:
                continue
            if trk["label"] != main_label:
                continue

            box = trk["box"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Expand the crop by margin_ratio on each side, clamped to the frame boundary.
            bw, bh = x2 - x1, y2 - y1
            margin_w = int(bw * margin_ratio)
            margin_h = int(bh * margin_ratio)
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(w, x2 + margin_w)
            y2 = min(h, y2 + margin_h)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                tracker_ids.append(trk["tracker_id"])

        return crops, tracker_ids

    def _get_all_active_trackers(self) -> List:
        """ Returns all age==0 trackers from both PersonDetector and ObjectDetector. """
        trackers = []
        if self.person_detector:
            trackers.extend(self.person_detector.get_active_trackers())
        if self.object_detector:
            trackers.extend(self.object_detector.get_active_trackers())
        return trackers

    def _build_depth_objects(self, od_result: dict) -> list:
        """ Converts active tracker states into the format expected by DepthEstimator.analyze(). """
        objects = []
        for trk in od_result["trackers"]:
            if trk["age"] == 0:
                objects.append({
                    "id": trk["tracker_id"],
                    "type": trk["label"],
                    "bbox": trk["box"],
                    "confidence": 1.0,
                })
        return objects

    def _draw_detections(
        self,
        frame: np.ndarray,
        combined_od: dict,
    ) -> np.ndarray:
        """ Draws active tracker bounding boxes on a copy of the frame. """
        vis = frame.copy()
        for trk in combined_od["trackers"]:
            if trk["age"] > 0:
                continue
            b = trk["box"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

            # Green for persons, cyan for other objects.
            if trk["label"] == "PERSON":
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{trk['tracker_id']} {trk['label']}"
            cv2.putText(vis, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    def _save_combined_detection(
        self,
        frame: np.ndarray,
        combined_od: dict,
        frame_id: int,
    ) -> str:
        """ Saves the detection visualization and a JSON sidecar; returns the image path. """
        det_dir = os.path.join(self.output_root, "detection")
        os.makedirs(det_dir, exist_ok=True)

        vis = self._draw_detections(frame, combined_od)

        img_path = os.path.join(det_dir, f"{frame_id:06d}.jpg")
        cv2.imwrite(img_path, vis)

        # Save the raw detection/tracker state for offline inspection.
        json_path = os.path.join(det_dir, f"{frame_id:06d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_od, f, indent=2, ensure_ascii=False)

        return img_path

    def _gather_rule_metadata(
        self,
        frame_id: int,
        rule: dict,
        ctx: dict,
    ) -> dict:
        """ Collects OD/pose/depth condition results for one alert rule into a metadata dict.

        This metadata is included in the VLM context and stored in the log for diagnostics.

        Returns a dict with:
          "condition": the alert rule condition string
          "metadata":  a human-readable string summarizing pose, depth, and target-pair state
        """
        condition = rule.get("condition", "")
        params = rule.get("parameters", {})
        distance_threshold = params.get("distance_m")
        metadata_parts = []

        main_target_trackers = ctx.get("main_target_trackers", [])

        # Append pose state for each tracked main-target person.
        if self.use_pose and ctx.get("pose_results"):
            pose_obs = []
            for trk in main_target_trackers:
                pose_r = ctx.get("pose_results", {}).get(trk.id)
                if pose_r:
                    pose_obs.append(f"trk{trk.id}={pose_r.get('posture', 'unknown')}")
            if pose_obs:
                metadata_parts.append(f"pose=[{', '.join(pose_obs)}]")

        # Append depth difference relative to the distance threshold if configured.
        if self.use_depth and distance_threshold is not None and ctx.get("depth_result"):
            target_pair = rule.get("target_pair", [])
            other_label = None
            for t in target_pair:
                if t and t != "main" and t.lower() != self.main_target.lower():
                    other_label = t
                    break

            if other_label and main_target_trackers:
                other_label_upper = other_label.upper()
                other_tid = None
                for trk in self._get_all_active_trackers():
                    if trk.label_name == other_label_upper:
                        other_tid = trk.id
                        break

                if other_tid is not None:
                    main_tid = main_target_trackers[0].id
                    # Find the comparison entry that matches the main/other tracker ID pair.
                    for comp in ctx["depth_result"].get("depth_comparisons", []):
                        ids = {comp.get("obj1_id"), comp.get("obj2_id")}
                        if ids == {main_tid, other_tid}:
                            depth_diff = comp.get("depth_diff", None)
                            if depth_diff is not None:
                                within_threshold = abs(depth_diff) <= distance_threshold
                                metadata_parts.append(
                                    f"depth_diff={depth_diff:.2f}m, "
                                    f"threshold={distance_threshold}m, "
                                    f"within={within_threshold}"
                                )

        # Append which target-pair labels are currently detected.
        target_pair = rule.get("target_pair", [])
        if len(target_pair) > 1:
            active_labels = {trk.label_name for trk in self._get_all_active_trackers()}
            pair_labels = []
            for t in target_pair:
                if t == "main":
                    pair_labels.append(self.main_target.upper())
                elif t:
                    pair_labels.append(t.upper())
            detected = [lbl for lbl in pair_labels if lbl in active_labels]
            metadata_parts.append(f"targets={detected}/{pair_labels}")

        metadata_str = ", ".join(metadata_parts) if metadata_parts else "none"
        return {
            "condition": condition,
            "metadata": metadata_str,
        }

    def _build_tool_context(self, ctx: dict, video_time: float = 0.0) -> dict:
        """ Assembles the CV tool output dict passed to EventClassifier as auxiliary context. """
        tool_ctx: Dict[str, Any] = {
            "disclaimer": "These are automated CV tool outputs for REFERENCE ONLY. "
                          "They may be inaccurate. Always prioritize visual observation from the image.",
            "posture_analysis": [],
            "spatial_detections": [],
            "depth_distance_analysis": None,
            "previous_frame": None,
        }

        # Format pose analysis for each tracked person.
        for trk_id, pose in ctx.get("pose_results", {}).items():
            posture = pose.get("posture", "unknown")
            angle = pose.get("body_angle", 0.0)
            detail = pose.get("pose_detail", {})
            head_pos = detail.get("head_position", "unknown")
            visibility = detail.get("visibility", {})
            arm = detail.get("arm_state", {})
            leg = detail.get("leg_state", {})
            proportion = detail.get("body_proportion", {})

            if posture == "unknown" and angle == 0.0:
                angle_desc = "Could not determine (insufficient keypoint visibility)"
            elif angle < 30:
                angle_desc = "Vertical/Standing"
            elif angle > 60:
                angle_desc = "Horizontal/Collapsed"
            else:
                angle_desc = "Ambiguous/Sitting/Bent"

            posture_entry = {
                "tracker_id": trk_id,
                "numeric_angle": f"{angle:.1f} degrees",
                "angle_description": angle_desc,
                "system_posture_hint": posture,
                "head_position": head_pos,
                "keypoint_visibility": f"{visibility.get('total', 0)}/17 "
                                       f"(upper:{visibility.get('upper', 0)}, "
                                       f"lower:{visibility.get('lower', 0)})",
                "arm_state": f"L={arm.get('left', '?')}, R={arm.get('right', '?')}",
                "leg_state": f"L={leg.get('left', '?')}, R={leg.get('right', '?')}",
                "body_aspect_ratio": proportion.get("aspect_ratio", 0.0),
            }
            tool_ctx["posture_analysis"].append(posture_entry)

        # Format spatial detections from active trackers.
        od = ctx.get("od_result")
        if od:
            tool_ctx["spatial_detections"] = [
                {
                    "id": trk["tracker_id"],
                    "label": trk["label"],
                    "box_coordinates": trk["box"],
                    "status": "Active" if trk["age"] == 0 else "Inferred",
                }
                for trk in od.get("trackers", []) if trk["age"] == 0
            ]

        # Format depth estimates as human-readable strings.
        depth = ctx.get("depth_result")
        if depth:
            tool_ctx["depth_distance_analysis"] = [
                f"Object {od_item.get('id')} ({od_item.get('type')}) "
                f"is at a median depth of {od_item['depth_info']['median_depth']:.3f}"
                for od_item in depth.get("object_depths", [])
            ]

        # Attach the previous frame summary for temporal reference.
        if self._prev_frame_context:
            prev = self._prev_frame_context
            time_ago = video_time - prev.get("video_time_sec", video_time)
            tool_ctx["previous_frame"] = {
                "time_ago_sec": round(time_ago, 3),
                "detections": prev.get("detections", []),
                "poses": prev.get("poses", []),
                "depths": prev.get("depths", []),
                "vlm_results": prev.get("vlm_results", []),
            }

        return tool_ctx

    def _build_prev_frame_context(
        self,
        video_time: float,
        ctx: dict,
        frame_vlm_results: list,  # VLM answers from this frame (stored as temporal reference).
    ) -> dict:
        """ Builds a lightweight summary of the current frame to be used as previous-frame context.

        Includes only active tracker positions and sizes; poses, depths, and VLM results
        are intentionally excluded to avoid the model inferring causality from past answers.

        Note:
          VLM results are excluded here — the classifier is explicitly instructed not to
          infer events from previous VLM answers.
        """
        prev_ctx: Dict[str, Any] = {
            "video_time_sec": video_time,
            "detections": [],
        }

        # Record the center and size of each active tracker's bounding box.
        od = ctx.get("od_result")
        if od:
            for trk in od.get("trackers", []):
                if trk["age"] == 0:
                    box = trk.get("box", [0, 0, 0, 0])
                    prev_ctx["detections"].append({
                        "tracker_id": trk["tracker_id"],
                        "label": trk["label"],
                        "center": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                        "size": [box[2] - box[0], box[3] - box[1]],
                    })

        return prev_ctx

    def _record_step_time(self, module_name: str, elapsed: float, scope: str = "invocation"):
        if module_name not in self.step_times:
            self.step_times[module_name] = []
        self.step_times[module_name].append(elapsed)
        self.latency_events.append({
            "run_seed": self.run_seed,
            "variant": self.variant,
            "scenario_id": self.scenario_id,
            "video_id": self.video_id,
            "component": module_name,
            "scope": scope,
            "frame_index": self._current_frame_id,
            "video_timestamp_sec": self._current_video_time,
            "rule_index": self._current_rule_index,
            "latency_sec": round(float(elapsed), 6),
        })

    def _compute_avg_step_times(self) -> dict:
        return {
            module: round(sum(times) / len(times), 4) if times else 0.0
            for module, times in self.step_times.items()
        }

    def _compute_step_latency_summary(self) -> List[dict]:
        return summarize_latencies(self.latency_events)

    def _write_audit_artifacts(self, result: Dict[str, Any]) -> None:
        """Persist raw audit events beside the per-scenario outputs."""
        if not self.audit_dir:
            return
        append_jsonl(os.path.join(self.audit_dir, "frame_audit.jsonl"), self.frame_audit_records)
        append_jsonl(os.path.join(self.audit_dir, "grounding_events.jsonl"), self.grounding_events)
        append_jsonl(os.path.join(self.audit_dir, "tracking_events.jsonl"), self.tracking_events)
        append_jsonl(os.path.join(self.audit_dir, "latency_events.jsonl"), self.latency_events)
        write_csv(os.path.join(self.audit_dir, "frame_schedule.csv"), result["frame_schedule"])
        write_csv(
            os.path.join(self.audit_dir, "latency_summary.csv"),
            result["step_latency_summary"],
        )
        write_json(os.path.join(self.audit_dir, "audit_summary.json"), result["audit_summary"])

    def _get_combined_alert_intervals(self) -> List[List[float]]:
        """ Returns all alert intervals from all rules, sorted by start time. """
        all_intervals = []
        for intervals in self._alert_intervals.values():
            all_intervals.extend(intervals)
        all_intervals.sort(key=lambda x: x[0])
        return all_intervals

    def save_vlm_visualizations(self, output_dir: str, scenario_id: str):
        """ Saves two timeline plots per alert rule:
          1. VLM trigger timeline (was VLM called this frame? 0/1)
          2. VLM classifier timeline (did VLM answer Yes? 0/1)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Executor] WARNING: matplotlib not installed, skipping VLM visualization")
            return

        rule_count = len(self.alert_rules)
        if rule_count == 0:
            return

        # Collect per-rule time series from the structured log.
        rule_data = {r: {"times": [], "called": [], "yes": []} for r in range(rule_count)}

        for fid in self.log._frames:
            frame = self.log._frames[fid]
            t = frame["video_time_sec"]

            rule_alerts = {}
            for alert in frame["alerts"]:
                rule_alerts[alert["rule_index"]] = alert

            for r in range(rule_count):
                rule_data[r]["times"].append(t)
                if r in rule_alerts:
                    a = rule_alerts[r]
                    called = 1 if a["vlm_triggered"] is not None else 0
                    yes = 1 if a["vlm_triggered"] is True else 0
                else:
                    called = 0
                    yes = 0
                rule_data[r]["called"].append(called)
                rule_data[r]["yes"].append(yes)

        if not rule_data[0]["times"]:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Compute shared x-axis range with a small margin.
        all_times = rule_data[0]["times"]
        x_min = min(all_times)
        x_max = max(all_times)
        x_margin = (x_max - x_min) * 0.02 if x_max > x_min else 0.5
        xlim = (x_min - x_margin, x_max + x_margin)

        # Figure size scales with the number of rules.
        fig_w, fig_h = 14, 2.5 * rule_count + 0.8

        # Plot 1: VLM trigger timeline (was VLM called this frame?).
        fig, axes = plt.subplots(rule_count, 1,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)
        for r in range(rule_count):
            ax = axes[r][0]
            times = rule_data[r]["times"]
            called = rule_data[r]["called"]
            condition = self.alert_rules[r].get("condition", f"rule_{r}")
            ax.fill_between(times, called, step='post', alpha=0.7, color='steelblue')
            ax.set_ylabel('VLM Called')
            ax.set_xlim(xlim)
            ax.set_ylim(-0.1, 1.3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0 (Not Called)', '1 (Called)'])
            ax.set_title(f'Rule {r}: {condition}')
            if r == rule_count - 1:
                ax.set_xlabel('Video Time (sec)')
        fig.suptitle(f'{scenario_id} — VLM Trigger', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        trig_path = os.path.join(output_dir, f"{scenario_id}_VLM_trig.png")
        fig.savefig(trig_path, dpi=150)
        plt.close(fig)

        # Plot 2: VLM classifier timeline (did VLM answer Yes?).
        fig, axes = plt.subplots(rule_count, 1,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)
        for r in range(rule_count):
            ax = axes[r][0]
            times = rule_data[r]["times"]
            yes = rule_data[r]["yes"]
            condition = self.alert_rules[r].get("condition", f"rule_{r}")
            ax.fill_between(times, yes, step='post', alpha=0.7, color='orangered')
            ax.set_ylabel('VLM Answer')
            ax.set_xlim(xlim)
            ax.set_ylim(-0.1, 1.3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0 (No)', '1 (Yes)'])
            ax.set_title(f'Rule {r}: {condition}')
            if r == rule_count - 1:
                ax.set_xlabel('Video Time (sec)')
        fig.suptitle(f'{scenario_id} — VLM Classifier', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        cls_path = os.path.join(output_dir, f"{scenario_id}_VLM_classifier.png")
        fig.savefig(cls_path, dpi=150)
        plt.close(fig)

        print(f"[Executor] VLM visualizations saved:")
        print(f"  - {trig_path}")
        print(f"  - {cls_path}")

    def close(self):
        """ Flushes the structured log and resets all tracker state. """
        self.log.close()
        if self.person_detector:
            self.person_detector.reset_trackers()
        if self.object_detector:
            self.object_detector.reset_trackers()
        print("[Executor] Resources released.")
