# modules/tools/object_detector.py
"""
2-Step Object Detection Pipeline:
  Step 1 — PersonDetector: YOLO11l detects persons (COCO class 0).
  Step 2 — ObjectDetector: OmDet-Turbo detects open-vocabulary object classes.

Both detectors share a common Kalman-filter tracker mixin (_TrackerMixin) with
separate tracker ID ranges to avoid collisions:
    PersonDetector: IDs 0 ~ 9999
    ObjectDetector: IDs 10000 ~

Usage:
    person_det = PersonDetector(device="cuda:1")
    result = person_det.detect(frame)

    object_det = ObjectDetector(device="cuda:1")
    result = object_det.detect(frame, target_classes=["forklift", "vehicle"])
"""

import cv2
import torch
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from transformers import AutoProcessor, OmDetTurboForObjectDetection
from ultralytics import YOLO
import os


# ================================================================
# Kalman Filter Tracker
# ================================================================
class KalmanTracker:
    _id_counter = 0  # Class-level counter used as fallback when no explicit ID is given.

    def __init__(self, box: np.ndarray, label_name: str, tracker_id: Optional[int] = None):
        if tracker_id is not None:
            self.id = tracker_id
        else:
            self.id = KalmanTracker._id_counter
            KalmanTracker._id_counter += 1

        # 4-dimensional Kalman filter: state and measurement are both [x1, y1, x2, y2].
        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array(
            [[box[0]], [box[1]], [box[2]], [box[3]]], np.float32
        )

        self.last_box = box.copy()
        self.label_name = label_name
        self.age = 0  # Frames since last matched detection (0 = matched this frame).

        # Track box height to detect sudden shrinkage (e.g., occlusion / fall).
        self.prev_height = box[3] - box[1]
        self.unstable_count = 0
        self.pose_conf_hist = []

    def predict(self) -> np.ndarray:
        pred = self.kf.predict()
        self.last_box = pred.flatten().astype(int)
        return self.last_box

    def update(self, box: np.ndarray):
        h = box[3] - box[1]

        # Increment instability counter when the box height drops abruptly (>30%).
        if h < self.prev_height * 0.7:
            self.unstable_count += 1
        else:
            self.unstable_count = max(0, self.unstable_count - 1)

        self.prev_height = h

        measurement = np.array(
            [[box[0]], [box[1]], [box[2]], [box[3]]], np.float32
        )
        self.kf.correct(measurement)
        self.last_box = box.copy()
        self.age = 0

# ================================================================
# Shared tracker management mixin
# ================================================================
class _TrackerMixin:
    """ Kalman-tracker lifecycle management shared by PersonDetector and ObjectDetector.

    Tracker IDs are assigned from a per-class offset to avoid collisions:
      - PersonDetector: 0 ~ 9999
      - ObjectDetector: 10000 ~
    """

    def _init_trackers(self, max_tracker_age: int, structure_log, id_offset: int = 0):
        self.trackers: List[KalmanTracker] = []
        self.max_tracker_age = max_tracker_age
        self.structure_log = structure_log
        self._next_tracker_id = id_offset
        self._id_offset = id_offset
        self._last_tracker_events: List[dict] = []

    def _create_tracker(self, box: np.ndarray, label: str) -> KalmanTracker:
        tid = self._next_tracker_id
        self._next_tracker_id += 1
        return KalmanTracker(box, label, tracker_id=tid)

    def _calculate_iou(self, b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def _get_iou_threshold(self, tracker: KalmanTracker) -> float:
        """ Returns a lower IoU threshold for unstable trackers to improve re-match rate.

        An unstable tracker (e.g., target partially occluded or rapidly changing size)
        uses a relaxed threshold of 0.1 so that nearby detections are more readily
        matched rather than spawning duplicate tracks.
        """
        return 0.1 if tracker.unstable_count >= 2 else 0.3

    def _update_trackers(self, detections: List[dict]):
        """ Matches detections to existing trackers via greedy IoU and updates state. """
        remaining = list(detections)
        new_trackers = []
        events: List[dict] = []

        for t in self.trackers:
            pred = t.predict()
            iou_th = self._get_iou_threshold(t)

            match_idx, best_iou = -1, 0
            for idx, d in enumerate(remaining):
                iou = self._calculate_iou(pred, d["box"])
                if iou > iou_th and iou > best_iou:
                    best_iou, match_idx = iou, idx

            if match_idx >= 0:
                t.update(np.array(remaining[match_idx]["box"]))
                del remaining[match_idx]
                new_trackers.append(t)
                events.append({"event": "match", "tracker_id": t.id, "label": t.label_name})
            elif t.age < self.max_tracker_age:
                t.age += 1
                new_trackers.append(t)
                events.append({"event": "coast", "tracker_id": t.id, "label": t.label_name, "age": t.age})
            else:
                # Age expired: attempt appearance-based re-identification before dropping.
                events.append({"event": "reid_attempted", "tracker_id": t.id, "label": t.label_name})
                reid_idx = self._try_reidentify(t, remaining)
                if reid_idx is not None:
                    t.update(np.array(remaining[reid_idx]["box"]))
                    del remaining[reid_idx]
                    new_trackers.append(t)
                    events.append({"event": "reid_succeeded", "tracker_id": t.id, "label": t.label_name})
                # If re-ID fails, the tracker is dropped (not appended).
                else:
                    events.append({"event": "reid_failed", "tracker_id": t.id, "label": t.label_name})
                    events.append({"event": "drop", "tracker_id": t.id, "label": t.label_name})

        # Spawn new trackers for unmatched detections.
        for d in remaining:
            tracker = self._create_tracker(np.array(d["box"]), d["label"])
            new_trackers.append(tracker)
            events.append({"event": "create", "tracker_id": tracker.id, "label": tracker.label_name})

        self.trackers = new_trackers
        self._last_tracker_events = events

    def _try_reidentify(self, tracker: KalmanTracker, remaining: List[dict]) -> Optional[int]:
        """ Attempts to re-identify a lost tracker using size and position history. """
        if not self.structure_log or not remaining:
            return None

        history = self.structure_log.get_tracker_history(tracker.id, last_n_frames=5)
        if not history:
            return None

        # Compute average size and center from recent detection history.
        widths = [h["x2"] - h["x1"] for h in history]
        heights = [h["y2"] - h["y1"] for h in history]
        cx_list = [(h["x1"] + h["x2"]) / 2 for h in history]
        cy_list = [(h["y1"] + h["y2"]) / 2 for h in history]

        avg_w = np.mean(widths)
        avg_h = np.mean(heights)
        avg_cx = np.mean(cx_list)
        avg_cy = np.mean(cy_list)

        best_idx = None
        best_score = float("inf")

        for idx, det in enumerate(remaining):
            if det["label"] != tracker.label_name:
                continue

            box = det["box"]
            det_w = box[2] - box[0]
            det_h = box[3] - box[1]
            det_cx = (box[0] + box[2]) / 2
            det_cy = (box[1] + box[3]) / 2

            # Reject candidates with drastically different aspect ratios.
            w_ratio = det_w / (avg_w + 1e-6)
            h_ratio = det_h / (avg_h + 1e-6)
            if w_ratio < 0.4 or w_ratio > 2.5 or h_ratio < 0.4 or h_ratio > 2.5:
                continue

            # Reject candidates too far from the last known center.
            diag = (avg_w**2 + avg_h**2) ** 0.5
            dist = ((det_cx - avg_cx)**2 + (det_cy - avg_cy)**2) ** 0.5
            if dist > diag * 4.0:
                continue

            # Score = size deviation + normalized displacement (lower is better).
            size_diff = abs(w_ratio - 1.0) + abs(h_ratio - 1.0)
            pos_diff = dist / (diag + 1e-6)
            score = size_diff + pos_diff

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _get_tracker_states(self) -> List[dict]:
        return [
            {
                "tracker_id": t.id,
                "label": t.label_name,
                "box": t.last_box.tolist() if isinstance(t.last_box, np.ndarray) else list(t.last_box),
                "age": t.age,
                "unstable_count": t.unstable_count,
            }
            for t in self.trackers
        ]

    def get_active_trackers(self) -> List[KalmanTracker]:
        """ Returns only trackers that were matched to a detection this frame (age == 0). """
        return [t for t in self.trackers if t.age == 0]

    def reset_trackers(self):
        """ Clears all tracker state and resets the ID counter to the class offset. """
        self.trackers = []
        self._next_tracker_id = self._id_offset
        self._last_tracker_events = []


# ================================================================
# Person Detector (YOLO + Kalman) — ID: 0~9999
# ================================================================
class PersonDetector(_TrackerMixin):
    """ YOLO11l-based person detector with Kalman-filter tracking.

    Restricted to COCO class 0 (person); tracker IDs start at 0.
    """
    PERSON_ID_OFFSET = 0

    def __init__(
        self,
        model_path: str = "yolo11l.pt",
        score_threshold: float = 0.15,
        nms_iou_threshold: float = 0.45,
        device: str = "cuda:1",
        max_tracker_age: int = 45,
        structure_log=None,
    ):
        if "cuda" in device and not torch.cuda.is_available():
            print("[PersonDetector] CUDA unavailable -> CPU")
            device = "cpu"

        self.device = device
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._init_trackers(max_tracker_age, structure_log, self.PERSON_ID_OFFSET)

        print(f"[PersonDetector] Loading YOLO v11l on {device}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"[PersonDetector] Model loaded.")

    # ================================================================
    # Main Detection
    # ================================================================
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """ Detects persons in the frame and updates Kalman trackers.

        Returns:
            {
                "detections": [{"box": [x1,y1,x2,y2], "label": "PERSON", "score": float}],
                "trackers": [{"tracker_id": int, "label": str, "box": [...], "age": int, "unstable_count": int}],
            }
        """
        results = self.model.predict(
            frame,
            classes=[0],  # COCO class 0 = person
            conf=self.score_threshold,
            iou=self.nms_iou_threshold,
            half=True,
            verbose=False,
        )

        raw_detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                score = float(boxes.conf[i].cpu().numpy())
                raw_detections.append({
                    "box": xyxy.tolist(),
                    "label": "PERSON",
                    "score": score,
                })

        self._update_trackers(raw_detections)

        return {
            "detections": raw_detections,
            "trackers": self._get_tracker_states(),
            "tracker_events": list(self._last_tracker_events),
        }

    # ================================================================
    # Visualization
    # ================================================================
    def save_results(self, frame: np.ndarray, result_dict: dict, frame_id: int, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f"{frame_id:06d}_person.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)


# ================================================================
# Object Detector (OmDet-Turbo + Kalman) — ID: 10000~
# ================================================================
class ObjectDetector(_TrackerMixin):
    """ OmDet-Turbo open-vocabulary object detector with Kalman-filter tracking.

    Detects arbitrary object classes specified at inference time; tracker IDs start at 10000.
    """
    OBJECT_ID_OFFSET = 10000

    def __init__(
        self,
        model_id: str = "omlab/omdet-turbo-swin-tiny-hf",
        score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.45,
        device: str = "cuda:1",
        max_tracker_age: int = 45,
        structure_log=None,
    ):
        """
        Args:
            model_id:          HuggingFace model ID for OmDet-Turbo.
            score_threshold:   Minimum detection confidence.
            nms_iou_threshold: IoU threshold for NMS.
            device:            Torch device string.
            max_tracker_age:   Frames a tracker survives without a matching detection.
            structure_log:     StructureLog instance used for re-ID history lookup.
        """
        if "cuda" in device and not torch.cuda.is_available():
            print("[ObjectDetector] CUDA unavailable -> CPU")
            device = "cpu"

        self.device = device
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._init_trackers(max_tracker_age, structure_log, self.OBJECT_ID_OFFSET)

        # Load OmDet-Turbo processor and model from HuggingFace.
        print(f"[ObjectDetector] Loading OmDet-Turbo on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print(f"[ObjectDetector] Model loaded.")

    # ================================================================
    # Main Detection
    # ================================================================
    def detect(self, frame: np.ndarray, target_classes: List[str]) -> Dict[str, Any]:
        """
        Detects objects belonging to target_classes in the frame and updates Kalman trackers.

        Args:
            frame: BGR np.ndarray (H, W, 3)
            target_classes: List of open-vocabulary class names to detect (e.g., ["forklift"]).

        Returns:
            {
                "detections": [{"box": [x1,y1,x2,y2], "label": str, "score": float}],
                "trackers": [{"tracker_id": int, "label": str, "box": [...], "age": int, "unstable_count": int}],
            }
        """
        if not target_classes:
            self._last_tracker_events = []
            return {
                "detections": [],
                "trackers": self._get_tracker_states(),
                "tracker_events": [],
            }

        # Convert BGR frame to RGB PIL Image for the OmDet processor.
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=image_pil, text=target_classes, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=target_sizes,
                classes=target_classes,
                score_threshold=self.score_threshold,
            )[0]

        # Apply NMS to remove duplicate boxes from the grounded detection output.
        raw_detections = []
        labels_out = results.get("labels", results.get("classes", []))
        boxes_tensor = results.get("boxes", torch.tensor([]))
        scores_tensor = results.get("scores", torch.tensor([]))

        if len(boxes_tensor) > 0:
            boxes_np = boxes_tensor.cpu().numpy().astype(int)
            scores_np = scores_tensor.cpu().numpy()

            indices = cv2.dnn.NMSBoxes(
                boxes_np.tolist(), scores_np.tolist(),
                self.score_threshold, self.nms_iou_threshold
            )

            if len(indices) > 0:
                for i in indices.flatten():
                    label = str(labels_out[i]).upper()
                    raw_detections.append({
                        "box": boxes_np[i].tolist(),
                        "label": label,
                        "score": float(scores_np[i]),
                    })

        # Update Kalman trackers with the new detections.
        self._update_trackers(raw_detections)

        return {
            "detections": raw_detections,
            "trackers": self._get_tracker_states(),
            "tracker_events": list(self._last_tracker_events),
        }

    # ================================================================
    # Visualization
    # ================================================================
    def save_results(self, frame: np.ndarray, result_dict: dict, frame_id: int, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # Save detection/tracker state as JSON.
        json_path = os.path.join(output_dir, f"{frame_id:06d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        # Save a visualization image with active tracker boxes overlaid.
        vis = frame.copy()
        for trk in result_dict["trackers"]:
            if trk["age"] > 0: continue  # Skip coasting trackers.
            b = trk["box"]
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
            cv2.putText(vis, f"ID:{trk['tracker_id']} {trk['label']}",
                        (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imwrite(os.path.join(output_dir, f"{frame_id:06d}.jpg"), vis)
