# modules/tools/pose_estimator.py
"""
Single-person pose estimation using YOLO11l-pose.

Estimates body orientation and keypoint positions for a tracked person,
then classifies posture into one of three states:
  - upright          (body angle < 45°)
  - lying_on_ground  (body angle > 60°)
  - unknown          (insufficient keypoint visibility)

Usage:
    estimator = PoseEstimator(device="cuda:1")
    result = estimator.estimate(frame, bbox=[100, 50, 300, 400])

    # Tracker-aware version (adjusts inference resolution based on tracker stability):
    result = estimator.estimate_with_tracker(frame, tracker)
"""

import os
import cv2
import json
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from ultralytics import YOLO


# COCO 17-keypoint names in index order.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Pairs of keypoints that form the skeleton for visualization.
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


class PoseEstimator:
    def __init__(
        self,
        model_path: str = "yolo11l-pose.pt",
        device: str = "cuda:1",
        conf_threshold: float = 0.1,
        kp_conf_threshold: float = 0.15,
    ):
        """
        Args:
            model_path:        Path to the YOLO pose model weights.
            device:            Torch device string; falls back to CPU if CUDA unavailable.
            conf_threshold:    Minimum YOLO person detection confidence.
            kp_conf_threshold: Minimum keypoint confidence for a keypoint to be considered valid.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.kp_conf_threshold = kp_conf_threshold

        print(f"[PoseEstimator] Loading {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[PoseEstimator] Model loaded.")

    # ================================================================
    # Public inference API
    # ================================================================
    def estimate(
        self,
        frame: np.ndarray,
        bbox: Optional[List[int]] = None,
        imgsz: int = 640,
    ) -> Dict[str, Any]:
        """
        Estimates pose for the person inside the given bounding box.

        Args:
            frame: BGR np.ndarray (H, W, 3)
            bbox:  [x1, y1, x2, y2] region of interest; uses the full frame if None.
            imgsz: YOLO inference image size.

        Returns:
            {
                "posture": str,           # "lying_on_ground" | "upright" | "unknown"
                "body_angle": float,      # degrees from vertical (0° = standing upright)
                "confidence": float,      # YOLO detection confidence
                "keypoints": {
                    "nose": (x, y, conf),
                    "left_shoulder": (x, y, conf),
                    ...
                },
                "keypoints_raw": np.ndarray,  # (17, 3) — x, y, conf
                "mean_kp_conf": float,
            }
            Returns an empty dict if no person is detected in the ROI.
        """
        if frame is None:
            return {}

        # Crop the region of interest with a small padding margin.
        crop, offset_x, offset_y = self._crop_roi(frame, bbox)
        if crop.size == 0:
            return {}

        # Run YOLO pose estimation on the cropped region.
        results = self.model.predict(
            crop,
            imgsz=imgsz,
            conf=self.conf_threshold,
            iou=0.5,
            half=True,
            verbose=False,
        )

        if not results or len(results[0].keypoints) == 0:
            return {}

        # keypoints.data: (N, 17, 3) — x, y, conf
        kpts_data = results[0].keypoints.data
        if kpts_data.shape[0] == 0:
            return {}

        # Take the highest-confidence detection (first in sorted output).
        kpts = kpts_data[0].cpu().numpy()  # (17, 3)

        # Translate keypoint coordinates back to the original frame space.
        kpts_adjusted = kpts.copy()
        for i in range(len(kpts_adjusted)):
            if kpts_adjusted[i][2] > 0:  # only shift visible keypoints
                kpts_adjusted[i][0] += offset_x
                kpts_adjusted[i][1] += offset_y

        # Named keypoints dict
        keypoints = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            x, y, c = kpts_adjusted[i]
            keypoints[name] = (int(x), int(y), round(float(c), 3))

        # Classify posture from the shoulder-to-hip vector angle.
        posture, body_angle = self._classify_posture(keypoints)

        # Compute the mean confidence of all visible keypoints.
        valid_confs = [c for _, _, c in kpts_adjusted if c > 0]
        mean_kp_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

        det_conf = float(results[0].boxes.conf[0].cpu().numpy()) if len(results[0].boxes) > 0 else 0.0

        # Run detailed pose analysis (arm/leg state, head position, proportions).
        pose_detail = self.analyze_pose_detail(keypoints)

        return {
            "posture": posture,
            "posture_type": "state_only",
            "body_angle": round(body_angle, 2),
            "confidence": round(det_conf, 4),
            "keypoints": keypoints,
            "keypoints_raw": kpts_adjusted,
            "mean_kp_conf": round(mean_kp_conf, 4),
            "pose_detail": pose_detail,
        }

    # ================================================================
    # Tracker-aware inference
    # ================================================================
    def estimate_with_tracker(self, frame: np.ndarray, tracker) -> Dict[str, Any]:
        """
        Estimates pose using the tracker's last known bounding box, and updates
        the tracker's pose confidence history to detect instability.

        Args:
            frame:   BGR np.ndarray — current video frame.
            tracker: KalmanTracker instance with .last_box and .pose_conf_hist.

        Returns:
            Pose result dict (same as estimate()), or empty dict if estimation fails.
        """
        bbox = tracker.last_box
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        # Use a higher inference resolution for unstable or small crops.
        crop, _, _ = self._crop_roi(frame, bbox)
        h, w = crop.shape[:2]
        if tracker.unstable_count >= 2 or min(h, w) < 200:
            imgsz = 960
        else:
            imgsz = 640

        result = self.estimate(frame, bbox=bbox, imgsz=imgsz)

        # Update the tracker's rolling keypoint confidence history.
        if result and "mean_kp_conf" in result:
            tracker.pose_conf_hist.append(result["mean_kp_conf"])
            tracker.pose_conf_hist = tracker.pose_conf_hist[-5:]

            if np.mean(tracker.pose_conf_hist) < 0.25:
                tracker.unstable_count += 1

        return result

    # ================================================================
    # ROI cropping
    # ================================================================
    def _crop_roi(self, frame: np.ndarray, bbox: Optional[List[int]] = None):
        """
        Crops the frame to the bounding box with a 20% padding margin.

        Returns:
            (crop, offset_x, offset_y)
        """
        if bbox is None:
            return frame, 0, 0

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h_img, w_img = frame.shape[:2]

        pad_h = int((y2 - y1) * 0.2)
        pad_w = int((x2 - x1) * 0.2)

        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w_img, x2 + pad_w)
        cy2 = min(h_img, y2 + pad_h)

        crop = frame[cy1:cy2, cx1:cx2]
        return crop, cx1, cy1

    # ================================================================
    # Posture Classification (Tri-state: lying_on_ground / upright / unknown)
    # ================================================================
    def _classify_posture(self, kp: dict) -> tuple:
        """ Classifies body orientation from the angle of the shoulder-to-hip vector.

        This method reports ONLY instantaneous body orientation.
        It does NOT imply a fall event or causality.

        Returns:
          posture:    "lying_on_ground" if angle > 60°
                      "upright"         if angle < 45°
                      "unknown"         if key landmarks are not visible
          body_angle: angle in degrees from the vertical axis (0° = perfectly upright)
        """
        ls = kp.get("left_shoulder", (0, 0, 0))
        rs = kp.get("right_shoulder", (0, 0, 0))
        lh = kp.get("left_hip", (0, 0, 0))
        rh = kp.get("right_hip", (0, 0, 0))

        if ls[2] == 0 or rs[2] == 0 or lh[2] == 0 or rh[2] == 0:
            return "unknown", 0.0

        sh_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
        hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])

        vector = hip_mid - sh_mid
        dx = abs(vector[0])
        dy = abs(vector[1])

        angle_deg = float(np.degrees(np.arctan2(dx, dy)))

        if angle_deg > 60:
            posture = "lying_on_ground"
        elif angle_deg < 45:
            posture = "upright"
        else:
            posture = "unknown"

        return posture, angle_deg

    # ================================================================
    # Detailed pose analysis
    # ================================================================
    def analyze_pose_detail(self, kp: dict) -> dict:
        """ Produces a detailed breakdown of visible body parts and their states.

        Analyzes keypoint visibility, arm/leg joint angles, body bounding-box
        proportions, and head position relative to the torso. All values are
        derived from the raw keypoint confidence and coordinates; they are
        heuristic estimates rather than ground-truth measurements.

        Returns:
            {
                "visibility": {"total": int, "upper": int, "lower": int,
                               "total_ratio": float},
                "arm_state": {"left": str, "right": str},
                "leg_state": {"left": str, "right": str},
                "body_proportion": {"height_px": float, "width_px": float,
                                    "aspect_ratio": float},
                "head_position": str,  # "above_torso" | "level_with_torso" | "below_torso"
            }
        """
        # Count visible keypoints in the upper and lower body separately.
        upper_keys = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                       "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                       "left_wrist", "right_wrist"]
        lower_keys = ["left_hip", "right_hip", "left_knee", "right_knee",
                       "left_ankle", "right_ankle"]

        upper_visible = sum(1 for k in upper_keys if kp.get(k, (0,0,0))[2] > self.kp_conf_threshold)
        lower_visible = sum(1 for k in lower_keys if kp.get(k, (0,0,0))[2] > self.kp_conf_threshold)
        total_visible = upper_visible + lower_visible

        # Classify each arm as raised / bent / lowered / partially_visible / not_visible.
        arm_state = {}
        for side in ["left", "right"]:
            sh = kp.get(f"{side}_shoulder", (0,0,0))
            el = kp.get(f"{side}_elbow", (0,0,0))
            wr = kp.get(f"{side}_wrist", (0,0,0))
            if sh[2] > self.kp_conf_threshold and el[2] > self.kp_conf_threshold:
                if wr[2] > self.kp_conf_threshold and wr[1] < sh[1]:
                    arm_state[side] = "raised"
                elif wr[2] > self.kp_conf_threshold and wr[1] > sh[1]:
                    arm_angle = self._calc_angle_3pt(sh, el, wr)
                    arm_state[side] = "bent" if arm_angle < 140 else "lowered"
                else:
                    arm_state[side] = "partially_visible"
            else:
                arm_state[side] = "not_visible"

        # Classify each leg as bent / straight / partially_visible / not_visible.
        leg_state = {}
        for side in ["left", "right"]:
            hp = kp.get(f"{side}_hip", (0,0,0))
            kn = kp.get(f"{side}_knee", (0,0,0))
            ak = kp.get(f"{side}_ankle", (0,0,0))
            if hp[2] > self.kp_conf_threshold and kn[2] > self.kp_conf_threshold:
                if ak[2] > self.kp_conf_threshold:
                    leg_angle = self._calc_angle_3pt(hp, kn, ak)
                    leg_state[side] = "bent" if leg_angle < 150 else "straight"
                else:
                    leg_state[side] = "partially_visible"
            else:
                leg_state[side] = "not_visible"

        # Compute bounding-box dimensions from visible keypoints.
        visible_pts = [(kp[k][0], kp[k][1]) for k in kp
                       if kp[k][2] > self.kp_conf_threshold and kp[k][0] > 0]
        if len(visible_pts) >= 2:
            xs, ys = zip(*visible_pts)
            h_px = max(ys) - min(ys)
            w_px = max(xs) - min(xs)
            aspect = round(h_px / max(w_px, 1.0), 2)
        else:
            h_px, w_px, aspect = 0.0, 0.0, 0.0

        # Determine head position relative to the shoulder midpoint.
        nose = kp.get("nose", (0,0,0))
        ls = kp.get("left_shoulder", (0,0,0))
        rs = kp.get("right_shoulder", (0,0,0))
        if nose[2] > self.kp_conf_threshold and ls[2] > self.kp_conf_threshold and rs[2] > self.kp_conf_threshold:
            sh_mid_y = (ls[1] + rs[1]) / 2
            if nose[1] < sh_mid_y - 20:
                head_pos = "above_torso"
            elif nose[1] > sh_mid_y + 20:
                head_pos = "below_torso"
            else:
                head_pos = "level_with_torso"
        else:
            head_pos = "not_visible"

        return {
            "visibility": {
                "total": total_visible,
                "upper": upper_visible,
                "lower": lower_visible,
                "total_ratio": round(total_visible / 17, 2),
            },
            "arm_state": arm_state,
            "leg_state": leg_state,
            "body_proportion": {
                "height_px": round(h_px, 1),
                "width_px": round(w_px, 1),
                "aspect_ratio": aspect,
            },
            "head_position": head_pos,
        }

    @staticmethod
    def _calc_angle_3pt(a: tuple, b: tuple, c: tuple) -> float:
        """ Returns the angle at vertex b formed by the segments b→a and b→c, in degrees. """
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    # ================================================================
    # Visualization
    # ================================================================
    def draw_pose(
        self,
        frame: np.ndarray,
        pose_result: Dict[str, Any],
        bbox: Optional[List[int]] = None,
    ) -> np.ndarray:
        """ Draws the bounding box, keypoints, skeleton, and posture label onto a copy of the frame. """
        vis = frame.copy()
        kp = pose_result.get("keypoints", {})

        # BBox
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Keypoints
        for name, (x, y, c) in kp.items():
            if c > self.kp_conf_threshold and x > 0 and y > 0:
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

        # Skeleton
        for p1, p2 in SKELETON_CONNECTIONS:
            if p1 in kp and p2 in kp:
                x1, y1, c1 = kp[p1]
                x2, y2, c2 = kp[p2]
                if c1 > self.kp_conf_threshold and c2 > self.kp_conf_threshold:
                    cv2.line(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)

        # Posture label
        posture = pose_result.get("posture", "unknown")
        angle = pose_result.get("body_angle", 0)
        label = f"{posture.upper()} ({angle:.1f} deg)"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return vis

    def save_results(self, frame: np.ndarray, pose_result: Dict[str, Any], frame_id: int, output_dir: str):
        """ Saves the pose result as a JSON file and a visualized JPEG for the given frame. """
        if not pose_result:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Save the pose result dict as JSON.
        json_path = os.path.join(output_dir, f"{frame_id:06d}.json")
        try:
            save_data = pose_result.copy()

            # Convert numpy arrays to lists for JSON serialization.
            if "keypoints_raw" in save_data and isinstance(save_data["keypoints_raw"], np.ndarray):
                save_data["keypoints_raw"] = save_data["keypoints_raw"].tolist()

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PoseEstimator] JSON Save Error at frame {frame_id}: {e}")

        # Save the skeleton visualization as a JPEG.
        try:
            vis = self.draw_pose(frame, pose_result)
            img_path = os.path.join(output_dir, f"{frame_id:06d}.jpg")
            cv2.imwrite(img_path, vis)
        except Exception as e:
            print(f"[PoseEstimator] Image Save Error at frame {frame_id}: {e}")
