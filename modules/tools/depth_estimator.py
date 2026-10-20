# modules/tools/depth_estimator.py

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from datetime import datetime
import os
import cv2
import json


class DepthEstimator:
    """
    Monocular depth estimation using Depth-Anything-V2.

    Supports both relative depth (MODEL_VARIANTS) and metric depth in metres
    (METRIC_VARIANTS). For relative depth, larger values mean closer to the
    camera. For metric variants, values are in metres and smaller = closer.

    Typical usage:
      estimator = DepthEstimator(model_variant="small", device="cuda:1")
      depth_map = estimator.estimate_depth(frame)
      depth_info = estimator.get_object_depth(depth_map, bbox=[x1, y1, x2, y2])
    """

    # Relative-depth variants (Apache-2.0 / CC-BY-NC-4.0).
    MODEL_VARIANTS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",        # ~25M,  Apache-2.0
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",         # ~97M,  CC-BY-NC-4.0
        "large": "depth-anything/Depth-Anything-V2-Large-hf",        # ~335M, CC-BY-NC-4.0
    }

    # Metric-depth variants (values are in metres).
    METRIC_VARIANTS = {
        "outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "outdoor-base":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "indoor-small":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "indoor-base":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "indoor-large":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    }

    def __init__(
        self,
        model_variant: str = "small",
        device: str = "cuda:1",
        save_depth_dir: Optional[str] = None,
    ):
        """
        Args:
            model_variant: Key from MODEL_VARIANTS / METRIC_VARIANTS, or a full HF model ID.
            device:        Torch device string (e.g. "cuda:1", "cpu").
            save_depth_dir: If set, depth visualizations are auto-saved here.
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("[DepthEstimator] CUDA unavailable -> switching to CPU")
            device = "cpu"

        self.device = device
        self.save_depth_dir = save_depth_dir

        if save_depth_dir:
            os.makedirs(save_depth_dir, exist_ok=True)

        # Resolve model ID from shorthand key or treat as a raw HF repo ID.
        if model_variant in self.MODEL_VARIANTS:
            model_id = self.MODEL_VARIANTS[model_variant]
        elif model_variant in self.METRIC_VARIANTS:
            model_id = self.METRIC_VARIANTS[model_variant]
        else:
            model_id = model_variant  # Treat as a full HF model ID.

        self.model_id = model_id
        self.model_variant = model_variant

        print(f"[DepthEstimator] Loading {model_id} on {device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
            self.model.eval()
            print(f"[DepthEstimator] Model loaded successfully ({model_variant})")
        except Exception as e:
            print(f"[DepthEstimator] Failed to load model: {e}")
            raise e

    # =================================================================
    # Core Inference
    # =================================================================
    @torch.no_grad()
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs depth estimation on a single frame.

        Args:
            frame: BGR np.ndarray (H, W, 3) or a PIL Image.

        Returns:
            depth_map: np.ndarray (float32, H x W)
                       Values are relative disparity for MODEL_VARIANTS,
                       or metric depth in metres for METRIC_VARIANTS.
        """
        # Convert OpenCV BGR to RGB PIL Image if necessary.
        if isinstance(frame, np.ndarray):
            # OpenCV BGR -> RGB
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            image_pil = Image.fromarray(frame_rgb)
        else:
            image_pil = frame

        orig_h, orig_w = image_pil.height, image_pil.width

        # Preprocess into model input tensors.
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        # Forward pass.
        outputs = self.model(**inputs)

        # Upsample prediction back to the original image resolution.
        post_processed = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(orig_h, orig_w)],
        )

        depth_map = post_processed[0]["predicted_depth"].cpu().numpy().astype(np.float32)

        return depth_map

    # =================================================================
    # Safety Monitoring API
    # =================================================================
    def get_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: List[int],
        use_center_crop: bool = True,
        center_ratio: float = 0.5,
    ) -> Dict[str, float]:
        """
        Returns depth statistics for the region covered by a bounding box.

        Args:
            depth_map: Float32 depth map from estimate_depth().
            bbox: [x1, y1, x2, y2] in pixel coordinates.
            use_center_crop: If True, sample only the central region of the bbox
                             to reduce background contamination.
            center_ratio: Fraction of the bbox dimensions used for the center crop.

        Returns:
            {
                "mean_depth": float,
                "median_depth": float,
                "min_depth": float,
                "max_depth": float,
            }
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        # Clamp bbox to the depth map boundaries.
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return {"mean_depth": 0.0, "median_depth": 0.0, "min_depth": 0.0, "max_depth": 0.0}

        if use_center_crop:
            # Shrink the ROI to a centered sub-region to avoid background edges.
            bw = x2 - x1
            bh = y2 - y1
            margin_x = int(bw * (1 - center_ratio) / 2)
            margin_y = int(bh * (1 - center_ratio) / 2)
            cx1 = x1 + margin_x
            cy1 = y1 + margin_y
            cx2 = x2 - margin_x
            cy2 = y2 - margin_y
            roi = depth_map[cy1:cy2, cx1:cx2]
        else:
            roi = depth_map[y1:y2, x1:x2]

        if roi.size == 0:
            return {"mean_depth": 0.0, "median_depth": 0.0, "min_depth": 0.0, "max_depth": 0.0}

        return {
            "mean_depth": float(np.mean(roi)),
            "median_depth": float(np.median(roi)),
            "min_depth": float(np.min(roi)),
            "max_depth": float(np.max(roi)),
        }

    def compare_object_depths(
        self,
        depth_map: np.ndarray,
        objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Computes pairwise depth comparisons for a list of detected objects.

        Args:
            depth_map: Float32 depth map from estimate_depth().
            objects: [{"id": int, "type": str, "bbox": [x1,y1,x2,y2]}, ...]

        Returns:
            [
                {
                    "obj1_id": int, "obj1_type": str,
                    "obj2_id": int, "obj2_type": str,
                    "obj1_depth": float, "obj2_depth": float,
                    "depth_diff": float,
                    "closer_object": str,
                },
                ...
            ]
        """
        # Compute median depth for each object.
        obj_depths = []
        for obj in objects:
            depth_info = self.get_object_depth(depth_map, obj["bbox"])
            obj_depths.append({
                **obj,
                "depth_info": depth_info,
            })

        # Generate all unique pairs and determine which object is closer.
        comparisons = []
        for i, o1 in enumerate(obj_depths):
            for o2 in obj_depths[i + 1:]:
                d1 = o1["depth_info"]["median_depth"]
                d2 = o2["depth_info"]["median_depth"]

                # For metric models, smaller depth = closer to camera.
                # For relative models, larger disparity = closer to camera.
                # "Metric" appears in the HF model ID for metric variants.
                if "Metric" in self.model_id:
                    closer = o1.get("type") if d1 < d2 else o2.get("type")
                else:
                    closer = o1.get("type") if d1 > d2 else o2.get("type")

                comparisons.append({
                    "obj1_id": o1.get("id", i),
                    "obj1_type": o1.get("type", "unknown"),
                    "obj2_id": o2.get("id", i + 1),
                    "obj2_type": o2.get("type", "unknown"),
                    "obj1_depth": round(d1, 3),
                    "obj2_depth": round(d2, 3),
                    "depth_diff": round(abs(d1 - d2), 3),
                    "closer_object": closer,
                })

        return comparisons

    # =================================================================
    # Visualization
    # =================================================================
    def visualize_depth(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalizes and colorizes a depth map for visualization.

        Args:
            depth_map: Float32 depth map from estimate_depth().
            colormap:  OpenCV colormap constant (default: COLORMAP_INFERNO).
            save_path: If provided, saves the image to this path.

        Returns:
            BGR colorized depth image (uint8).
        """
        # Normalize depth values to [0, 255].
        d_min = depth_map.min()
        d_max = depth_map.max()

        if d_max - d_min > 0:
            depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply false-color mapping.
        depth_colored = cv2.applyColorMap(depth_norm, colormap)

        # Save to explicit path or fall back to the instance-level save directory.
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, depth_colored)
            print(f"[DepthEstimator] Depth map saved to {save_path}")
        elif self.save_depth_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"depth_{timestamp}.jpg"
            fpath = os.path.join(self.save_depth_dir, filename)
            cv2.imwrite(fpath, depth_colored)
            print(f"[DepthEstimator] Depth map saved to {fpath}")

        return depth_colored

    # =================================================================
    # Combined Pipeline (Detection + Depth)
    # =================================================================
    def analyze(
        self,
        frame: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: runs depth estimation and per-object depth extraction
        in a single call.

        Args:
            frame: BGR np.ndarray (H, W, 3)
            detected_objects: [{"type": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
            save_dir: Optional directory for saving the depth visualization.

        Returns:
            {
                "depth_map": np.ndarray (H, W),
                "object_depths": [{"type": str, "bbox": [...], "depth_info": {...}}, ...],
                "depth_comparisons": [...],
                "depth_image_path": str or None,
            }
        """
        # Run monocular depth inference on the full frame.
        depth_map = self.estimate_depth(frame)

        # Extract per-object depth statistics.
        object_depths = []
        for i, obj in enumerate(detected_objects):
            depth_info = self.get_object_depth(depth_map, obj["bbox"])
            object_depths.append({
                "id": obj.get("id", i),
                "type": obj.get("type", "unknown"),
                "confidence": obj.get("confidence", 0.0),
                "bbox": obj["bbox"],
                "depth_info": depth_info,
            })

        # Compute pairwise depth comparisons.
        depth_comparisons = self.compare_object_depths(depth_map, detected_objects)

        # Save colorized depth map if an output directory is configured.
        depth_image_path = None
        output_dir = save_dir or self.save_depth_dir
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            depth_dir = os.path.join(output_dir, "depth_maps")
            os.makedirs(depth_dir, exist_ok=True)
            depth_image_path = os.path.join(depth_dir, f"depth_{timestamp}.jpg")
            self.visualize_depth(depth_map, save_path=depth_image_path)

        return {
            "depth_map": depth_map,
            "object_depths": object_depths,
            "depth_comparisons": depth_comparisons,
            "depth_image_path": depth_image_path,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self, frame: np.ndarray, analyze_result: Dict[str, Any], frame_id: int, output_dir: str):
        """ Persists the depth analysis result (JSON + colorized image) for a single frame. """
        if not analyze_result or "depth_map" not in analyze_result:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Save the analysis dict as JSON, excluding the raw depth array.
        json_path = os.path.join(output_dir, f"{frame_id:06d}.json")
        try:
            save_dict = {k: v for k, v in analyze_result.items() if k != "depth_map"}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Depth] JSON Error: {e}")

        # Save the colorized depth visualization.
        try:
            img_path = os.path.join(output_dir, f"{frame_id:06d}.jpg")
            self.visualize_depth(analyze_result["depth_map"], save_path=img_path)
        except Exception as e:
            print(f"[Depth] Image Error: {e}")
