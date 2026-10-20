# modules/executor/event_classifier.py
"""
Single-frame visual event classifier backed by the Qwen-VL model.

Sends a full-scene frame (and optional ROI crops) together with structured
CV tool outputs to the VLM and parses a binary Yes/No answer.

Usage:
    classifier = EventClassifier(provider)
    result = await classifier.classify(frame, verification_prompt, tool_context)
    # result = {"answer": "Yes", "raw_response": "Yes", "latency_sec": 0.42}
"""

import base64
import cv2
import numpy as np
from typing import Dict, Any
from modules.qwen_provider import QwenVLProvider


class EventClassifier:
    def __init__(self, provider: QwenVLProvider):
        """
        Args:
            provider: Shared QwenVLProvider instance used to inherit server URL,
                      model name, and timeout; overrides temperature/max_tokens
                      for deterministic binary classification.
        """
        self.provider = QwenVLProvider(
            server_url=provider.server_url,
            model_name=provider.model_name,
            temperature=0.1,
            max_tokens=8,
            timeout=provider.timeout,
            seed=provider.seed,
        )

    async def classify(
        self,
        frame: np.ndarray,
        verification_prompt: dict,
        tool_context: dict,
        roi_crops: list = None,
        od_hit_supplement: str = None,
    ) -> Dict[str, Any]:
        """
        Classifies whether a safety event is occurring in the given frame.

        Sends the full frame plus optional ROI crops to the VLM together with
        CV tool context (pose, depth, detections) as auxiliary reference text.

        Args:
            frame: BGR np.ndarray — the current video frame.
            verification_prompt: {
                "system": str,
                "instruction": str,
                "query": str,
            }
            tool_context: Dict of CV tool outputs (pose, depth, detections, prev frame).
            roi_crops:    Optional list of tight BGR crops around the main target.
            od_hit_supplement: Optional text appended to the query when OD detected the target.

        Returns:
            {"answer": "Yes"|"No"|"Error", "raw_response": str, "latency_sec": float}
        """
        # Encode the full frame as a base64 JPEG data URL.
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_b64}"

        # Build the context text from CV tool outputs.
        context_text = self._build_context_text(tool_context)
        query = verification_prompt.get("query", "Is there a safety event?")

        # System prompt instructs the model to answer from visual evidence only.
        system_msg = (
            "You are a visual safety verification model. "
            "Answer the question using ONLY the provided image(s). "
            "Previous frame information is provided for reference only. "
            "Do NOT infer events or temporal causality from auxiliary data."
        )

        # Add a focus hint when a cropped ROI is also supplied.
        focus_prefix = ""
        if roi_crops:
            focus_prefix = (
                "A cropped close-up of the detected target is provided alongside the full scene. "
                "Use the cropped image for detailed assessment.\n\n"
            )

        # Append OD-hit supplement text only when a crop is available.
        supplement_text = ""
        if od_hit_supplement and roi_crops:
            supplement_text = f"\n{od_hit_supplement}"

        user_text = (
            f"Answer with 'Yes' or 'No' only. Do not explain.\n\n"
            f"{focus_prefix}"
            f"{context_text}"
            f"Question: {query}{supplement_text}"
        )

        # Assemble the user content: full frame first, then ROI crops, then text.
        user_content = [
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        if roi_crops:
            for crop in roi_crops:
                _, crop_buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                crop_b64 = base64.b64encode(crop_buf).decode('utf-8')
                crop_url = f"data:image/jpeg;base64,{crop_b64}"
                user_content.append(
                    {"type": "image_url", "image_url": {"url": crop_url}}
                )
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        # Send the request and retain transport status separately from parsing.
        provider_result = await self.provider.generate_detailed(messages)
        response_text = provider_result["response_text"]
        latency = provider_result["latency_sec"]

        # Normalize the response to Yes / No; anything else is an Error.
        answer = "Error"
        parser_status = "error"
        cleaned = response_text.strip().lower()
        if cleaned.startswith("yes"):
            answer = "Yes"
            parser_status = "yes"
        elif cleaned.startswith("no"):
            answer = "No"
            parser_status = "no"

        return {
            "answer": answer,
            "raw_response": response_text.strip(),
            "latency_sec": round(latency, 4),
            "parser_status": parser_status,
            "binary_evidence": 1 if answer == "Yes" else 0,
            "parser_failure": parser_status == "error",
            "api_success": provider_result["api_success"],
            "http_status": provider_result["http_status"],
            "api_error": provider_result["api_error"],
            "request_seed": provider_result["request_seed"],
        }

    def _build_context_text(self, tool_context: dict) -> str:
        """ Formats CV tool outputs into a structured reference text block for the VLM.

        The text is clearly labelled as auxiliary/reference data so the model
        knows to deprioritize it in favour of direct visual observation:
          - Pose analysis (body angle, keypoint visibility, arm/leg state)
          - Spatial detections (tracker IDs and labels)
          - Depth distance estimates
          - Previous frame detections (temporal reference only)
        """
        lines = [
            "[Auxiliary Reference - for reference only, may be inaccurate]",
            "The following are automated CV tool outputs. "
            "If they conflict with what you see in the image, trust your visual observation.",
        ]

        # Append pose analysis for each tracked person.
        for pa in tool_context.get("posture_analysis", []):
            tid = pa.get("tracker_id", "?")
            angle = pa.get("numeric_angle", "N/A")
            desc = pa.get("angle_description", "")
            hint = pa.get("system_posture_hint", "")
            head = pa.get("head_position", "")
            kp_vis = pa.get("keypoint_visibility", "")
            arm = pa.get("arm_state", "")
            leg = pa.get("leg_state", "")
            aspect = pa.get("body_aspect_ratio", "")
            lines.append(f"  Person (tracker {tid}): angle={angle}, {desc}, posture_hint={hint}")
            if kp_vis:
                lines.append(f"    keypoints={kp_vis}, head={head}, arms=[{arm}], legs=[{leg}], aspect_ratio={aspect}")

        # Append the list of currently detected/tracked objects.
        detections = tool_context.get("spatial_detections", [])
        if detections:
            labels = [f"{d.get('label', '?')}(id:{d.get('id', '?')})" for d in detections]
            lines.append(f"  Detected objects: {', '.join(labels)}")

        # Append depth estimates; higher value = closer for non-metric models.
        depth_lines = tool_context.get("depth_distance_analysis")
        if depth_lines:
            lines.append("  (Note: higher depth value = closer to camera)")
            for dl in depth_lines:
                lines.append(f"  {dl}")

        # Append previous-frame detections as temporal context (reference only).
        prev = tool_context.get("previous_frame")
        if prev:
            lines.append("")
            lines.append("[Previous Frame Summary - reference only]")
            lines.append("Do NOT infer events or temporal causality from this.")

            # List each tracked entity's last known position and size.
            prev_dets = prev.get("detections", [])
            if prev_dets:
                lines.append("Detected entities:")
                for d in prev_dets:
                    center = d.get("center", [])
                    size = d.get("size", [])
                    center_str = f"center=({center[0]},{center[1]})" if center else ""
                    size_str = f"size=({size[0]},{size[1]})" if size else ""
                    lines.append(f"  {d['label']}(id:{d['tracker_id']}): present, {center_str}, {size_str}")

        return "\n".join(lines) + "\n\n"
