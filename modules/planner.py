# modules/planner.py
import json
import re
from typing import List, Optional
from .prompt_templates import PLANNER_SYSTEM_PROMPT
from .workflow_contract import validate_instruction_language, validate_workflow_contract


class SafetyPlanner:
    def __init__(self, provider):
        self.provider = provider
        self.last_raw_response = ""
        self.last_latency_sec = None
        self.last_contract_issues = []

    async def create_plan(self, user_query: str, image_urls: Optional[List[str]] = None) -> dict:
        """
        Calls the VLM planner to generate a structured workflow JSON from a safety scenario description.

        Args:
            user_query: Natural-language description of the safety scenario to monitor.
            image_urls: Optional list of base64 data URLs (e.g., first video frame)
                        to give the planner visual context of the scene.
        """
        instruction_issues = validate_instruction_language(user_query)
        if instruction_issues:
            self.last_raw_response = ""
            self.last_latency_sec = 0.0
            self.last_contract_issues = instruction_issues
            return {
                "error": "Unsupported instruction language",
                "stage": "instruction_validation",
                "issues": instruction_issues,
            }

        # Build the multimodal message content list.
        user_content = []

        # Prepend image frames so the model sees visuals before the text query.
        if image_urls:
            for idx, img_url in enumerate(image_urls):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
            print(f"[Planner] Processing {len(image_urls)} image(s)...")

        # Append the text query as the final content item.
        user_content.append({"type": "text", "text": user_query})

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # Call the VLM with the assembled messages.
        print(f"[Planner] VLM Reasoning: '{user_query[:50]}...'")
        response_text, vlm_latency = await self.provider.generate(messages=messages)
        self.last_raw_response = response_text
        self.last_latency_sec = vlm_latency

        # Extract structured JSON from the LLM response.
        plan_json = self._parse_json_from_llm(response_text)
        if "error" not in plan_json:
            contract_issues = validate_workflow_contract(plan_json)
            self.last_contract_issues = contract_issues
            if contract_issues:
                return {
                    "error": "Workflow contract validation failed",
                    "stage": "workflow_contract_validation",
                    "issues": contract_issues,
                    "raw": response_text,
                }

        # Embed latency metrics into the plan for downstream reporting.
        if "error" not in plan_json:
            plan_json['metrics'] = {
                'vlm_latency_sec': round(vlm_latency, 4),
                'num_images': len(image_urls) if image_urls else 0
            }
        return plan_json

    def _parse_json_from_llm(self, text: str) -> dict:
        try:
            if not text:
                raise ValueError("Empty response text")

            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                clean_text = match.group(1)
            else:
                clean_text = text.strip()
            return json.loads(clean_text)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Planner] JSON Parsing Failed. Error: {e}")
            print(f"[Planner] Raw output:\n{text}")
            return {"error": "Failed to parse plan", "raw": text}
