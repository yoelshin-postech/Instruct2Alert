# modules/json_csv_parser.py
"""
Converts a planner workflow JSON into a step-by-step pipeline DataFrame.

Each alert rule in the workflow produces two pipeline steps:
  1. Metadata — evaluates OD conditions and rule parameters (distance, duration).
  2. VLM      — runs the verification prompt against the current frame.

Optional CV module steps (Pose, Depth) are inserted after the Tracker step
only when listed in `additional_modules`.

Output columns:
  step_id    : execution order (1-indexed)
  module     : OD | Tracker | Pose | Depth | Metadata | VLM
  targets    : comma-separated target label(s)
  condition  : condition string or VLM query text
  output_key : key used to store the step result in the execution context
"""
import pandas as pd

def json_to_pipeline_csv(json_config):
    steps = []
    step_id = 1

    # Extract target names; "main" is resolved to its real label.
    main_real_name = json_config['targets']['main']
    other_targets = json_config['targets']['other_targets']
    all_targets = [main_real_name] + other_targets

    # Step 1: Object Detection — detect all targets in the scene.
    steps.append({"step_id": step_id, "module": "OD", "targets": ",".join(all_targets), "condition": "detect_all", "output_key": "bbox_list"})
    step_id += 1

    # [Step 2] Tracker (Default)
    steps.append({"step_id": step_id, "module": "Tracker", "targets": main_real_name, "condition": "track_main", "output_key": "track_id"})
    step_id += 1

    # Check which optional CV modules (Pose, Depth) are required.
    additional_mods = json_config.get("additional_modules", [])

    # 3-1. Pose Module Check
    if "Pose" in additional_mods:
        steps.append({"step_id": step_id,  "module": "Pose",  "targets": main_real_name, "condition": "extract_keypoints", "output_key": "pose_data"})
        step_id += 1

    # 3-2. Depth Module Check
    if "Depth" in additional_mods:
        steps.append({"step_id": step_id,  "module": "Depth",  "targets": ",".join(all_targets), "condition": "estimate_distance", "output_key": "depth_map"})
        step_id += 1

    # [Step 4 & 5] Alert Rules
    # One Metadata step + one VLM step per alert rule.
    for idx, rule in enumerate(json_config['alert_rules']):
        real_target_pair = []
        for t in rule['target_pair']:
            if t == "main": real_target_pair.append(main_real_name)
            elif t is not None: real_target_pair.append(t)

        target_str = ",".join(real_target_pair)

        params = rule.get('parameters', {})
        duration_sec = params.get('duration_sec', 0.0)
        distance_m = params.get('distance_m', None)
        base_condition = rule['condition']
        param_strings = []
        if duration_sec > 0:
            param_strings.append(f"duration={duration_sec}")
        if distance_m is not None:
            param_strings.append(f"dist={distance_m}")
        if param_strings:
            full_condition = f"{base_condition}|{'&'.join(param_strings)}"
        else:
            full_condition = base_condition

        # Metadata step: evaluates OD conditions and rule parameters.
        steps.append({
            "step_id": step_id,
            "module": "Metadata",
            "targets": target_str,
            "condition": full_condition,
            "output_key": f"rule_metadata_{idx}"
        })
        step_id += 1

        # VLM step: runs the verification prompt against the current frame.
        vp = rule.get('verification_prompt', {})
        vlm_condition = vp.get('query', rule.get('description', ''))
        steps.append({
            "step_id": step_id,
            "module": "VLM",
            "targets": target_str,
            "condition": vlm_condition,
            "output_key": f"vlm_result_{idx}"
        })
        step_id += 1

    return pd.DataFrame(steps)
