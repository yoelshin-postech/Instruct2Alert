# run_e2e_pipeline.py
import asyncio
import argparse
import base64
import cv2
import json
import os
from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from modules.qwen_provider import QwenVLProvider
from modules.planner import SafetyPlanner
from modules.json_csv_parser import json_to_pipeline_csv
from modules.executor import PipelineExecutor
from modules.evaluation import (
    compute_advanced_metrics,
    compute_confusion_metrics,
    evaluate_run,
    format_summary,
    is_in_gt_interval,
    parse_gt_intervals,
)
from modules.workflow_contract import workflow_execution_semantics
from modules.run_audit import (
    append_jsonl,
    apply_run_seed,
    runtime_environment,
    snapshot_core_files,
    summarize_latencies,
    utc_now,
    write_csv,
    write_json,
)


def fill_metric_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces blank numeric metric cells with 0 for cleaner CSV/XLSX summaries."""
    if df is None or df.empty:
        return df

    out = df.copy()
    metric_tokens = (
        "precision", "recall", "f1", "accuracy", "fpr", "ttd",
        "fa_", "segment", "tp", "tn", "fp", "fn", "frames",
        "latency", "calls", "count", "alerts", "duration",
    )
    metric_cols = [
        c for c in out.columns
        if any(token in str(c).lower() for token in metric_tokens)
    ]
    if metric_cols:
        for col in metric_cols:
            try:
                out[col] = pd.to_numeric(out[col]).fillna(0)
            except (TypeError, ValueError):
                out[col] = out[col].fillna(0)
    return out


RUN_SUMMARY_METRIC_COLUMNS = [
    "planner_latency_sec", "executor_latency_sec", "frame_latency_sec",
    "pipeline_steps", "total_vlm_calls", "vlm_yes_count", "total_alerts",
    "gt_precision", "gt_recall", "gt_f1", "gt_accuracy",
    "gt_tp", "gt_tn", "gt_fp", "gt_fn",
    "gt_alert_precision", "gt_alert_recall", "gt_alert_f1", "gt_alert_accuracy",
    "gt_alert_tp", "gt_alert_tn", "gt_alert_fp", "gt_alert_fn",
    "gt_fa_episodes", "gt_micro_fpr", "gt_segment_recall", "gt_ttd_p90_s",
    "gt_alert_fa_episodes", "gt_alert_micro_fpr",
    "gt_alert_segment_recall", "gt_alert_ttd_p90_s",
]


def normalize_run_summary_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Ensures run_summary metric columns exist and are numeric 0 instead of blanks."""
    if summary_df is None or summary_df.empty:
        return summary_df

    out = summary_df.copy()
    for col in RUN_SUMMARY_METRIC_COLUMNS:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return fill_metric_blanks(out)


def compute_macro_metrics_from_summary(summary_df):
    """Computes scenario-level macro averages from the per-scenario summary table.

    Blank or failed scenario metrics are treated as 0 so the denominator remains
    the full scenario count reported in run_summary.xlsx.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy()
    metric_df = fill_metric_blanks(df)
    if "status" in df.columns:
        failed_count = int((df["status"] != "OK").sum())
    else:
        failed_count = 0

    def mean_col(col):
        if col not in metric_df.columns:
            return 0.0
        return round(float(pd.to_numeric(metric_df[col], errors="coerce").fillna(0).mean()), 4)

    def zero_tp_count(tp_col, fn_col):
        if tp_col not in metric_df.columns or fn_col not in metric_df.columns:
            return 0
        tp = pd.to_numeric(metric_df[tp_col], errors="coerce").fillna(0)
        fn = pd.to_numeric(metric_df[fn_col], errors="coerce").fillna(0)
        return int(((tp == 0) & ((tp + fn) > 0)).sum())

    rows = [
        {
            "Method": "Grounded Verify",
            "Macro_Precision": mean_col("gt_precision"),
            "Macro_Recall": mean_col("gt_recall"),
            "Macro_F1": mean_col("gt_f1"),
            "Macro_Accuracy": mean_col("gt_accuracy"),
            "Macro_FA_Episodes": mean_col("gt_fa_episodes"),
            "Macro_Micro_FPR": mean_col("gt_micro_fpr"),
            "Macro_Segment_Recall": mean_col("gt_segment_recall"),
            "Macro_TTD_P90_s": mean_col("gt_ttd_p90_s"),
            "Scenario_Count": len(metric_df),
            "Failed_Count": failed_count,
            "Zero_TP_Scenarios": zero_tp_count("gt_tp", "gt_fn"),
        },
        {
            "Method": "Full (Ours)",
            "Macro_Precision": mean_col("gt_alert_precision"),
            "Macro_Recall": mean_col("gt_alert_recall"),
            "Macro_F1": mean_col("gt_alert_f1"),
            "Macro_Accuracy": mean_col("gt_alert_accuracy"),
            "Macro_FA_Episodes": mean_col("gt_alert_fa_episodes"),
            "Macro_Micro_FPR": mean_col("gt_alert_micro_fpr"),
            "Macro_Segment_Recall": mean_col("gt_alert_segment_recall"),
            "Macro_TTD_P90_s": mean_col("gt_alert_ttd_p90_s"),
            "Scenario_Count": len(metric_df),
            "Failed_Count": failed_count,
            "Zero_TP_Scenarios": zero_tp_count("gt_alert_tp", "gt_alert_fn"),
        },
    ]
    return pd.DataFrame(rows)


def print_running_macro_summary(results_summary, current_idx, total_count):
    """Prints compact scenario-level macro metrics accumulated so far."""
    if not results_summary:
        return

    running_df = normalize_run_summary_metrics(pd.DataFrame(results_summary))
    macro_df = compute_macro_metrics_from_summary(running_df)
    if macro_df is None or macro_df.empty:
        return

    print("\n" + "-" * 95)
    print(f"[E2E] RUNNING MACRO SUMMARY ({len(running_df)}/{total_count} scenarios processed)")
    for _, row in macro_df.iterrows():
        print(
            f"[E2E] {row['Method']}: "
            f"P={row['Macro_Precision']:.4f} "
            f"R={row['Macro_Recall']:.4f} "
            f"F1={row['Macro_F1']:.4f} "
            f"Acc={row['Macro_Accuracy']:.4f} "
            f"FA={row['Macro_FA_Episodes']:.4f} "
            f"µFPR={row['Macro_Micro_FPR']:.4f} "
            f"SegRec={row['Macro_Segment_Recall']:.4f} "
            f"TTD_P90={row['Macro_TTD_P90_s']:.4f}s "
            f"ZeroTP={int(row['Zero_TP_Scenarios'])}"
        )
    print("-" * 95 + "\n")


def replay_temporal_controller(records, duration_sec, long_gap_threshold_sec=2.0,
                               include_off_tolerance=False):
    """Replay a temporal controller and expose its per-frame state trace.

    ``include_off_tolerance=False`` implements Workflow + Onset. ``True``
    additionally applies the Full controller's fixed one-second clearing rule.
    Long gaps reset evidence streaks but do not fabricate an alert transition.
    """
    trace = []
    yes_start_time = None
    no_start_time = None
    previous_time = None
    alert_active = False

    for rec in records:
        pred = rec["Inference_Result"]
        video_time = float(rec["Video_Time_sec"])
        delta_t_sec = 0.0 if previous_time is None else video_time - previous_time
        long_gap = delta_t_sec > long_gap_threshold_sec
        if long_gap:
            yes_start_time = None
            no_start_time = None

        if pred == "Yes":
            no_start_time = None
            if yes_start_time is None:
                yes_start_time = video_time
            positive_streak = video_time - yes_start_time
            negative_streak = 0.0
        else:
            yes_start_time = None
            if no_start_time is None:
                no_start_time = video_time
            positive_streak = 0.0
            negative_streak = video_time - no_start_time

        if duration_sec <= 0:
            alert_active = pred == "Yes"
        elif pred == "Yes" and positive_streak >= duration_sec:
            alert_active = True
        elif include_off_tolerance and pred != "Yes" and alert_active and negative_streak >= 1.0:
            alert_active = False
        elif not include_off_tolerance:
            alert_active = False

        trace.append({
            "replayed_alert": "Yes" if alert_active else "No",
            "positive_streak_sec": round(positive_streak, 6),
            "negative_streak_sec": round(negative_streak, 6),
            "final_alert_state": bool(alert_active),
            "long_gap_detected": long_gap,
            "long_gap_action": "reset_streaks" if long_gap else "none",
        })
        previous_time = video_time
    return trace


def normalize_frame_schedule(frame_schedule):
    """Return a sorted, de-duplicated canonical frame schedule.

    The executor owns frame identity. Variants must consume its persisted
    schedule instead of recalculating timestamps from decoder state.
    """
    normalized = {}
    for row in frame_schedule:
        frame_index = int(float(row.get("frame_index", row.get("Frame_Index"))))
        timestamp = float(row.get("video_timestamp_sec", row.get("Video_Time_sec")))
        key = (frame_index, round(timestamp, 6))
        normalized[key] = {
            "frame_index": frame_index,
            "video_timestamp_sec": timestamp,
            "delta_t_sec": float(row.get("delta_t_sec", row.get("Delta_t_sec", 0.0))),
        }
    return sorted(normalized.values(), key=lambda row: (row["video_timestamp_sec"], row["frame_index"]))


def pool_rule_records(records, prediction_field):
    """Pool rule-local frame evidence into a scenario-level any-positive stream."""
    grouped = {}
    for record in records:
        frame_index = int(float(record["Frame_Index"]))
        timestamp = round(float(record["Video_Time_sec"]), 6)
        grouped.setdefault((frame_index, timestamp), []).append(record)

    pooled = []
    for (_, _), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        group = sorted(
            group,
            key=lambda row: int(row.get("Rule_Index", 0))
            if row.get("Rule_Index") not in (None, "", "pooled") else -1,
        )
        base = dict(group[0])
        positive = any(str(row.get(prediction_field, "No")) == "Yes" for row in group)
        base[prediction_field] = "Yes" if positive else "No"
        base["Binary_Evidence"] = 1 if positive else 0
        base["Rule_Index"] = "pooled"
        base["Pooled_Rule_Count"] = len(group)
        base["Pooled_Rule_Indices"] = json.dumps([
            (int(row.get("Rule_Index", 0))
             if row.get("Rule_Index") not in (None, "", "pooled") else None)
            for row in group
        ])
        pooled.append(base)
    return pooled


async def run_vanilla_experiment(
    video_path: str,
    frame_schedule: list,
    scenario_id,
    video_name: str,
    query_text: str,
    gt_intervals: list,
    provider,
    output_dir: str,
    label: str = "Vanilla",
    output_suffix: str = "Vanilla_result",
    run_seed: int = None,
    variant: str = "direct_prompt",
    long_gap_threshold_sec: float = 2.0,
    rule_index=None,
    persist: bool = True,
):
    """
    Runs a vanilla VLM experiment: sends each frame directly to the VLM with a
    single prompt and records per-frame Yes/No answers against GT labels.
    No planner, no OD, no tracker — pure image-to-text inference.
    """
    vanilla_provider = QwenVLProvider(
        server_url=provider.server_url,
        model_name=provider.model_name,
        temperature=0.1,
        max_tokens=8,
        timeout=provider.timeout,
        seed=provider.seed,
    )

    prompt = (
        f"{query_text} "
        f"Answer only 'Yes' or 'No'."
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Vanilla] WARNING: Cannot open video: {video_path}")
        return None

    canonical_schedule = normalize_frame_schedule(frame_schedule)
    vanilla_records = []
    total = len(canonical_schedule)
    previous_video_time_sec = None

    print(f"\n[{label}] Running vanilla experiment on {total} frames...")

    for idx, sample in enumerate(canonical_schedule):
        frame_index = sample["frame_index"]
        # Seek to (frame_index - 1) because cap.read() advances the position by one.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ret, frame = cap.read()
        if not ret:
            continue

        video_time_sec = sample["video_timestamp_sec"]

        # Frame → base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_b64}"

        # Build the single-turn user message with image + text prompt.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        provider_result = await vanilla_provider.generate_detailed(messages)
        response_text = provider_result["response_text"]
        vlm_latency = provider_result["latency_sec"]

        # Preserve parser failures while exporting them as negative evidence,
        # matching the executor's fixed parser policy.
        cleaned = response_text.strip().lower() if response_text else ""
        if cleaned.startswith("yes"):
            inference_result = "Yes"
            parser_status = "yes"
        elif cleaned.startswith("no"):
            inference_result = "No"
            parser_status = "no"
        else:
            inference_result = "No"
            parser_status = "error"

        gt_result = "Yes" if is_in_gt_interval(frame_index, gt_intervals) else "No"
        delta_t_sec = (0.0 if previous_video_time_sec is None
                       else video_time_sec - previous_video_time_sec)
        previous_video_time_sec = video_time_sec
        long_gap_detected = delta_t_sec > long_gap_threshold_sec

        vanilla_records.append({
            "Run_Seed": run_seed,
            "Variant": variant,
            "Scenario_ID": scenario_id,
            "Video_name": video_name,
            "Query": query_text,
            "Prompt": prompt,
            "Frame_Index": frame_index,
            "Video_Time_sec": round(video_time_sec, 6),
            "Delta_t_sec": round(delta_t_sec, 6),
            "Long_Gap_Detected": long_gap_detected,
            "Long_Gap_Action": "reset_streaks" if long_gap_detected else "none",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "VLM_Latency_sec": round(vlm_latency, 4),
            "Raw_Response": response_text.strip(),
            "Parsed_Answer": "Yes" if parser_status == "yes" else ("No" if parser_status == "no" else "Error"),
            "Parser_Status": parser_status,
            "Parser_Failure": parser_status == "error",
            "Binary_Evidence": 1 if inference_result == "Yes" else 0,
            "API_Success": provider_result["api_success"],
            "API_Error": provider_result["api_error"],
            "Request_Seed": provider_result["request_seed"],
            "Inference_Result": inference_result,
            "GT_Inference_Result": gt_result,
            "Ground_Truth_State": gt_result,
            "Rule_Index": rule_index,
        })

        print(f"\r  [{label}] [{idx+1}/{total}] frame={frame_index} "
              f"result={inference_result} ({vlm_latency:.2f}s)", end="", flush=True)

    cap.release()
    print()

    if not vanilla_records:
        return None, []

    # Persist per-frame results to CSV.
    if persist:
        os.makedirs(output_dir, exist_ok=True)
        vanilla_df = pd.DataFrame(vanilla_records)
        vanilla_csv_path = os.path.join(output_dir, f"{scenario_id}_{output_suffix}.csv")
        vanilla_df.to_csv(vanilla_csv_path, index=False)
        print(f"[{label}] Results saved: {vanilla_csv_path}")

    # GT Confusion Matrix
    pred_list = [rec["Inference_Result"] for rec in vanilla_records]
    gt_list = [rec["GT_Inference_Result"] for rec in vanilla_records]
    gt_metrics = compute_confusion_metrics(pred_list, gt_list)

    # Append average latency to the confusion-matrix dict for convenience.
    avg_latency = sum(rec["VLM_Latency_sec"] for rec in vanilla_records) / len(vanilla_records)
    gt_metrics["avg_latency_sec"] = round(avg_latency, 4)

    print(f"[{label}] GT Eval: P={gt_metrics['precision']:.4f} "
          f"R={gt_metrics['recall']:.4f} F1={gt_metrics['f1']:.4f} "
          f"Acc={gt_metrics['accuracy']:.4f} "
          f"(TP={gt_metrics['tp']} TN={gt_metrics['tn']} "
          f"FP={gt_metrics['fp']} FN={gt_metrics['fn']}) "
          f"avg_latency={avg_latency:.4f}s")

    return gt_metrics, vanilla_records


# Project-relative path constants — all I/O is anchored to BASE_DIR.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "datasets", "safety_vlm_dataset.xlsx")
VIDEO_DIR = os.path.join(DATA_DIR, "input_video")
OUTPUT_DIR = os.path.join(DATA_DIR, "results")
IMAGE_OUTPUT_DIR = os.path.join(DATA_DIR, "images_result")
SCENARIO_LOOP_SLEEP_SEC = float(os.getenv("SCENARIO_LOOP_SLEEP_SEC", "2"))
PLANNER_TO_EXECUTOR_SLEEP_SEC = float(os.getenv("PLANNER_TO_EXECUTOR_SLEEP_SEC", "2"))
RUN_CONTEXT = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Instruct2Alert end to end.")
    parser.add_argument("--run-root", default=None,
                        help="Run root containing sibling results/ and images_result/ directories.")
    parser.add_argument("--run-seed", type=int, default=None,
                        help="Seed applied to Python, CV modules, vLLM requests, and recorded metadata.")
    parser.add_argument("--scenario-id", action="append", default=[],
                        help="Run only this scenario ID; may be supplied more than once.")
    parser.add_argument("--target-fps", type=float, default=10.0,
                        help="Configured maximum source-frame read rate.")
    parser.add_argument("--no-realtime-skip", action="store_true",
                        help="Disable wall-clock-aware frame skipping.")
    parser.add_argument("--long-gap-threshold-sec", type=float, default=2.0,
                        help="Observation-continuity reset threshold.")
    parser.add_argument("--internal-variants", action="store_true",
                        help="Run Direct Prompt and Workflow Verify on the saved Full frame schedule.")
    parser.add_argument("--variant", default="full",
                        help="Recorded variant label for this execution.")
    return parser.parse_args()


def configure_run(args):
    """Set run-local output roots and persist an immutable configuration snapshot."""
    global OUTPUT_DIR, IMAGE_OUTPUT_DIR, RUN_CONTEXT
    project_root = Path(BASE_DIR)
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        OUTPUT_DIR = str(run_root / "results")
        IMAGE_OUTPUT_DIR = str(run_root / "images_result")
    else:
        run_root = Path(DATA_DIR).resolve()
        OUTPUT_DIR = str(run_root / "results")
        IMAGE_OUTPUT_DIR = str(run_root / "images_result")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if args.run_seed is not None:
        seed_policy = apply_run_seed(args.run_seed)
    else:
        seed_policy = {"seed": None, "seeded": False}

    config_snapshot_dir = run_root / "config_snapshot"
    checksums = snapshot_core_files(project_root, config_snapshot_dir)
    fixed_execution_config = {
        "planner": {
            "model": "qwen-vl-8b",
            "temperature": 0.1,
            "max_tokens": 1024,
            "reference_frame_jpeg_quality": 85,
        },
        "verifier": {
            "model": "qwen-vl-8b",
            "temperature": 0.1,
            "max_tokens": 8,
            "parser": "leading_yes_no_else_negative_error",
            "full_frame_always_supplied": True,
            "roi_jpeg_quality": 85,
        },
        "person_detector": {
            "model": "yolo11l.pt",
            "score_threshold": 0.15,
            "nms_iou_threshold": 0.45,
        },
        "object_detector": {
            "model": "omlab/omdet-turbo-swin-tiny-hf",
            "score_threshold": 0.45,
            "nms_iou_threshold": 0.45,
        },
        "pose": {
            "model": "yolo11l-pose.pt",
            "score_threshold": 0.1,
            "keypoint_confidence_threshold": 0.15,
        },
        "depth": {"model": "depth-anything/Depth-Anything-V2-Small-hf"},
        "tracker": {
            "type": "Kalman IoU tracker with appearance-based re-identification",
            "max_tracker_age_frames": 45,
            "base_iou_match_threshold": 0.3,
            "unstable_iou_match_threshold": 0.1,
        },
        "temporal_controller": {
            "on_threshold": "workflow duration_sec",
            "off_threshold_sec": 1.0,
            "event_rule_behavior": "current binary verifier evidence",
            "long_gap_policy": "reset_streaks",
            "long_gap_threshold_sec": args.long_gap_threshold_sec,
        },
        "strict_interval_evaluation": {
            "temporal_iou_threshold": 0.3,
            "detection_latency_threshold_sec": 3.0,
            "matching": "maximum_cardinality_then_total_tiou",
        },
    }
    RUN_CONTEXT = {
        "run_root": str(run_root),
        "results_dir": OUTPUT_DIR,
        "images_result_dir": IMAGE_OUTPUT_DIR,
        "run_seed": args.run_seed,
        "variant": args.variant,
        "target_fps": args.target_fps,
        "realtime_skip": not args.no_realtime_skip,
        "long_gap_policy": "reset",
        "long_gap_threshold_sec": args.long_gap_threshold_sec,
        "internal_variants": args.internal_variants,
        "workflow_execution_semantics": workflow_execution_semantics(),
        "seed_policy": seed_policy,
        "fixed_execution_config": fixed_execution_config,
        "code_checksums": checksums,
        "runtime_environment": runtime_environment(),
        "created_at_utc": utc_now(),
        "status": "running",
    }
    write_json(run_root / "run_manifest.json", RUN_CONTEXT)
    return RUN_CONTEXT


def finalize_run_manifest(status: str, **extra):
    if not RUN_CONTEXT:
        return
    payload = dict(RUN_CONTEXT)
    payload.update(extra)
    payload["status"] = status
    payload["completed_at_utc"] = utc_now()
    write_json(Path(payload["run_root"]) / "run_manifest.json", payload)


def selected_dataset(dataset: pd.DataFrame, scenario_ids: list[str]) -> pd.DataFrame:
    if not scenario_ids:
        return dataset
    selected_ids = {normalize_scenario_id(scenario_id) for scenario_id in scenario_ids}
    normalized = dataset["Scenario_ID"].map(normalize_scenario_id)
    selected = dataset.loc[normalized.isin(selected_ids)].copy()
    missing = selected_ids - set(selected["Scenario_ID"].map(normalize_scenario_id))
    if missing:
        raise ValueError(f"Scenario IDs not found: {sorted(missing)}")
    return selected


def normalize_scenario_id(scenario_id) -> str:
    """ Converts a scenario ID (possibly a float from Excel) to a clean string for use in file names. """
    try:
        if pd.isna(scenario_id):
            return "unknown"
    except Exception:
        pass

    text = str(scenario_id).strip()
    try:
        num = float(text)
        if num.is_integer():
            text = str(int(num))
    except Exception:
        pass

    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def get_scenario_output_dirs(scenario_id) -> dict:
    """ Returns a dict of per-scenario output subdirectory paths, creating them if missing. """
    sid = normalize_scenario_id(scenario_id)
    scenario_root = os.path.join(OUTPUT_DIR, "scenarios", sid)
    dirs = {
        "sid": sid,
        "root": scenario_root,
        "planner": os.path.join(scenario_root, "planner"),
        "executor": os.path.join(scenario_root, "executor"),
        "vlm": os.path.join(scenario_root, "vlm"),
        "visualizations": os.path.join(scenario_root, "visualizations"),
        "vanilla": os.path.join(scenario_root, "vanilla"),
        "audit": os.path.join(scenario_root, "audit"),
    }
    # Create all sub-directories upfront (skip the "sid" string key).
    for key, path in dirs.items():
        if key == "sid":
            continue
        os.makedirs(path, exist_ok=True)
    return dirs


def get_summary_output_dir() -> str:
    """ Returns the summaries output directory, creating it if it does not exist. """
    summary_dir = os.path.join(OUTPUT_DIR, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    return summary_dir


def get_device() -> str:
    """ Selects cuda:1 if two or more GPUs are available, otherwise cuda:0 or cpu. """
    if not torch.cuda.is_available():
        print("[E2E] CUDA unavailable → CPU")
        return "cpu"
    n = torch.cuda.device_count()
    if n >= 2:
        dev = "cuda:1"
    else:
        dev = "cuda:0"
    print(f"[E2E] GPU count={n} → using {dev}")
    return dev


def extract_first_frame(video_path: str) -> str:
    """ Opens the video, reads the first frame, and returns it as a base64 JPEG data URL. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read first frame from: {video_path}")

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def save_result(scenario_id, plan_json, pipeline_df, planner_latency, output_dir):
    """
    Saves the workflow plan and execution results to an xlsx with multiple sheets:
    Sheet 1: Workflow JSON (key-value flattened rows)
    Sheet 2: Pipeline CSV (step-by-step DataFrame)
    Also dumps the raw workflow JSON to a .json file for easy inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{scenario_id}_JSON.xlsx"
    filepath = os.path.join(output_dir, filename)

    # Flatten the workflow JSON into key-value rows for Sheet 1.
    json_rows = []
    json_rows.append({"key": "scenario_name", "value": plan_json.get("scenario_name", "")})
    json_rows.append({"key": "targets.main", "value": plan_json.get("targets", {}).get("main", "")})
    json_rows.append({"key": "targets.other_targets", "value": str(plan_json.get("targets", {}).get("other_targets", []))})
    json_rows.append({"key": "additional_modules", "value": str(plan_json.get("additional_modules", []))})

    for idx, rule in enumerate(plan_json.get("alert_rules", [])):
        prefix = f"alert_rules[{idx}]"
        json_rows.append({"key": f"{prefix}.type", "value": rule.get("type", "")})
        json_rows.append({"key": f"{prefix}.target_pair", "value": str(rule.get("target_pair", []))})
        json_rows.append({"key": f"{prefix}.condition", "value": rule.get("condition", "")})
        params = rule.get("parameters", {})
        json_rows.append({"key": f"{prefix}.duration_sec", "value": params.get("duration_sec", 0.0)})
        json_rows.append({"key": f"{prefix}.distance_m", "value": params.get("distance_m", None)})
        vp = rule.get("verification_prompt", {})
        json_rows.append({"key": f"{prefix}.verification_query", "value": vp.get("query", rule.get("description", ""))})

    json_df = pd.DataFrame(json_rows)

    # Metrics
    metrics = plan_json.get("metrics", {})
    metrics_rows = [
        {"metric": "planner_vlm_latency_sec", "value": metrics.get("vlm_latency_sec", planner_latency)},
        {"metric": "num_images", "value": metrics.get("num_images", 0)},
        {"metric": "timestamp", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    # Collect per-evaluation-type rows for the GT_Evaluation sheet.
    gt_eval_rows = []
    gt_eval_vlm = plan_json.get("gt_evaluation")
    gt_eval_alert = plan_json.get("gt_alert_evaluation")
    gt_adv_vlm = plan_json.get("gt_advanced_evaluation")
    gt_adv_alert = plan_json.get("gt_alert_advanced_evaluation")
    if gt_eval_vlm:
        gt_eval_rows.append({"type": "VLM (per-frame)", **gt_eval_vlm})
    if gt_eval_alert:
        gt_eval_rows.append({"type": "Alert (duration)", **gt_eval_alert})
    if gt_adv_vlm:
        gt_eval_rows.append({"type": "VLM (advanced)", **gt_adv_vlm})
    if gt_adv_alert:
        gt_eval_rows.append({"type": "Alert (advanced)", **gt_adv_alert})
    gt_eval_df = pd.DataFrame(gt_eval_rows) if gt_eval_rows else None

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        json_df.to_excel(writer, sheet_name="WorkflowJSON", index=False)
        pipeline_df.to_excel(writer, sheet_name="PipelineCSV", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        if gt_eval_df is not None:
            gt_eval_df.to_excel(writer, sheet_name="GT_Evaluation", index=False)

    # Also dump the raw JSON alongside the xlsx for easy inspection.
    json_path = os.path.join(output_dir, f"{scenario_id}_JSON.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(plan_json, f, indent=4, ensure_ascii=False)

    return filepath


def summarize_sampling(rows: list[dict]) -> list[dict]:
    summaries = []
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scenario_id"], row["video_id"]), []).append(row)
    for (scenario_id, video_id), values in sorted(grouped.items()):
        gaps = np.asarray([float(value["delta_t_sec"]) for value in values[1:]], dtype=float)
        timestamps = [float(value["video_timestamp_sec"]) for value in values]
        span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        summaries.append({
            "run_seed": RUN_CONTEXT.get("run_seed"),
            "scenario_id": scenario_id,
            "video_id": video_id,
            "processed_frame_count": len(values),
            "realized_hz": round((len(values) - 1) / span, 6) if span > 0 else 0.0,
            "gap_mean_sec": round(float(np.mean(gaps)), 6) if gaps.size else 0.0,
            "gap_median_sec": round(float(np.median(gaps)), 6) if gaps.size else 0.0,
            "gap_p90_sec": round(float(np.percentile(gaps, 90)), 6) if gaps.size else 0.0,
            "gap_p99_sec": round(float(np.percentile(gaps, 99)), 6) if gaps.size else 0.0,
            "gap_max_sec": round(float(np.max(gaps)), 6) if gaps.size else 0.0,
            "stride_skipped_frames": sum(int(value["stride_skipped_frames"]) for value in values),
            "latency_skipped_frames": sum(int(value["latency_skipped_frames"]) for value in values),
            "tail_unprocessed_frames": max(
                0, int(values[-1].get("source_total_frames", values[-1]["frame_index"])) -
                int(values[-1]["frame_index"]),
            ),
            "total_unprocessed_frames": max(
                0, int(values[-1].get("source_total_frames", len(values))) - len(values),
            ),
        })
    return summaries


def write_variant_evidence(audit_dir: str, filename: str, records: list[dict],
                           variant: str, evidence_source: str, run_seed: int):
    """Persist raw VLM evidence once so controllers can be replayed without VLM calls."""
    rows = []
    for record in records or []:
        rows.append({
            "run_seed": run_seed,
            "variant": variant,
            "evidence_source": evidence_source,
            "scenario_id": record.get("Scenario_ID"),
            "video_id": record.get("Video_name"),
            "rule_index": record.get("Rule_Index"),
            "frame_index": record.get("Frame_Index"),
            "video_timestamp_sec": record.get("Video_Time_sec"),
            "delta_t_sec": record.get("Delta_t_sec", 0.0),
            "long_gap_detected": record.get("Long_Gap_Detected", False),
            "long_gap_action": record.get("Long_Gap_Action", "none"),
            "raw_response": record.get("Raw_Response"),
            "parser_status": record.get("Parser_Status"),
            "parser_failure": record.get("Parser_Failure", False),
            "binary_evidence": record.get("Binary_Evidence"),
            "api_success": record.get("API_Success"),
            "api_error": record.get("API_Error"),
            "request_seed": record.get("Request_Seed"),
            "ground_truth_state": record.get("Ground_Truth_State", record.get("GT_Inference_Result")),
            "grounding_attempted": record.get("Grounding_Attempted"),
            "grounding_success": record.get("Grounding_Success"),
            "full_frame_fallback": record.get("Full_Frame_Fallback"),
            "fallback_reason": record.get("Fallback_Reason"),
        })
    append_jsonl(Path(audit_dir) / filename, rows)


def write_controller_replay(audit_dir: str, records: list[dict], predictions: list[str],
                            controller: str, evidence_source: str, run_seed: int,
                            long_gap_threshold_sec: float, filename: str,
                            controller_trace: list[dict] | None = None):
    rows = []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        trace = controller_trace[index] if controller_trace and index < len(controller_trace) else {}
        rows.append({
            "run_seed": run_seed,
            "scenario_id": record.get("Scenario_ID"),
            "video_id": record.get("Video_name"),
            "rule_index": record.get("Rule_Index"),
            "frame_index": record.get("Frame_Index"),
            "video_timestamp_sec": record.get("Video_Time_sec"),
            "delta_t_sec": record.get("Delta_t_sec", 0.0),
            "long_gap_detected": record.get("Long_Gap_Detected", False),
            "long_gap_action": record.get("Long_Gap_Action", "none"),
            "controller": controller,
            "evidence_source": evidence_source,
            "long_gap_policy": "reset",
            "long_gap_threshold_sec": long_gap_threshold_sec,
            "binary_evidence": record.get("Binary_Evidence"),
            "positive_streak_sec": trace.get(
                "positive_streak_sec", record.get("Yes_Streak_sec")),
            "negative_streak_sec": trace.get(
                "negative_streak_sec", record.get("No_Streak_sec")),
            "final_alert_state": trace.get(
                "final_alert_state", str(prediction) == "Yes"),
            "replayed_alert": prediction,
        })
    write_csv(Path(audit_dir) / filename, rows)


def variant_latency_events(records: list[dict], component: str) -> list[dict]:
    """Convert per-frame variant VLM records into the shared latency schema."""
    return [
        {
            "run_seed": record.get("Run_Seed"),
            "variant": record.get("Variant"),
            "scenario_id": record.get("Scenario_ID"),
            "video_id": record.get("Video_name"),
            "component": component,
            "scope": "frame",
            "frame_index": record.get("Frame_Index"),
            "video_timestamp_sec": record.get("Video_Time_sec"),
            "rule_index": None,
            "latency_sec": record.get("VLM_Latency_sec"),
        }
        for record in records or []
    ]


async def main(args):
    configure_run(args)
    print(f"[E2E] Loading dataset: {DATASET_PATH}")
    dataset = pd.read_excel(DATASET_PATH)
    dataset = selected_dataset(dataset, args.scenario_id)
    print(f"[E2E] Dataset loaded: {len(dataset)} scenarios")

    # Initialize shared inference provider and planner; reused across all scenarios.
    device = get_device()
    provider = QwenVLProvider(
        server_url="http://localhost:8000/v1/chat/completions",
        model_name="qwen-vl-8b",
        temperature=0.1,
        max_tokens=1024,
        seed=args.run_seed,
    )
    planner = SafetyPlanner(provider)

    # Iterate over all scenarios in the dataset.
    num_test = len(dataset)
    results_summary = []
    all_latency_events = []
    all_schedule_rows = []

    for i in range(num_test):
        if SCENARIO_LOOP_SLEEP_SEC > 0:
            print(f"[E2E] Loop throttle: sleeping {SCENARIO_LOOP_SLEEP_SEC:.1f}s before next scenario...")
            await asyncio.sleep(SCENARIO_LOOP_SLEEP_SEC)

        row = dataset.iloc[i]
        scenario_id = row["Scenario_ID"]
        scenario_dirs = get_scenario_output_dirs(scenario_id)
        scenario_key = scenario_dirs["sid"]
        video_name = row["Video_name"]
        scenario_en = row["Scenario_EN"]
        video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
        gt_intervals = parse_gt_intervals(row.get("GT_intervals_frame", "[]"))

        print("=" * 60)
        print(f"[E2E] [{i+1}/{num_test}] Scenario_ID: {scenario_id}")
        print(f"[E2E] Video: {video_name}.mp4")
        print(f"[E2E] Query: {scenario_en}")
        print("=" * 60)

        # Skip scenarios where the video file is missing rather than crashing.
        if not os.path.exists(video_path):
            print(f"[E2E] WARNING: Video not found: {video_path}, skipping.")
            results_summary.append({
                "Scenario_ID": scenario_id,
                "status": "SKIP",
                "reason": "video_not_found"
            })
            print_running_macro_summary(results_summary, i + 1, num_test)
            continue

        # Extract a single representative frame for the planner's visual context.
        print(f"[E2E] Extracting first frame...")
        t0_frame = time.time()
        first_frame_url = extract_first_frame(video_path)
        frame_latency = time.time() - t0_frame
        print(f"[E2E] Frame extracted ({frame_latency:.3f}s)")

        # Run the planner to get a structured workflow JSON for this scenario.
        print(f"[Planner] Creating plan...")
        t0_plan = time.time()
        plan_json = await planner.create_plan(
            user_query=scenario_en,
            image_urls=[first_frame_url]
        )
        planner_latency = time.time() - t0_plan
        all_latency_events.append({
            "run_seed": args.run_seed,
            "variant": args.variant,
            "scenario_id": scenario_key,
            "component": "Planner",
            "scope": "scenario",
            "frame_index": None,
            "video_timestamp_sec": None,
            "rule_index": None,
            "latency_sec": round(planner_latency, 6),
        })

        if plan_json and "error" not in plan_json:
            print(f"[Planner] Plan created successfully ({planner_latency:.3f}s)")
            print(f"[Planner] WorkflowJSON:")
            print(json.dumps(plan_json, indent=2, ensure_ascii=False))

            # Convert the workflow JSON into a step-by-step pipeline DataFrame.
            print(f"[Parser] Converting to pipeline CSV...")
            pipeline_df = json_to_pipeline_csv(plan_json)
            print(f"[Parser] Pipeline ({len(pipeline_df)} steps):")
            print(pipeline_df.to_string(index=False))

            # Initialize the executor and run the full frame-by-frame pipeline.
            if PLANNER_TO_EXECUTOR_SLEEP_SEC > 0:
                print(f"[E2E] Planner-to-executor throttle: sleeping {PLANNER_TO_EXECUTOR_SLEEP_SEC:.1f}s...")
                await asyncio.sleep(PLANNER_TO_EXECUTOR_SLEEP_SEC)

            print(f"\n[Executor] Initializing pipeline...")
            t0_init = time.time()
            db_path = os.path.join(scenario_dirs["executor"], f"{scenario_key}_executor.json")
            executor = PipelineExecutor(
                workflow_json=plan_json,
                pipeline_df=pipeline_df,
                video_path=video_path,
                provider=provider,
                db_path=db_path,
                device=device,
                realtime_skip=not args.no_realtime_skip,
                target_fps=args.target_fps,
                scenario_id=scenario_id,
                image_output_dir=IMAGE_OUTPUT_DIR,
                audit_dir=scenario_dirs["audit"],
                run_seed=args.run_seed,
                variant=args.variant,
                long_gap_threshold_sec=args.long_gap_threshold_sec,
                video_id=video_name,
                gt_intervals=gt_intervals,
            )
            init_latency = time.time() - t0_init
            all_latency_events.append({
                "run_seed": args.run_seed,
                "variant": args.variant,
                "scenario_id": scenario_key,
                "component": "ExecutorInitialization",
                "scope": "scenario",
                "frame_index": None,
                "video_timestamp_sec": None,
                "rule_index": None,
                "latency_sec": round(init_latency, 6),
            })
            print(f"[Executor] Init done ({init_latency:.2f}s, model loading)")

            print(f"[Executor] Running pipeline on video...")
            t0_exec = time.time()
            try:
                exec_result = await executor.run()
            finally:
                executor.close()
            exec_latency = time.time() - t0_exec
            all_latency_events.extend(exec_result.get("latency_events", []))
            for schedule_row in exec_result.get("frame_schedule", []):
                all_schedule_rows.append({
                    "run_seed": args.run_seed,
                    "variant": args.variant,
                    "scenario_id": scenario_key,
                    "video_id": video_name,
                    **schedule_row,
                })
            print(f"[Executor] Done ({exec_latency:.2f}s, init={init_latency:.2f}s)")

            # Save VLM trigger/answer timeline plots (non-fatal if it fails).
            try:
                executor.save_vlm_visualizations(scenario_dirs["visualizations"], scenario_key)
            except Exception as e:
                print(f"[E2E] WARNING: VLM visualization failed: {e}")

            # Attach executor metrics to the plan JSON for logging in the output xlsx.
            db_summary = exec_result.get("db_summary", {})
            plan_json["executor_metrics"] = {
                "exec_latency_sec": round(exec_latency, 4),
                "db_summary": db_summary,
                "step_avg_times": exec_result.get("step_avg_times", {}),
                "step_latency_summary": exec_result.get("step_latency_summary", []),
                "run_seed": args.run_seed,
                "variant": args.variant,
                "long_gap_policy": "reset",
                "long_gap_threshold_sec": args.long_gap_threshold_sec,
                "frame_schedule_path": os.path.join(scenario_dirs["audit"], "frame_schedule.csv"),
                "audit_summary": exec_result.get("audit_summary", {}),
                # VLM call counters.
                "total_vlm_calls": exec_result.get("total_vlm_calls", 0),
                "total_vlm_yes_count": exec_result.get("total_vlm_yes_count", 0),
                # Alert edge counters.
                "total_alerts_triggered": exec_result.get("total_alerts_triggered", 0),
                "alert_intervals": exec_result.get("alert_intervals", []),
                # Per-step OD/VLM counts from the structured log.
                "OD_logic_triggered_count": db_summary.get("OD_logic_triggered_count", 0),
                "vlm_called_count": db_summary.get("vlm_called_count", 0),
                "vlm_yes_count": db_summary.get("vlm_yes_count", 0),
                "vlm_triggered_count": db_summary.get("vlm_triggered_count", 0),
            }

            # Process VLM call records: annotate with GT labels and compute per-scenario metrics.
            vlm_records = exec_result.get("vlm_call_records", [])
            if vlm_records:
                # Annotate each record with video metadata and GT label.
                for rec in vlm_records:
                    rec["Video_name"] = video_name
                    rec["Scenario_EN"] = scenario_en
                    rec["GT_Inference_Result"] = "Yes" if is_in_gt_interval(rec["Frame_Index"], gt_intervals) else "No"
                vlm_df = pd.DataFrame(vlm_records)
                # Reorder columns for consistent CSV output.
                col_order = [
                    "Run_Seed", "Variant", "Scenario_ID", "Video_name", "Scenario_EN", "Prompt",
                    "Frame_Index", "Video_Time_sec", "Delta_t_sec", "Long_Gap_Detected",
                    "Long_Gap_Action", "Timestamp", "VLM_Latency_sec", "Raw_Response",
                    "Parser_Status", "Parser_Failure", "Binary_Evidence", "Inference_Result",
                    "Alert_Result", "Yes_Streak_sec", "No_Streak_sec", "GT_Inference_Result",
                    "Ground_Truth_State",
                    "API_Success", "API_Error", "Request_Seed",
                    "Rule_Index", "Condition",
                    "OD_Detected", "Input_Mode", "OD_Hit_Supplement", "Grounding_Attempted",
                    "Grounding_Success", "Active_Main_Target_Detected", "Coasting_Only",
                    "ROI_Requested", "ROI_Count", "ROI_Track_IDs", "Grounding_Hint_Used",
                    "Full_Frame_Fallback", "Fallback_Reason",
                ]
                vlm_df = vlm_df[[c for c in col_order if c in vlm_df.columns]]
                vlm_csv_path = os.path.join(scenario_dirs["vlm"], f"{scenario_key}_VLM_results.csv")
                vlm_df.to_csv(vlm_csv_path, index=False)
                print(f"[E2E] VLM results saved: {vlm_csv_path}")

                # The Full pipeline has already produced the grounded evidence.
                # Preserve it once so Grounded Verify and Full controller states
                # can be compared without another VLM request.
                write_variant_evidence(
                    scenario_dirs["audit"], "grounded_verify_evidence.jsonl",
                    vlm_records, "grounded_verify", "grounded_verify", args.run_seed,
                )

                # Full execution evaluates every rule. Pool simultaneous
                # rule-local outputs with the benchmark's scenario-level
                # existential semantics before computing frame metrics.
                pooled_vlm_records = pool_rule_records(vlm_records, "Inference_Result")
                pooled_alert_records = pool_rule_records(vlm_records, "Alert_Result")
                write_csv(
                    Path(scenario_dirs["audit"]) / "grounded_verify_pooled.csv",
                    pooled_vlm_records,
                )
                write_csv(
                    Path(scenario_dirs["audit"]) / "full_controller_pooled.csv",
                    pooled_alert_records,
                )

                # Compute per-frame VLM confusion metrics against GT labels.
                pred_list = [rec["Inference_Result"] for rec in pooled_vlm_records]
                gt_list = [rec["GT_Inference_Result"] for rec in pooled_vlm_records]
                video_time_list = [float(rec["Video_Time_sec"]) for rec in pooled_vlm_records]
                gt_metrics_vlm = compute_confusion_metrics(pred_list, gt_list)
                plan_json["gt_evaluation"] = gt_metrics_vlm
                gt_adv_vlm = compute_advanced_metrics(pred_list, gt_list, video_time_list)
                plan_json["gt_advanced_evaluation"] = {
                    k: v for k, v in gt_adv_vlm.items() if not k.startswith("_")
                }
                print(f"[E2E] GT Eval (VLM):   P={gt_metrics_vlm['precision']:.4f} "
                      f"R={gt_metrics_vlm['recall']:.4f} F1={gt_metrics_vlm['f1']:.4f} "
                      f"Acc={gt_metrics_vlm['accuracy']:.4f} "
                      f"(TP={gt_metrics_vlm['tp']} TN={gt_metrics_vlm['tn']} "
                      f"FP={gt_metrics_vlm['fp']} FN={gt_metrics_vlm['fn']})")
                print(f"[E2E] GT Adv  (VLM):   FA={gt_adv_vlm['fa_episodes']} "
                      f"µFPR={gt_adv_vlm['micro_fpr']:.4f} "
                      f"SegRec={gt_adv_vlm['segment_recall'] if gt_adv_vlm['segment_recall'] is not None else 'N/A'} "
                      f"TTD_P90={gt_adv_vlm['ttd_p90_s'] if gt_adv_vlm['ttd_p90_s'] is not None else 'N/A'}s")

                # Compute the same metrics using the temporally-gated alert signal.
                alert_list = [rec["Alert_Result"] for rec in pooled_alert_records]
                write_controller_replay(
                    # Preserve rule-local replay rows. The
                    # pooled rows above are used only for scenario-level metrics.
                    scenario_dirs["audit"], vlm_records,
                    [rec["Alert_Result"] for rec in vlm_records],
                    "full", "grounded_verify", args.run_seed,
                    args.long_gap_threshold_sec, "full_controller_replay.csv",
                )
                gt_metrics_alert = compute_confusion_metrics(alert_list, gt_list)
                plan_json["gt_alert_evaluation"] = gt_metrics_alert
                gt_adv_alert = compute_advanced_metrics(alert_list, gt_list, video_time_list)
                plan_json["gt_alert_advanced_evaluation"] = {
                    k: v for k, v in gt_adv_alert.items() if not k.startswith("_")
                }
                print(f"[E2E] GT Eval (Alert): P={gt_metrics_alert['precision']:.4f} "
                      f"R={gt_metrics_alert['recall']:.4f} F1={gt_metrics_alert['f1']:.4f} "
                      f"Acc={gt_metrics_alert['accuracy']:.4f} "
                      f"(TP={gt_metrics_alert['tp']} TN={gt_metrics_alert['tn']} "
                      f"FP={gt_metrics_alert['fp']} FN={gt_metrics_alert['fn']})")
                print(f"[E2E] GT Adv  (Alert): FA={gt_adv_alert['fa_episodes']} "
                      f"µFPR={gt_adv_alert['micro_fpr']:.4f} "
                      f"SegRec={gt_adv_alert['segment_recall'] if gt_adv_alert['segment_recall'] is not None else 'N/A'} "
                      f"TTD_P90={gt_adv_alert['ttd_p90_s'] if gt_adv_alert['ttd_p90_s'] is not None else 'N/A'}s")

            # Persist the full workflow JSON and pipeline CSV to xlsx.
            result_path = save_result(
                scenario_id=scenario_key,
                plan_json=plan_json,
                pipeline_df=pipeline_df,
                planner_latency=planner_latency,
                output_dir=scenario_dirs["planner"],
            )
            print(f"[E2E] Result saved: {result_path}")

            gt_eval_vlm = plan_json.get("gt_evaluation", {})
            gt_eval_alert = plan_json.get("gt_alert_evaluation", {})
            gt_adv_vlm = plan_json.get("gt_advanced_evaluation", {})
            gt_adv_alert = plan_json.get("gt_alert_advanced_evaluation", {})
            results_summary.append({
                "Scenario_ID": scenario_id,
                "status": "OK",
                "planner_latency_sec": round(planner_latency, 4),
                "executor_latency_sec": round(exec_latency, 4),
                "frame_latency_sec": round(frame_latency, 4),
                "pipeline_steps": len(pipeline_df),
                # VLM raw results (per-frame).
                "total_vlm_calls": exec_result.get("total_vlm_calls", 0),
                "vlm_yes_count": db_summary.get("vlm_yes_count", 0),
                # Alert results (temporally gated).
                "total_alerts": exec_result.get("total_alerts_triggered", 0),
                "alert_intervals": str(exec_result.get("alert_intervals", [])),
                # VLM confusion metrics.
                "gt_precision": gt_eval_vlm.get("precision"),
                "gt_recall": gt_eval_vlm.get("recall"),
                "gt_f1": gt_eval_vlm.get("f1"),
                "gt_accuracy": gt_eval_vlm.get("accuracy"),
                "gt_tp": gt_eval_vlm.get("tp"),
                "gt_tn": gt_eval_vlm.get("tn"),
                "gt_fp": gt_eval_vlm.get("fp"),
                "gt_fn": gt_eval_vlm.get("fn"),
                # Alert confusion metrics.
                "gt_alert_precision": gt_eval_alert.get("precision"),
                "gt_alert_recall": gt_eval_alert.get("recall"),
                "gt_alert_f1": gt_eval_alert.get("f1"),
                "gt_alert_accuracy": gt_eval_alert.get("accuracy"),
                "gt_alert_tp": gt_eval_alert.get("tp"),
                "gt_alert_tn": gt_eval_alert.get("tn"),
                "gt_alert_fp": gt_eval_alert.get("fp"),
                "gt_alert_fn": gt_eval_alert.get("fn"),
                # Advanced VLM metrics.
                "gt_fa_episodes": gt_adv_vlm.get("fa_episodes"),
                "gt_micro_fpr": gt_adv_vlm.get("micro_fpr"),
                "gt_segment_recall": gt_adv_vlm.get("segment_recall"),
                "gt_ttd_p90_s": gt_adv_vlm.get("ttd_p90_s"),
                # Advanced alert metrics.
                "gt_alert_fa_episodes": gt_adv_alert.get("fa_episodes"),
                "gt_alert_micro_fpr": gt_adv_alert.get("micro_fpr"),
                "gt_alert_segment_recall": gt_adv_alert.get("segment_recall"),
                "gt_alert_ttd_p90_s": gt_adv_alert.get("ttd_p90_s"),
                "output_file": result_path
            })
            print_running_macro_summary(results_summary, i + 1, num_test)
        else:
            print(f"[Planner] FAILED to create plan ({planner_latency:.3f}s)")
            print(f"[Planner] Error: {plan_json}")
            write_json(
                Path(scenario_dirs["planner"]) / f"{scenario_key}_planner_error.json",
                {
                    "scenario_id": scenario_id,
                    "scenario_query": scenario_en,
                    "run_seed": args.run_seed,
                    "planner_latency_sec": round(planner_latency, 6),
                    "planner_error": plan_json,
                },
            )
            results_summary.append({
                "Scenario_ID": scenario_id,
                "status": "FAIL",
                "planner_latency_sec": round(planner_latency, 4),
                "reason": plan_json.get("error", "unknown")
            })
            print_running_macro_summary(results_summary, i + 1, num_test)

    # Print per-scenario run summary.
    print("\n" + "=" * 60)
    print("[E2E] SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(results_summary)
    summary_df = normalize_run_summary_metrics(summary_df)
    print(summary_df.to_string(index=False))

    # Save summary DataFrame to xlsx.
    summary_dir = get_summary_output_dir()
    summary_path = os.path.join(summary_dir, "run_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"[E2E] Summary saved: {summary_path}")


    # Direct/Workflow variants share the Full schedule.
    if args.internal_variants:
        print("\n" + "=" * 60)
        print("[Vanilla] Running internal variants")
        print("=" * 60)

        for i in range(num_test):
            row = dataset.iloc[i]
            scenario_id = row["Scenario_ID"]
            scenario_dirs = get_scenario_output_dirs(scenario_id)
            scenario_key = scenario_dirs["sid"]
            video_name = row["Video_name"]
            scenario_en = row["Scenario_EN"]
            video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
            gt_intervals = parse_gt_intervals(row.get("GT_intervals_frame", "[]"))

            # The Full executor owns the canonical schedule. Every internal
            # variant consumes this persisted frame/timestamp sequence rather
            # than recalculating decoder positions or wall-clock skips.
            schedule_path = Path(scenario_dirs["audit"]) / "frame_schedule.csv"
            if not schedule_path.exists():
                print(f"[Vanilla] SKIP {scenario_id}: frame_schedule.csv not found")
                continue
            frame_schedule = normalize_frame_schedule(
                pd.read_csv(schedule_path).to_dict("records"))
            if not frame_schedule:
                print(f"[Vanilla] SKIP {scenario_id}: empty frame schedule")
                continue

            # Load all compiled verification rules. Each is evaluated with the
            # same direct-VLM interface, then pooled any-positive per frame to
            # match the benchmark's scenario-level alert semantics.
            json_path = os.path.join(scenario_dirs["planner"], f"{scenario_key}_JSON.json")
            alert_rules = []
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    plan_data = json.load(f)
                alert_rules = plan_data.get("alert_rules", [])

            print(f"\n[Vanilla] [{i+1}/{num_test}] Scenario_ID: {scenario_id} "
                  f"({len(frame_schedule)} frames, {len(alert_rules)} rules)")

            # Run Workflow Verify once per compiled rule. Controller-only
            # variants reuse these frozen binary responses; they never issue a
            # second VLM request.
            vq_metrics = None
            vq_dur_metrics = None
            workflow_rule_records = []
            workflow_onset_rule_records = []
            for rule_idx, rule in enumerate(alert_rules):
                verification_query = rule.get("verification_prompt", {}).get("query", "")
                if not verification_query:
                    print(f"[Vanilla-VQ] Rule {rule_idx} skipped: missing verification query")
                    continue
                duration_sec = float(rule.get("parameters", {}).get("duration_sec", 0.0) or 0.0)
                print(f"[Vanilla-VQ] rule={rule_idx} duration={duration_sec:g}s | "
                      f"Query: {verification_query}")
                _, rule_records = await run_vanilla_experiment(
                    video_path=video_path,
                    frame_schedule=frame_schedule,
                    scenario_id=scenario_key,
                    video_name=video_name,
                    query_text=verification_query,
                    gt_intervals=gt_intervals,
                    provider=provider,
                    output_dir=scenario_dirs["vanilla"],
                    label=f"Vanilla-VQ r{rule_idx}",
                    output_suffix=f"Vanilla_VQ_rule_{rule_idx}_result",
                    run_seed=args.run_seed,
                    variant="workflow_verify",
                    long_gap_threshold_sec=args.long_gap_threshold_sec,
                    rule_index=rule_idx,
                    persist=False,
                )
                if not rule_records:
                    continue
                workflow_rule_records.extend(rule_records)
                onset_trace = replay_temporal_controller(
                    rule_records, duration_sec, args.long_gap_threshold_sec,
                    include_off_tolerance=False,
                )
                gated_pred = [row["replayed_alert"] for row in onset_trace]
                write_controller_replay(
                    scenario_dirs["audit"], rule_records, gated_pred,
                    "workflow_onset", "workflow_verify", args.run_seed,
                    args.long_gap_threshold_sec,
                    f"workflow_onset_rule_{rule_idx}_replay.csv",
                    controller_trace=onset_trace,
                )
                for record, prediction, trace_row in zip(rule_records, gated_pred, onset_trace):
                    onset_record = dict(record)
                    onset_record["Alert_Result"] = prediction
                    onset_record["Yes_Streak_sec"] = trace_row["positive_streak_sec"]
                    onset_record["No_Streak_sec"] = trace_row["negative_streak_sec"]
                    onset_record["Replay_Final_Alert_State"] = trace_row["final_alert_state"]
                    workflow_onset_rule_records.append(onset_record)

            if workflow_rule_records:
                write_variant_evidence(
                    scenario_dirs["audit"], "workflow_verify_evidence.jsonl",
                    workflow_rule_records, "workflow_verify", "workflow_verify", args.run_seed,
                )
                all_latency_events.extend(
                    variant_latency_events(workflow_rule_records, "VLMWorkflowVerify"))
                write_csv(
                    Path(scenario_dirs["vanilla"]) / f"{scenario_key}_Vanilla_VQ_rule_records.csv",
                    workflow_rule_records,
                )
                write_controller_replay(
                    scenario_dirs["audit"], workflow_onset_rule_records,
                    [record["Alert_Result"] for record in workflow_onset_rule_records],
                    "workflow_onset", "workflow_verify", args.run_seed,
                    args.long_gap_threshold_sec, "workflow_onset_replay.csv",
                )

                pooled_vq_records = pool_rule_records(
                    workflow_rule_records, "Inference_Result")
                pooled_onset_records = pool_rule_records(
                    workflow_onset_rule_records, "Alert_Result")
                write_csv(
                    Path(scenario_dirs["vanilla"]) / f"{scenario_key}_Vanilla_VQ_result.csv",
                    pooled_vq_records,
                )
                write_csv(
                    Path(scenario_dirs["audit"]) / "workflow_verify_pooled.csv",
                    pooled_vq_records,
                )
                write_csv(
                    Path(scenario_dirs["audit"]) / "workflow_onset_pooled.csv",
                    pooled_onset_records,
                )

                gt_list = [record["GT_Inference_Result"] for record in pooled_vq_records]
                vq_pred = [record["Inference_Result"] for record in pooled_vq_records]
                onset_pred = [record["Alert_Result"] for record in pooled_onset_records]
                vq_metrics = compute_confusion_metrics(vq_pred, gt_list)
                vq_dur_metrics = compute_confusion_metrics(onset_pred, gt_list)
                print(f"[Vanilla-VQ pooled] P={vq_metrics['precision']:.4f} "
                      f"R={vq_metrics['recall']:.4f} F1={vq_metrics['f1']:.4f} "
                      f"Acc={vq_metrics['accuracy']:.4f}")
                print(f"[Vanilla-VQ+Onset pooled] P={vq_dur_metrics['precision']:.4f} "
                      f"R={vq_dur_metrics['recall']:.4f} F1={vq_dur_metrics['f1']:.4f} "
                      f"Acc={vq_dur_metrics['accuracy']:.4f}")
            else:
                print(f"[Vanilla-VQ] SKIP: no executable verification rule found in JSON")

            print(f"[Vanilla-SE] Query: {scenario_en}")
            _, se_records = await run_vanilla_experiment(
                video_path=video_path,
                frame_schedule=frame_schedule,
                scenario_id=scenario_key,
                video_name=video_name,
                query_text=scenario_en,
                gt_intervals=gt_intervals,
                provider=provider,
                output_dir=scenario_dirs["vanilla"],
                label="Vanilla-SE",
                output_suffix="Vanilla_SE_result",
                run_seed=args.run_seed,
                variant="direct_prompt",
                long_gap_threshold_sec=args.long_gap_threshold_sec,
            )
            write_variant_evidence(
                scenario_dirs["audit"], "direct_prompt_evidence.jsonl",
                se_records, "direct_prompt", "direct_prompt", args.run_seed,
            )
            all_latency_events.extend(
                variant_latency_events(se_records, "VLMDirectPrompt"))

    telemetry_dir = Path(OUTPUT_DIR) / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(telemetry_dir / "latency_events.jsonl", all_latency_events)
    write_csv(telemetry_dir / "latency_events.csv", all_latency_events)
    write_csv(telemetry_dir / "latency_summary.csv", summarize_latencies(all_latency_events))
    write_csv(telemetry_dir / "frame_schedule.csv", all_schedule_rows)
    sampling_summary = summarize_sampling(all_schedule_rows)
    write_csv(telemetry_dir / "sampling_summary.csv", sampling_summary)

    # Evaluate every executed variant from the saved per-frame outputs. This is
    # a post-processing pass and issues no further VLM requests.
    try:
        summary_table = evaluate_run(RUN_CONTEXT["run_root"], DATASET_PATH)
        print("\n" + "=" * 95)
        print("[E2E] RESULTS")
        print("=" * 95)
        print(format_summary(summary_table))
        print(f"[E2E] Results saved: {Path(OUTPUT_DIR) / 'summary.csv'}, {Path(OUTPUT_DIR) / 'per_scenario.csv'}")
    except Exception as exc:
        # Keep the run artifacts even if evaluation fails; rerun scripts/evaluate.py.
        print(f"[E2E] WARNING: evaluation failed: {exc}")

    completed = int(sum(row.get("status") == "OK" for row in results_summary))
    finalize_run_manifest(
        "complete",
        requested_scenario_ids=[normalize_scenario_id(value) for value in args.scenario_id],
        processed_scenarios=completed,
        total_selected_scenarios=num_test,
        results_summary_path=os.path.join(OUTPUT_DIR, "summaries", "run_summary.xlsx"),
        telemetry_dir=str(telemetry_dir),
    )
    Path(RUN_CONTEXT["run_root"], "COMPLETE").touch()
    print(f"[E2E] Run manifest finalized: {Path(RUN_CONTEXT['run_root']) / 'run_manifest.json'}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
