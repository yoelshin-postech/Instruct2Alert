# modules/evaluation.py
"""
Run-level evaluation for Instruct2Alert.

Reads the per-frame outputs saved by a run and reports, for every variant that
was executed, the metrics of the main results table:

  P, R, F1, Acc, Macro FPR, TTD P90  scenario-macro over sampled timestamps
  FA Ep.                              false-alert episodes summed over scenarios
  SegRec                              ground-truth intervals with at least one
                                      positive sampled timestamp, pooled over scenarios
  R@tIoU0.3                           one-to-one interval matching at temporal IoU 0.3
                                      (modules/interval_metrics.py)

Scenarios that were processed but produced no output for a variant (e.g. a
planner failure) count as zero, so failures stay in the denominator.

Outputs, written under <run_root>/results/:
  summary.csv        one row per variant
  per_scenario.csv   one row per variant and scenario
  interval_metrics/  interval-level details, including Onset/Offset MAE and
                     DetRec@3s (intervals first under alert within 3 s,
                     pooled over ground-truth intervals)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.interval_metrics import evaluate_saved_run


# (display name, interval_metrics key, per-frame file under results/scenarios/<id>/, prediction column)
METHODS = [
    ("Direct Prompt", "direct_prompt", "vanilla/{sid}_Vanilla_SE_result.csv", "Inference_Result"),
    ("Workflow Verify", "workflow_verify", "audit/workflow_verify_pooled.csv", "Inference_Result"),
    ("Workflow + Onset", "workflow_onset", "audit/workflow_onset_pooled.csv", "Alert_Result"),
    ("Grounded Verify", "grounded_verify", "audit/grounded_verify_pooled.csv", "Inference_Result"),
    ("Full (Ours)", "full", "audit/full_controller_pooled.csv", "Alert_Result"),
]

SCENARIO_MACRO_METRICS = ["precision", "recall", "f1", "accuracy", "macro_fpr", "ttd_p90_s"]


# ================================================================
# Frame-level metrics
# ================================================================
def parse_gt_intervals(gt_str):
    """ Parses a JSON string of [start, end] frame index pairs into a list of intervals. """
    if pd.isna(gt_str) or not gt_str:
        return []
    try:
        intervals = json.loads(str(gt_str))
        return intervals if isinstance(intervals, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def is_in_gt_interval(frame_index, gt_intervals):
    """ Returns True if frame_index falls within any ground-truth event interval. """
    for interval in gt_intervals:
        if len(interval) >= 2 and interval[0] <= frame_index <= interval[1]:
            return True
    return False


def compute_confusion_metrics(pred_list, gt_list):
    """ Computes per-frame binary classification metrics (TP, TN, FP, FN, precision, recall, F1, accuracy). """
    tp = sum(1 for p, g in zip(pred_list, gt_list) if p == "Yes" and g == "Yes")
    tn = sum(1 for p, g in zip(pred_list, gt_list) if p == "No" and g == "No")
    fp = sum(1 for p, g in zip(pred_list, gt_list) if p == "Yes" and g == "No")
    fn = sum(1 for p, g in zip(pred_list, gt_list) if p == "No" and g == "Yes")
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def extract_segments(binary_list):
    """ Returns a list of (start, end) index tuples for contiguous True runs in binary_list. """
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(binary_list):
        if v and not in_seg:
            in_seg = True
            start = i
        elif not v and in_seg:
            in_seg = False
            segments.append((start, i - 1))
    if in_seg:
        segments.append((start, len(binary_list) - 1))
    return segments


def compute_advanced_metrics(pred_list, gt_list, video_time_list):
    """
    Computes segment-level and temporal metrics beyond per-frame confusion matrix:
    - FA_Episodes:    Number of false-alarm predicted-Yes segments (no GT overlap).
    - Micro_FPR:      Frame-level false positive rate.
    - Segment_Recall: Fraction of GT event segments that contained at least one TP.
    - TTD_P90_s:      90th-percentile time-to-detection within GT event segments.
    """
    n = min(len(pred_list), len(gt_list), len(video_time_list))
    pred = pred_list[:n]
    gt = gt_list[:n]
    times = [float(t) for t in video_time_list[:n]]

    pred_bin = [1 if str(x).strip() == "Yes" else 0 for x in pred]
    gt_bin = [1 if str(x).strip() == "Yes" else 0 for x in gt]

    # Count FA episodes: predicted-Yes segments with no overlap with any GT-Yes segment.
    pred_segments = extract_segments(pred_bin)
    gt_segments = extract_segments(gt_bin)
    fa_episodes = 0
    for ps, pe in pred_segments:
        has_overlap = any(not (pe < gs or ge < ps) for gs, ge in gt_segments)
        if not has_overlap:
            fa_episodes += 1

    tn = sum(1 for p, g in zip(pred_bin, gt_bin) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(pred_bin, gt_bin) if p == 1 and g == 0)
    micro_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    total_gt_segments = len(gt_segments)
    detected_gt_segments = 0
    ttds = []

    for s, e in gt_segments:
        first_tp_idx = None
        for j in range(s, e + 1):
            if pred_bin[j] == 1:
                first_tp_idx = j
                break
        if first_tp_idx is not None:
            detected_gt_segments += 1
            ttds.append(times[first_tp_idx] - times[s])

    segment_recall = (detected_gt_segments / total_gt_segments) if total_gt_segments > 0 else None
    ttd_p90_s = float(np.percentile(ttds, 90)) if ttds else None

    return {
        "fa_episodes": fa_episodes,
        "micro_fpr": round(micro_fpr, 4),
        "segment_recall": round(segment_recall, 4) if segment_recall is not None else None,
        "ttd_p90_s": round(ttd_p90_s, 3) if ttd_p90_s is not None else None,
    }


# ================================================================
# Run-level evaluation
# ================================================================
def _processed_scenarios(results_root: Path, dataset: pd.DataFrame) -> list:
    """Scenario IDs handled by the run, including failed ones."""
    run_summary = results_root / "summaries" / "run_summary.xlsx"
    if run_summary.exists():
        ids = pd.read_excel(run_summary)["Scenario_ID"]
        return [int(float(value)) for value in ids]
    return [int(sid) for sid in dataset["Scenario_ID"]
            if (results_root / "scenarios" / str(int(sid)) / "audit" / "frame_schedule.csv").exists()]


def _scenario_metrics(frames: pd.DataFrame, prediction_column: str, gt_intervals: list) -> dict:
    """Frame-level metrics for one variant on one scenario, plus pooled SegRec counts."""
    zero = {metric: 0.0 for metric in SCENARIO_MACRO_METRICS}
    zero.update(fa_episodes=0, gt_intervals=len(gt_intervals), detected_gt_intervals=0)
    if frames is None or frames.empty:
        return zero

    frame_index = frames["Frame_Index"].astype(int).tolist()
    pred = frames[prediction_column].astype(str).str.strip().tolist()
    gt = ["Yes" if is_in_gt_interval(f, gt_intervals) else "No" for f in frame_index]
    times = frames["Video_Time_sec"].astype(float).tolist()

    confusion = compute_confusion_metrics(pred, gt)
    advanced = compute_advanced_metrics(pred, gt, times)
    positive_frames = [f for f, p in zip(frame_index, pred) if p == "Yes"]
    detected = sum(any(start <= f <= end for f in positive_frames) for start, end in gt_intervals)
    return {
        "precision": confusion["precision"],
        "recall": confusion["recall"],
        "f1": confusion["f1"],
        "accuracy": confusion["accuracy"],
        "macro_fpr": advanced["micro_fpr"],
        "ttd_p90_s": advanced["ttd_p90_s"] or 0.0,
        "fa_episodes": advanced["fa_episodes"],
        "gt_intervals": len(gt_intervals),
        "detected_gt_intervals": detected,
    }


def evaluate_run(run_root, dataset_path) -> pd.DataFrame:
    """Evaluate every executed variant of a saved run and return the summary table."""
    run_root = Path(run_root)
    results_root = run_root / "results"
    dataset = pd.read_excel(dataset_path)
    rows_by_id = {int(row["Scenario_ID"]): row for _, row in dataset.iterrows()}
    scenario_ids = [sid for sid in _processed_scenarios(results_root, dataset) if sid in rows_by_id]

    per_scenario = []
    for name, _, pattern, column in METHODS:
        paths = {sid: results_root / "scenarios" / str(sid) / pattern.format(sid=sid) for sid in scenario_ids}
        if not any(path.exists() for path in paths.values()):
            continue  # variant not executed in this run
        for sid in scenario_ids:
            frames = pd.read_csv(paths[sid]) if paths[sid].exists() else None
            gt_intervals = parse_gt_intervals(rows_by_id[sid].get("GT_intervals_frame", "[]"))
            per_scenario.append({
                "method": name,
                "scenario_id": sid,
                "video_name": rows_by_id[sid]["Video_name"],
                "query_type": rows_by_id[sid].get("Query_Type"),
                **_scenario_metrics(frames, column, gt_intervals),
            })
    per_scenario = pd.DataFrame(per_scenario)
    if per_scenario.empty:
        return pd.DataFrame()

    interval_summary, _, _ = evaluate_saved_run(run_root, dataset_path, tiou_threshold=0.3)
    r_tiou = dict(zip(interval_summary.get("method", []), interval_summary.get("R_at_tIoU_0_3", [])))

    summary = []
    for name, key, _, _ in METHODS:
        group = per_scenario[per_scenario["method"] == name]
        if group.empty:
            continue
        row = {"method": name, "scenarios": len(group)}
        for metric in ["precision", "recall", "f1", "accuracy"]:
            row[metric] = group[metric].mean()
        row["fa_episodes"] = int(group["fa_episodes"].sum())
        row["macro_fpr"] = group["macro_fpr"].mean()
        n_gt = group["gt_intervals"].sum()
        row["segment_recall"] = group["detected_gt_intervals"].sum() / n_gt if n_gt else np.nan
        row["r_tiou_0_3"] = r_tiou.get(key, np.nan)
        row["ttd_p90_s"] = group["ttd_p90_s"].mean()
        summary.append(row)
    summary = pd.DataFrame(summary)

    summary.to_csv(results_root / "summary.csv", index=False)
    per_scenario.to_csv(results_root / "per_scenario.csv", index=False)
    return summary


def format_summary(summary: pd.DataFrame) -> str:
    """Render the summary table with the column names of the README."""
    if summary.empty:
        return "No evaluable outputs were found."
    table = summary.rename(columns={
        "method": "Method", "precision": "P", "recall": "R", "f1": "F1", "accuracy": "Acc",
        "fa_episodes": "FA Ep.", "macro_fpr": "Macro FPR", "segment_recall": "SegRec",
        "r_tiou_0_3": "R@tIoU0.3", "ttd_p90_s": "TTD P90 (s)",
    }).drop(columns=["scenarios"])
    return table.to_string(index=False, float_format=lambda value: f"{value:.3f}")
