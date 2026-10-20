"""Strict interval metrics for Instruct2Alert evaluation.

The module operates only on saved evidence/controller outputs.  It does not
contact a model.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class Interval:
    """A half-open temporal interval in seconds."""

    start: float
    end: float

    def as_dict(self) -> dict:
        return {"start_sec": round(self.start, 6), "end_sec": round(self.end, 6)}


def temporal_iou(prediction: Interval, ground_truth: Interval) -> float:
    """Return temporal IoU for two half-open intervals."""
    intersection = max(0.0, min(prediction.end, ground_truth.end) -
                       max(prediction.start, ground_truth.start))
    union = max(prediction.end, ground_truth.end) - min(prediction.start, ground_truth.start)
    return intersection / union if union > 0.0 else 0.0


def maximum_cardinality_matching(
    predictions: Sequence[Interval],
    ground_truths: Sequence[Interval],
    threshold: float | None = None,
    positive_overlap_only: bool = False,
) -> list[tuple[int, int, float]]:
    """Return a one-to-one maximum-cardinality matching.

    Valid edges require ``tIoU >= threshold`` or, for boundary metrics,
    strictly positive overlap.  The assignment score includes a cardinality
    bonus larger than any possible total IoU, so cardinality is optimized
    first; total tIoU resolves ties.
    """
    if not predictions or not ground_truths:
        return []
    if threshold is None and not positive_overlap_only:
        raise ValueError("Provide a threshold or request positive-overlap matching.")

    rows, cols = len(predictions), len(ground_truths)
    valid = [[False for _ in range(cols)] for _ in range(rows)]
    ious = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for pred_index, prediction in enumerate(predictions):
        for gt_index, ground_truth in enumerate(ground_truths):
            iou = temporal_iou(prediction, ground_truth)
            ious[pred_index][gt_index] = iou
            edge_is_valid = iou > 0.0 if positive_overlap_only else iou >= float(threshold)
            if edge_is_valid:
                valid[pred_index][gt_index] = True

    # A successive shortest augmenting-path min-cost flow solves the exact
    # lexicographic objective without adding a SciPy runtime dependency:
    # augment until no valid path remains (maximum cardinality), then use
    # edge cost -tIoU to maximize total tIoU at that cardinality.
    source = 0
    prediction_offset = 1
    ground_truth_offset = prediction_offset + rows
    sink = ground_truth_offset + cols
    graph: list[list[list[float | int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: float) -> None:
        graph[start].append([end, len(graph[end]), capacity, cost])
        graph[end].append([start, len(graph[start]) - 1, 0, -cost])

    for pred_index in range(rows):
        add_edge(source, prediction_offset + pred_index, 1, 0.0)
    for pred_index in range(rows):
        for gt_index in range(cols):
            if valid[pred_index][gt_index]:
                add_edge(
                    prediction_offset + pred_index,
                    ground_truth_offset + gt_index,
                    1,
                    -ious[pred_index][gt_index],
                )
    for gt_index in range(cols):
        add_edge(ground_truth_offset + gt_index, sink, 1, 0.0)

    node_count = sink + 1
    while True:
        distance = [float("inf")] * node_count
        parent: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0.0
        # Bellman-Ford handles residual reverse edges introduced by previous
        # assignments and is small here because event interval counts are low.
        for _ in range(node_count - 1):
            changed = False
            for start, edges in enumerate(graph):
                if distance[start] == float("inf"):
                    continue
                for edge_index, (end, _, capacity, cost) in enumerate(edges):
                    if capacity and distance[start] + float(cost) < distance[int(end)] - 1e-12:
                        distance[int(end)] = distance[start] + float(cost)
                        parent[int(end)] = (start, edge_index)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            break
        node = sink
        while node != source:
            start, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[start][edge_index]
            edge[2] = int(edge[2]) - 1
            reverse_index = int(edge[1])
            graph[node][reverse_index][2] = int(graph[node][reverse_index][2]) + 1
            node = start

    matches = []
    for pred_index in range(rows):
        for end, _, capacity, _ in graph[prediction_offset + pred_index]:
            gt_index = int(end) - ground_truth_offset
            if 0 <= gt_index < cols and int(capacity) == 0 and valid[pred_index][gt_index]:
                matches.append((pred_index, gt_index, ious[pred_index][gt_index]))
    return sorted(matches, key=lambda item: (item[1], item[0]))


def sampled_states_to_intervals(
    records: Sequence[dict],
    prediction_key: str,
    video_duration_sec: float | None = None,
) -> list[Interval]:
    """Convert sampled binary states to temporal intervals.

    A positive sample starts an alert at its sampled timestamp. Its interval
    ends at the timestamp of the first subsequent negative observation. If a
    positive state is still active at the final sample, the recorded video
    duration is used when available; otherwise the final observed timestamp is
    used. This convention preserves observed onset times and avoids inventing
    evidence during unobserved gaps.
    """
    ordered = sorted(
        records,
        key=lambda row: (float(row["video_timestamp_sec"]), int(row["frame_index"])),
    )
    intervals: list[Interval] = []
    start = None
    last_time = None
    for row in ordered:
        timestamp = float(row["video_timestamp_sec"])
        positive = str(row.get(prediction_key, "No")).strip().lower() in {"yes", "1", "true"}
        if positive and start is None:
            start = timestamp
        elif not positive and start is not None:
            if timestamp > start:
                intervals.append(Interval(start, timestamp))
            start = None
        last_time = timestamp
    if start is not None:
        end = video_duration_sec if video_duration_sec is not None else last_time
        if end is not None and end > start:
            intervals.append(Interval(start, float(end)))
    return intervals


def frame_intervals_to_seconds(
    frame_intervals: Iterable[Sequence[int]], source_fps: float,
) -> list[Interval]:
    """Convert 1-based inclusive annotation frame intervals to seconds."""
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    converted = []
    for interval in frame_intervals:
        if len(interval) < 2:
            continue
        start_frame, end_frame = int(interval[0]), int(interval[1])
        if end_frame < start_frame:
            continue
        converted.append(Interval((start_frame - 1) / source_fps, end_frame / source_fps))
    return converted


def evaluate_interval_quality(
    predictions: Sequence[Interval],
    ground_truths: Sequence[Interval],
    tiou_threshold: float = 0.3,
) -> tuple[dict, list[dict]]:
    """Evaluate strict interval quality and return summary plus event records."""
    strict_matches = maximum_cardinality_matching(
        predictions, ground_truths, threshold=tiou_threshold)
    overlap_matches = maximum_cardinality_matching(
        predictions, ground_truths, positive_overlap_only=True)
    overlap_by_gt = {gt_index: (pred_index, iou)
                     for pred_index, gt_index, iou in overlap_matches}
    strict_by_gt = {gt_index: (pred_index, iou)
                    for pred_index, gt_index, iou in strict_matches}

    onset_errors = []
    offset_errors = []
    event_records = []
    for gt_index, ground_truth in enumerate(ground_truths):
        overlap = overlap_by_gt.get(gt_index)
        strict = strict_by_gt.get(gt_index)
        if overlap is not None:
            prediction = predictions[overlap[0]]
            onset_error = abs(prediction.start - ground_truth.start)
            offset_error = abs(prediction.end - ground_truth.end)
            onset_errors.append(onset_error)
            offset_errors.append(offset_error)
        else:
            onset_error = None
            offset_error = None

        event_records.append({
            "gt_interval_index": gt_index,
            "gt_start_sec": round(ground_truth.start, 6),
            "gt_end_sec": round(ground_truth.end, 6),
            "positive_overlap_matched": overlap is not None,
            "positive_overlap_prediction_index": overlap[0] if overlap else None,
            "positive_overlap_tiou": round(overlap[1], 6) if overlap else None,
            "strict_matched": strict is not None,
            "strict_prediction_index": strict[0] if strict else None,
            "strict_tiou": round(strict[1], 6) if strict else None,
            "onset_abs_error_sec": round(onset_error, 6) if onset_error is not None else None,
            "offset_abs_error_sec": round(offset_error, 6) if offset_error is not None else None,
        })

    metrics = {
        "ground_truth_interval_count": len(ground_truths),
        "prediction_interval_count": len(predictions),
        "strict_match_count": len(strict_matches),
        "positive_overlap_match_count": len(overlap_matches),
        "R_at_tIoU_0_3": (len(strict_matches) / len(ground_truths)) if ground_truths else None,
        "OnsetMAE_sec": (sum(onset_errors) / len(onset_errors)) if onset_errors else None,
        "OffsetMAE_sec": (sum(offset_errors) / len(offset_errors)) if offset_errors else None,
    }
    return metrics, event_records


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_yes_no(value) -> str:
    return "Yes" if str(value).strip().lower() in {"yes", "1", "true"} else "No"


def _pool_rule_rows(rows: Sequence[dict], prediction_key: str) -> list[dict]:
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        frame_index = int(float(row["frame_index"]))
        timestamp = round(float(row["video_timestamp_sec"]), 6)
        grouped.setdefault((frame_index, timestamp), []).append(row)
    pooled = []
    for (frame_index, timestamp), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        pooled.append({
            "frame_index": frame_index,
            "video_timestamp_sec": timestamp,
            prediction_key: "Yes" if any(_as_yes_no(row.get(prediction_key)) == "Yes" for row in group) else "No",
        })
    return pooled


def evaluate_saved_run(
    run_root: str | Path,
    dataset_path: str | Path,
    tiou_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every available internal variant from saved run artifacts."""
    run_root = Path(run_root)
    results_root = run_root / "results"
    dataset = pd.read_excel(dataset_path)
    sources = {
        "direct_prompt": ("direct_prompt_evidence.jsonl", "binary_evidence"),
        "workflow_verify": ("workflow_verify_evidence.jsonl", "binary_evidence"),
        "workflow_onset": ("workflow_onset_replay.csv", "replayed_alert"),
        "grounded_verify": ("grounded_verify_evidence.jsonl", "binary_evidence"),
        "full": ("full_controller_replay.csv", "replayed_alert"),
    }
    scenario_rows = []
    event_rows = []
    interval_rows = []
    for _, dataset_row in dataset.iterrows():
        scenario_id = str(int(dataset_row["Scenario_ID"]))
        audit_dir = results_root / "scenarios" / scenario_id / "audit"
        schedule_path = audit_dir / "frame_schedule.csv"
        if not schedule_path.exists():
            continue
        schedule = _read_rows(schedule_path)
        if not schedule:
            continue
        source_fps = float(schedule[0].get("source_fps", 0.0))
        source_total_frames = int(float(schedule[0].get("source_total_frames", 0)))
        duration_sec = (source_total_frames / source_fps) if source_fps > 0 else None
        try:
            raw_gt = json.loads(str(dataset_row.get("GT_intervals_frame", "[]")))
        except json.JSONDecodeError:
            raw_gt = []
        ground_truths = frame_intervals_to_seconds(raw_gt, source_fps)
        # Store GT endpoints once per scenario. Predicted endpoints are stored
        # once per method below, so every strict metric can be audited without
        # reconstructing intervals from frame-level evidence again.
        for interval_index, interval in enumerate(ground_truths):
            interval_rows.append({
                "interval_type": "ground_truth",
                "method": None,
                "scenario_id": int(dataset_row["Scenario_ID"]),
                "video_id": dataset_row["Video_name"],
                "interval_index": interval_index,
                **interval.as_dict(),
                "source_fps": source_fps,
            })
        for method, (filename, prediction_key) in sources.items():
            source_path = audit_dir / filename
            if not source_path.exists():
                continue
            rows = _read_rows(source_path)
            if not rows:
                continue
            pooled = _pool_rule_rows(rows, prediction_key)
            predictions = sampled_states_to_intervals(pooled, prediction_key, duration_sec)
            for interval_index, interval in enumerate(predictions):
                interval_rows.append({
                    "interval_type": "prediction",
                    "method": method,
                    "scenario_id": int(dataset_row["Scenario_ID"]),
                    "video_id": dataset_row["Video_name"],
                    "interval_index": interval_index,
                    **interval.as_dict(),
                    "source_fps": source_fps,
                })
            metrics, events = evaluate_interval_quality(
                predictions, ground_truths,
                tiou_threshold=tiou_threshold,
            )
            scenario_rows.append({
                "method": method,
                "scenario_id": int(dataset_row["Scenario_ID"]),
                "video_id": dataset_row["Video_name"],
                "tiou_threshold": tiou_threshold,
                "ground_truth_intervals": json.dumps(
                    [interval.as_dict() for interval in ground_truths]),
                "predicted_intervals": json.dumps(
                    [interval.as_dict() for interval in predictions]),
                **metrics,
            })
            for event in events:
                event_rows.append({
                    "method": method,
                    "scenario_id": int(dataset_row["Scenario_ID"]),
                    "video_id": dataset_row["Video_name"],
                    **event,
                })

    scenario_df = pd.DataFrame(scenario_rows)
    event_df = pd.DataFrame(event_rows)
    interval_df = pd.DataFrame(interval_rows)
    summary_rows = []
    for method, group in scenario_df.groupby("method") if not scenario_df.empty else []:
        n_gt = int(group["ground_truth_interval_count"].sum())
        n_strict = int(group["strict_match_count"].sum())
        overlap = group[group["positive_overlap_match_count"] > 0]
        total_overlap = int(group["positive_overlap_match_count"].sum())
        # MAE is event-weighted, so aggregate from event rows rather than
        # averaging per-scenario estimates with unequal event counts.
        method_events = event_df[(event_df["method"] == method) &
                                 event_df["positive_overlap_matched"]] if not event_df.empty else pd.DataFrame()
        summary_rows.append({
            "method": method,
            "scenario_count": len(group),
            "ground_truth_interval_count": n_gt,
            "prediction_interval_count": int(group["prediction_interval_count"].sum()),
            "strict_match_count": n_strict,
            "positive_overlap_match_count": total_overlap,
            "R_at_tIoU_0_3": n_strict / n_gt if n_gt else None,
            "OnsetMAE_sec": method_events["onset_abs_error_sec"].mean() if not method_events.empty else None,
            "OffsetMAE_sec": method_events["offset_abs_error_sec"].mean() if not method_events.empty else None,
        })
    summary_df = pd.DataFrame(summary_rows)

    output_dir = results_root / "interval_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(output_dir / "scenario_interval_metrics.csv", index=False)
    event_df.to_csv(output_dir / "event_interval_metrics.csv", index=False)
    interval_df.to_csv(output_dir / "interval_records.csv", index=False)
    summary_df.to_csv(output_dir / "interval_metrics_summary.csv", index=False)
    return summary_df, scenario_df, event_df
