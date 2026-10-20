"""Utilities for append-only run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


def json_default(value: Any):
    """Convert NumPy values produced by CV modules into portable JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True,
                  default=json_default)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    default=json_default))
            handle.write("\n")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def apply_run_seed(seed: int) -> dict[str, Any]:
    """Apply the run seed to the CV process and report the exact policy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn-only preserves operational compatibility while exposing unsupported
    # nondeterministic kernels in the captured execution log.
    torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": int(seed),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "opencv_rng_seeded": True,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": "warn_only",
    }


def snapshot_core_files(project_root: str | Path, destination: str | Path) -> dict[str, str]:
    """Copy the execution-critical source files and return their checksums."""
    project_root = Path(project_root)
    destination = Path(destination)
    relative_paths = [
        Path("run_e2e_pipeline.py"),
        Path("scripts/evaluate.py"),
        Path("run_dual_env.sh"),
        Path("requirements/cv.txt"),
        Path("requirements/vlm.txt"),
    ]
    relative_paths.extend(
        path.relative_to(project_root)
        for path in (project_root / "modules").rglob("*.py")
        if "__pycache__" not in path.parts
    )

    checksums: dict[str, str] = {}
    for relative_path in sorted(set(relative_paths)):
        source = project_root / relative_path
        if not source.is_file():
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksums[str(relative_path)] = sha256_file(source)
    return checksums


def runtime_environment() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def summarize_latencies(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize invocation-level latency events without discarding raw events."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        latency = row.get("latency_sec")
        if latency is None:
            continue
        grouped.setdefault(str(row.get("component", "unknown")), []).append(float(latency))

    summaries = []
    for component, values in sorted(grouped.items()):
        values_array = np.asarray(values, dtype=float)
        summaries.append({
            "component": component,
            "invocation_count": int(values_array.size),
            "mean_sec": round(float(np.mean(values_array)), 6),
            "median_sec": round(float(np.median(values_array)), 6),
            "p90_sec": round(float(np.percentile(values_array, 90)), 6),
            "p99_sec": round(float(np.percentile(values_array, 99)), 6),
            "max_sec": round(float(np.max(values_array)), 6),
        })
    return summaries
