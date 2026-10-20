"""Evaluate a saved Instruct2Alert run without calling the VLM.

Rewrites results/summary.csv, results/per_scenario.csv and
results/interval_metrics/ from the per-frame outputs of the run.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # repository root
from modules.evaluation import evaluate_run, format_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved Instruct2Alert run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--dataset-path",
        default=str(ROOT / "data" / "datasets" / "safety_vlm_dataset.xlsx"),
    )
    args = parser.parse_args()

    summary = evaluate_run(args.run_root, args.dataset_path)
    print(format_summary(summary))
    if summary.empty:
        return 1
    print(f"Saved under {Path(args.run_root) / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
