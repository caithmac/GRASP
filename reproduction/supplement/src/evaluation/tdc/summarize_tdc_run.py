"""Validate and summarize one fixed-checkpoint 22-task TDC ADMET run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=22)
    parser.add_argument(
        "--exclude-task", action="append", default=[],
        help="Task name to ignore without deleting its historical JSON (repeatable).",
    )
    args = parser.parse_args()

    rows = []
    checkpoints = set()
    excluded_tasks = set(args.exclude_task)
    for path in sorted(args.results_dir.glob("results_*.json")):
        data = json.loads(path.read_text())
        if data["task"] in excluded_tasks:
            continue
        metric = data["primary_metric"]
        result = data["pretrained"]
        checkpoints.add(data["ckpt"])
        rows.append({
            "task": data["task"],
            "task_type": data["task_type"],
            "metric": metric,
            "test_mean": result[f"test_{metric}_mean"],
            "test_std": result[f"test_{metric}_std"],
            "val_mean": result[f"val_{metric}_mean"],
            "val_std": result[f"val_{metric}_std"],
            "n_seeds": result["n_seeds"],
        })
    if len(rows) != args.expected_tasks:
        raise SystemExit(f"ERROR: found {len(rows)}/{args.expected_tasks} task JSONs")
    if len({row["task"] for row in rows}) != len(rows):
        raise SystemExit("ERROR: duplicate task names in result JSONs")
    if len(checkpoints) != 1:
        raise SystemExit(f"ERROR: results reference {len(checkpoints)} checkpoints")

    with (args.results_dir / "tdc_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / "tdc_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Validated {len(rows)} tasks from fixed checkpoint {next(iter(checkpoints))}")


if __name__ == "__main__":
    main()
