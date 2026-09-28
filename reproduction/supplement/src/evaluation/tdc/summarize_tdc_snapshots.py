"""Summarize and completeness-check all Step-2 snapshot TDC JSON results."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RESULTS_ROOT", "/mnt/results_1.5b_p3_s2"))
EXPECTED_SNAPSHOTS = int(os.environ.get("EXPECTED_SNAPSHOTS", "8"))
EXPECTED_TASKS = int(os.environ.get("EXPECTED_TASKS", "23"))

rows = []
for path in sorted(ROOT.glob("step_*/results_*.json")):
    payload = json.loads(path.read_text())
    metric = payload["primary_metric"]
    pretrained = payload["pretrained"]
    rows.append(
        {
            "step": int(path.parent.name.split("_")[1]),
            "task": payload["task"],
            "task_type": payload["task_type"],
            "metric": metric,
            "mean": pretrained[f"test_{metric}_mean"],
            "std": pretrained[f"test_{metric}_std"],
            "n_seeds": pretrained["n_seeds"],
            "checkpoint": payload["ckpt"],
            "result_file": str(path),
        }
    )

expected = EXPECTED_SNAPSHOTS * EXPECTED_TASKS
if len(rows) != expected:
    raise SystemExit(f"ERROR: found {len(rows)}/{expected} expected result files")

fields = list(rows[0])
with (ROOT / "tdc_all_snapshots.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "tdc_all_snapshots.json").write_text(json.dumps(rows, indent=2) + "\n")

best = []
for task in sorted({row["task"] for row in rows}):
    candidates = [row for row in rows if row["task"] == task]
    reverse = candidates[0]["metric"] != "mae"
    winner = sorted(candidates, key=lambda row: row["mean"], reverse=reverse)[0]
    best.append(winner)
with (ROOT / "tdc_best_snapshot_by_task.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(best)

print(f"Validated {len(rows)} results: {EXPECTED_SNAPSHOTS} snapshots x {EXPECTED_TASKS} tasks")
