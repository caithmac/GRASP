#!/usr/bin/env python3
"""Lock split/label hashes after cross-checking all historical TDC logs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tdc_classical_baselines import TASKS, sha256_file, split_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--historical-log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-revision-audit-id",
        default="REDACTED_FOR_ANONYMOUS_REVIEW",
        help="Anonymous audit identifier written in place of the source commit.",
    )
    args = parser.parse_args()
    tasks, logs = {}, {}
    for task in TASKS:
        _, _, audit = split_audit(task, args.data_dir / f"{task}.tab")
        log_path = args.historical_log_dir / f"finetune_{task}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        found = {name: int(value) for name, value in re.findall(r"\b(train|valid|test): n=(\d+)", text)}
        observed = {name: audit["partitions"][name]["effective_rows"] for name in ("train", "valid", "test")}
        if found != observed:
            raise RuntimeError(f"{task}: reconstructed counts {observed} != historical log {found}")
        tasks[task] = {
            "source_sha256": audit["source_sha256"],
            "loader_rows": audit["loader_rows"],
            "partitions": audit["partitions"],
        }
        logs[task] = {"sha256": sha256_file(log_path), "effective_rows": found}
    payload = {
        "schema_version": 1,
        "historical_environment": {
            "pytdc": "0.4.1",
            "rdkit_distribution": "rdkit-pypi==2022.9.5",
            "historical_run_date": "2026-07-22",
            "raw_data_last_git_commit": args.source_revision_audit_id,
        },
        "protocol": "PyTDC 0.4.1 loader plus create_scaffold_split; scaffold seed 42; frac 0.70/0.15/0.15; post-split 1..96-heavy-atom filter",
        "provenance_argument": (
            "The raw .tab files were committed before the historical runs; the split algorithm is deterministic. "
            "All 23 reconstructed effective partition counts exactly match the retained historical logs."
        ),
        "task_order": list(TASKS),
        "tasks": tasks,
        "historical_logs": logs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(tasks)} verified tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
