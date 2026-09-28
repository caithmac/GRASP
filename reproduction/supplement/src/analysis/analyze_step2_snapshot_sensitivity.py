#!/usr/bin/env python3
"""Descriptive sensitivity analysis across fixed Step-2 checkpoints.

This script intentionally does *not* select a best checkpoint per task. It
summarizes how the reported test metrics vary over a declared set of snapshots.
All pairwise sign tests are descriptive, two-sided, and uncorrected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "1.5b_p3_step2_matched_tdc_results"
    "/results_1.5b_p3_s2_matched/tdc_all_snapshots.csv"
)
DEFAULT_OUTPUT = Path("paper_iclr/evidence/step2_snapshot_sensitivity")
EXPECTED_STEPS = 8
EXPECTED_TASKS = 23
EXPECTED_SEEDS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test under p=0.5; ties are omitted."""

    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, probability)


def average_descending_ranks(values_by_step: dict[int, float]) -> dict[int, float]:
    """Return average ranks (1 is best), treating exact equal values as ties."""

    ordered = sorted(values_by_step.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for step, _ in ordered[index:end]:
            ranks[step] = average_rank
        index = end
    return ranks


def load_and_validate(path: Path) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    required = {
        "step",
        "task",
        "task_type",
        "metric",
        "mean",
        "std",
        "n_seeds",
        "checkpoint",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        raw_rows = list(reader)

    expected_rows = EXPECTED_STEPS * EXPECTED_TASKS
    if len(raw_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} task-step rows, found {len(raw_rows)}")

    rows: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, int]] = set()
    metrics_by_task: dict[str, set[str]] = defaultdict(set)
    task_types_by_task: dict[str, set[str]] = defaultdict(set)
    checkpoints_by_step: dict[int, set[str]] = defaultdict(set)
    steps_by_task: dict[str, set[int]] = defaultdict(set)

    for line_number, raw in enumerate(raw_rows, start=2):
        try:
            step = int(raw["step"])
            task = raw["task"].strip()
            task_type = raw["task_type"].strip().lower()
            metric = raw["metric"].strip().lower()
            mean = float(raw["mean"])
            std = float(raw["std"])
            n_seeds = int(raw["n_seeds"])
            checkpoint = raw["checkpoint"].strip()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed value at CSV line {line_number}: {exc}") from exc

        if not task or not metric or not task_type or not checkpoint:
            raise ValueError(f"Blank required value at CSV line {line_number}")
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise ValueError(f"Invalid mean/std at CSV line {line_number}")
        if n_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"Expected n_seeds={EXPECTED_SEEDS} in every task-step cell; "
                f"found {n_seeds} at CSV line {line_number}"
            )
        cell = (task, step)
        if cell in seen_cells:
            raise ValueError(f"Duplicate task-step cell: task={task!r}, step={step}")
        seen_cells.add(cell)

        metrics_by_task[task].add(metric)
        task_types_by_task[task].add(task_type)
        checkpoints_by_step[step].add(checkpoint)
        steps_by_task[task].add(step)
        oriented_mean = -mean if metric == "mae" else mean
        rows.append(
            {
                "step": step,
                "task": task,
                "task_type": task_type,
                "metric": metric,
                "mean": mean,
                "std": std,
                "n_seeds": n_seeds,
                "checkpoint": checkpoint,
                "orientation": "lower_is_better" if metric == "mae" else "higher_is_better",
                "oriented_mean": oriented_mean,
            }
        )

    steps = sorted(checkpoints_by_step)
    tasks = sorted(steps_by_task)
    if len(steps) != EXPECTED_STEPS:
        raise ValueError(f"Expected {EXPECTED_STEPS} steps, found {len(steps)}: {steps}")
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} tasks, found {len(tasks)}")
    all_steps = set(steps)
    incomplete = {task: sorted(all_steps - values) for task, values in steps_by_task.items() if values != all_steps}
    if incomplete:
        raise ValueError(f"Tasks do not all contain every step: {incomplete}")
    inconsistent_metrics = {task: sorted(values) for task, values in metrics_by_task.items() if len(values) != 1}
    if inconsistent_metrics:
        raise ValueError(f"Metric changes across steps: {inconsistent_metrics}")
    inconsistent_types = {task: sorted(values) for task, values in task_types_by_task.items() if len(values) != 1}
    if inconsistent_types:
        raise ValueError(f"Task type changes across steps: {inconsistent_types}")
    multi_checkpoint_steps = {
        step: sorted(values) for step, values in checkpoints_by_step.items() if len(values) != 1
    }
    if multi_checkpoint_steps:
        raise ValueError(f"A step maps to multiple checkpoints: {multi_checkpoint_steps}")
    unique_checkpoints = {next(iter(values)) for values in checkpoints_by_step.values()}
    if len(unique_checkpoints) != EXPECTED_STEPS:
        raise ValueError(
            f"Expected {EXPECTED_STEPS} unique checkpoints, found {len(unique_checkpoints)}"
        )

    return rows, steps, tasks


def analyze(rows: list[dict[str, Any]], steps: list[int], tasks: list[str], reference_step: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if reference_step not in steps:
        raise ValueError(f"Reference step {reference_step} is absent; available steps: {steps}")

    by_task: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_task[row["task"]][row["step"]] = row

    rank_rows: list[dict[str, Any]] = []
    ranks_by_step: dict[int, list[float]] = defaultdict(list)
    wins_by_step: Counter[int] = Counter()
    for task in tasks:
        oriented = {step: by_task[task][step]["oriented_mean"] for step in steps}
        ranks = average_descending_ranks(oriented)
        best_value = max(oriented.values())
        for step in steps:
            row = by_task[task][step]
            is_task_win = oriented[step] == best_value
            ranks_by_step[step].append(ranks[step])
            wins_by_step[step] += int(is_task_win)
            rank_rows.append(
                {
                    "task": task,
                    "step": step,
                    "metric": row["metric"],
                    "orientation": row["orientation"],
                    "reported_mean": row["mean"],
                    "oriented_mean": row["oriented_mean"],
                    "rank": ranks[step],
                    "is_task_win": is_task_win,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for step in steps:
        ranks = sorted(ranks_by_step[step])
        midpoint = len(ranks) // 2
        median = ranks[midpoint] if len(ranks) % 2 else (ranks[midpoint - 1] + ranks[midpoint]) / 2.0
        summary_rows.append(
            {
                "step": step,
                "checkpoint": by_task[tasks[0]][step]["checkpoint"],
                "n_tasks": len(tasks),
                "mean_rank": sum(ranks) / len(ranks),
                "median_rank": median,
                "task_wins_including_exact_ties": wins_by_step[step],
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for step in steps:
        if step == reference_step:
            continue
        better = worse = ties = 0
        for task in tasks:
            candidate = by_task[task][step]["oriented_mean"]
            reference = by_task[task][reference_step]["oriented_mean"]
            if candidate > reference:
                better += 1
            elif candidate < reference:
                worse += 1
            else:
                ties += 1
        pairwise_rows.append(
            {
                "step": step,
                "reference_step": reference_step,
                "n_tasks": len(tasks),
                "better_than_reference": better,
                "worse_than_reference": worse,
                "exact_ties": ties,
                "non_tied_tasks": better + worse,
                "exact_two_sided_sign_test_p_descriptive_uncorrected": exact_two_sided_sign_test(better, worse),
            }
        )

    return summary_rows, pairwise_rows, rank_rows


def render_results_md(
    input_arg: Path,
    input_sha256: str,
    script_sha256: str,
    reference_step: int,
    summary_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
) -> str:
    summary_lines = [
        "| Step | Mean rank | Median rank | Task wins* |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"| {row['step']} | {row['mean_rank']:.4f} | {row['median_rank']:.4f} | "
            f"{row['task_wins_including_exact_ties']} |"
        )

    pairwise_lines = [
        f"| Step vs {reference_step} | Better | Worse | Ties | Exact two-sided sign-test p** |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pairwise_rows:
        pairwise_lines.append(
            f"| {row['step']} | {row['better_than_reference']} | "
            f"{row['worse_than_reference']} | {row['exact_ties']} | "
            f"{row['exact_two_sided_sign_test_p_descriptive_uncorrected']:.8g} |"
        )

    return "\n".join(
        [
            "# Step-2 snapshot sensitivity (descriptive)",
            "",
            "This artifact describes test-metric sensitivity across eight already evaluated "
            "checkpoints. It is **not** a checkpoint-selection analysis, and no per-task "
            "test-selected best checkpoint is used or recommended as a model-selection result.",
            "",
            "## Provenance and validation",
            "",
            f"- Input: `{input_arg.as_posix()}`",
            f"- Input SHA256: `{input_sha256}`",
            f"- Analysis script SHA256: `{script_sha256}`",
            "- Validated design: exactly 8 unique checkpoints × 23 tasks; every unique "
            "task-step cell reports `n_seeds = 3`.",
            "- Orientation: MAE is lower-is-better; every other reported metric is "
            "higher-is-better.",
            f"- Declared reference checkpoint: step {reference_step}.",
            "- The rationale for the original checkpoint choice is not inferred from these "
            "test results or from this retrospective analysis.",
            "",
            "## Rank summary",
            "",
            *summary_lines,
            "",
            "*A task win is the best oriented reported mean among the eight snapshots; exact "
            "ties count as wins for every tied step. Ranks use average ranks for exact ties.",
            "",
            "## Pairwise directions versus the declared reference",
            "",
            *pairwise_lines,
            "",
            "**P-values are exact two-sided sign tests over non-tied tasks. They are "
            "descriptive and uncorrected for the seven comparisons; they must not be read as "
            "confirmatory evidence or as a checkpoint-selection criterion.",
            "",
            "Machine-readable per-task ranks and orientations are stored in `summary.json`; "
            "aggregate tables are in `summary_by_step.csv` and `pairwise_vs_reference.csv`.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-step", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    script_path = Path(__file__).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows, steps, tasks = load_and_validate(input_path)
    summary_rows, pairwise_rows, rank_rows = analyze(rows, steps, tasks, args.reference_step)
    input_sha256 = sha256_file(input_path)
    script_sha256 = sha256_file(script_path)

    summary_fields = [
        "step",
        "checkpoint",
        "n_tasks",
        "mean_rank",
        "median_rank",
        "task_wins_including_exact_ties",
    ]
    pairwise_fields = [
        "step",
        "reference_step",
        "n_tasks",
        "better_than_reference",
        "worse_than_reference",
        "exact_ties",
        "non_tied_tasks",
        "exact_two_sided_sign_test_p_descriptive_uncorrected",
    ]
    payload = {
        "analysis_scope": "descriptive_snapshot_sensitivity_not_model_selection",
        "caveats": [
            "No test-selected best-per-task checkpoint is used as a model-selection result.",
            "The original checkpoint-selection rationale is not inferred.",
            "Exact two-sided sign-test p-values are descriptive and uncorrected.",
        ],
        "provenance": {
            "input_path": args.input.as_posix(),
            "input_sha256": input_sha256,
            "script_path": script_path.name,
            "script_sha256": script_sha256,
        },
        "validation": {
            "expected_unique_checkpoints": EXPECTED_STEPS,
            "observed_unique_checkpoints": len(steps),
            "expected_tasks": EXPECTED_TASKS,
            "observed_tasks": len(tasks),
            "expected_seeds_per_task_step": EXPECTED_SEEDS,
            "observed_task_step_cells": len(rows),
            "unique_task_step_cells": True,
            "complete_rectangular_matrix": True,
        },
        "orientation_rule": {"mae": "lower_is_better", "all_other_metrics": "higher_is_better"},
        "reference_step": args.reference_step,
        "summary_by_step": summary_rows,
        "pairwise_vs_reference": pairwise_rows,
        "per_task_ranks": rank_rows,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "summary_by_step.csv", csv_text(summary_fields, summary_rows))
    atomic_write_text(output_dir / "pairwise_vs_reference.csv", csv_text(pairwise_fields, pairwise_rows))
    atomic_write_text(output_dir / "summary.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_write_text(
        output_dir / "RESULTS.md",
        render_results_md(
            args.input,
            input_sha256,
            script_sha256,
            args.reference_step,
            summary_rows,
            pairwise_rows,
        ),
    )


if __name__ == "__main__":
    main()
