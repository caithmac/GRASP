#!/usr/bin/env python3
"""Validate and aggregate the audited TDC classical-baseline task outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from tdc_classical_baselines import (
    FINAL_SEEDS,
    REPRESENTATIONS,
    TASKS,
    TASK_CONFIG,
    candidate_grid,
    canonical_json_sha256,
    sha256_file,
    utility,
)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def exact_sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    lower = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(lower + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def outcome(left: float, right: float, metric: str, tolerance: float = 1e-12) -> int:
    delta = utility(metric, left) - utility(metric, right)
    return 1 if delta > tolerance else (-1 if delta < -tolerance else 0)


def checked_score(value: Any, metric: str, context: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"{context}: non-finite {metric} score")
    if metric in {"auroc", "auprc"} and not 0.0 <= score <= 1.0:
        raise RuntimeError(f"{context}: {metric} outside [0, 1]")
    if metric == "spearman" and not -1.0 <= score <= 1.0:
        raise RuntimeError(f"{context}: spearman outside [-1, 1]")
    if metric == "mae" and score < 0.0:
        raise RuntimeError(f"{context}: negative MAE")
    return score


def comparator_value(path: Path, task: str, metric: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task:
        raise ValueError(f"task mismatch in {path}: {payload.get('task')} != {task}")
    if payload.get("primary_metric") != metric:
        raise ValueError(f"metric mismatch in {path}: {payload.get('primary_metric')} != {metric}")
    results = payload.get("results", payload.get("pretrained", {}))
    key = f"test_{metric}_mean"
    if key not in results:
        raise ValueError(f"missing {key} in {path}")
    return checked_score(results[key], metric, str(path))


def validate_result_payload(
    item: dict[str, Any], task: str, reference: dict[str, Any], expected_runner_sha256: str,
) -> None:
    if item.get("schema_version") != 1 or item.get("task") != task:
        raise RuntimeError(f"{task}: schema/task mismatch")
    if set(item.get("representations", {})) != set(REPRESENTATIONS):
        raise RuntimeError(f"{task}: representation set mismatch")
    expected_type, expected_metric = TASK_CONFIG.get(task, ("classification", "auroc"))
    if (item.get("task_type"), item.get("primary_metric")) != (expected_type, expected_metric):
        raise RuntimeError(f"{task}: task type or primary metric mismatch")

    audit = item.get("split_audit", {})
    comparable = {
        "source_sha256": audit.get("source_sha256"),
        "loader_rows": audit.get("loader_rows"),
        "partitions": audit.get("partitions"),
    }
    if comparable != reference["tasks"][task]:
        raise RuntimeError(f"{task}: split provenance differs from locked reference")
    leakage = audit.get("leakage_audit", {})
    if leakage.get("status") != "PASS":
        raise RuntimeError(f"{task}: missing passing leakage audit")
    for key in ("canonical_smiles_overlap_counts", "murcko_scaffold_overlap_counts"):
        counts = leakage.get(key)
        if not isinstance(counts, dict) or set(counts) != {"train_valid", "train_test", "valid_test"}:
            raise RuntimeError(f"{task}: malformed {key}")
        if any(value != 0 for value in counts.values()):
            raise RuntimeError(f"{task}: nonzero {key}")

    protocol = item.get("protocol", {})
    if protocol.get("inference_unit") != "endpoint" or tuple(protocol.get("representations", ())) != REPRESENTATIONS:
        raise RuntimeError(f"{task}: protocol mismatch")
    if tuple(protocol.get("final_seeds", ())) != FINAL_SEEDS:
        raise RuntimeError(f"{task}: protocol seed mismatch")
    if protocol.get("runner_sha256") != expected_runner_sha256:
        raise RuntimeError(f"{task}: runner hash mismatch")
    if protocol.get("reference_content_sha256") != canonical_json_sha256(reference):
        raise RuntimeError(f"{task}: reference hash mismatch")
    n_estimators = protocol.get("n_estimators")
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise RuntimeError(f"{task}: invalid estimator count")

    software = item.get("software", {})
    required_software = {"python", "rdkit", "numpy", "pandas", "scikit_learn"}
    if set(software) != required_software or any(not str(software[key]) for key in required_software):
        raise RuntimeError(f"{task}: incomplete software manifest")
    expected_versions = {
        "rdkit": "2022.09.5",
        "numpy": "1.26.4",
        "pandas": "1.5.3",
        "scikit_learn": "1.5.2",
    }
    mismatches = {
        key: (software.get(key), value)
        for key, value in expected_versions.items()
        if software.get(key) != value
    }
    if not str(software.get("python", "")).startswith("3.10.") or mismatches:
        raise RuntimeError(f"{task}: software pins differ from audited runtime: {mismatches}")

    expected_candidates = candidate_grid()
    for representation in REPRESENTATIONS:
        result = item["representations"][representation]
        expected_estimator = "ExtraTreesClassifier" if expected_type == "classification" else "ExtraTreesRegressor"
        if result.get("estimator") != expected_estimator or result.get("n_estimators") != n_estimators:
            raise RuntimeError(f"{task}/{representation}: estimator contract mismatch")
        if representation == "ecfp4" and result.get("feature_count") != 2048:
            raise RuntimeError(f"{task}/{representation}: unexpected feature count")
        if representation == "rdkit_descriptors" and (
            not isinstance(result.get("feature_count"), int) or result["feature_count"] <= 0
        ):
            raise RuntimeError(f"{task}/{representation}: invalid descriptor count")
        if not isinstance(result.get("feature_schema_sha256"), str) or len(result["feature_schema_sha256"]) != 64:
            raise RuntimeError(f"{task}/{representation}: invalid feature-schema hash")

        candidates = result.get("validation_candidates")
        if not isinstance(candidates, list) or len(candidates) != len(expected_candidates):
            raise RuntimeError(f"{task}/{representation}: validation grid incomplete")
        if [entry.get("params") for entry in candidates] != expected_candidates:
            raise RuntimeError(f"{task}/{representation}: validation grid differs from runner")
        for index, entry in enumerate(candidates):
            score = checked_score(entry.get("validation_score"), expected_metric, f"{task}/{representation}/candidate-{index}")
            expected_utility = utility(expected_metric, score)
            if not math.isclose(float(entry.get("validation_utility")), expected_utility, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"{task}/{representation}: invalid validation utility")
        selected = max(candidates, key=lambda entry: (entry["validation_utility"], -entry["params"]["min_samples_leaf"]))
        if result.get("selected_params") != selected["params"]:
            raise RuntimeError(f"{task}/{representation}: selected parameters do not follow validation rule")

        runs = result.get("test_runs")
        if not isinstance(runs, list) or [entry.get("seed") for entry in runs] != list(FINAL_SEEDS):
            raise RuntimeError(f"{task}/{representation}: expected exact ordered final seeds {FINAL_SEEDS}")
        scores = np.asarray([
            checked_score(entry.get("test_score"), expected_metric, f"{task}/{representation}/seed-{entry.get('seed')}")
            for entry in runs
        ])
        if not math.isclose(float(result.get("test_mean")), float(scores.mean()), rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"{task}/{representation}: test mean does not match seeds")
        if not math.isclose(float(result.get("test_std_ddof0")), float(scores.std(ddof=0)), rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"{task}/{representation}: test standard deviation does not match seeds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--comparator-dir", type=Path)
    args = parser.parse_args()
    complete_path = args.results_dir / "COMPLETE"
    if complete_path.exists():
        raise RuntimeError(f"refusing to finalize non-fresh directory with existing marker: {complete_path}")
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if tuple(reference["task_order"]) != TASKS:
        raise RuntimeError("reference task order differs from runner")
    expected_paths = {args.results_dir / f"results_{task}.json" for task in TASKS}
    observed_paths = set(args.results_dir.glob("results_*.json"))
    if observed_paths != expected_paths:
        missing = sorted(path.name for path in expected_paths - observed_paths)
        extra = sorted(path.name for path in observed_paths - expected_paths)
        raise RuntimeError(f"result file set mismatch; missing={missing}, extra={extra}")
    expected_runner_sha256 = sha256_file(Path(__file__).with_name("tdc_classical_baselines.py"))

    payloads: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        path = args.results_dir / f"results_{task}.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        validate_result_payload(item, task, reference, expected_runner_sha256)
        payloads[task] = item
        for representation in REPRESENTATIONS:
            result = item["representations"][representation]
            rows.append({
                "task": task, "task_type": item["task_type"], "metric": item["primary_metric"],
                "representation": representation,
                "test_mean": checked_score(result["test_mean"], item["primary_metric"], f"{task}/{representation}"),
                "test_std_ddof0": result["test_std_ddof0"],
                "selected_max_features": result["selected_params"]["max_features"],
                "selected_min_samples_leaf": result["selected_params"]["min_samples_leaf"],
            })

    rep_counts = {"ecfp4_wins": 0, "rdkit_descriptors_wins": 0, "ties": 0}
    for task, item in payloads.items():
        metric = item["primary_metric"]
        result = outcome(item["representations"]["ecfp4"]["test_mean"],
                         item["representations"]["rdkit_descriptors"]["test_mean"], metric)
        rep_counts["ecfp4_wins" if result > 0 else "rdkit_descriptors_wins" if result < 0 else "ties"] += 1
    rep_counts["exact_two_sided_sign_p"] = exact_sign_test(rep_counts["ecfp4_wins"], rep_counts["rdkit_descriptors_wins"])

    comparisons: dict[str, Any] = {}
    if args.comparator_dir:
        for representation in REPRESENTATIONS:
            wins = losses = ties = 0
            details = []
            for task, item in payloads.items():
                metric = item["primary_metric"]
                baseline = float(item["representations"][representation]["test_mean"])
                comparator = comparator_value(args.comparator_dir / f"results_{task}.json", task, metric)
                result = outcome(baseline, comparator, metric)
                wins += result > 0
                losses += result < 0
                ties += result == 0
                details.append({"task": task, "metric": metric, "baseline": baseline,
                                "comparator": comparator, "baseline_outcome": "win" if result > 0 else "loss" if result < 0 else "tie"})
            comparisons[representation] = {
                "inference_unit": "endpoint", "baseline_wins": wins, "baseline_losses": losses,
                "ties": ties, "exact_two_sided_sign_p": exact_sign_test(wins, losses), "details": details,
            }

    summary = {
        "schema_version": 1, "status": "PASS", "endpoint_count": len(TASKS),
        "endpoint_is_inference_unit": True,
        "representation_comparison": rep_counts,
        "optional_comparator": comparisons or None,
        "limitations": [
            "The two representations share ExtraTrees but are separately tuned on validation data.",
            "Three forest seeds quantify algorithmic randomness on one fixed scaffold partition; they are not independent data splits.",
            "No heterogeneous task metrics are numerically averaged; across-task comparisons use endpoint directions.",
        ],
    }
    csv_path = args.results_dir / "tdc_classical_baselines_summary.csv"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    atomic_text(csv_path, csv_buffer.getvalue())
    atomic_text(args.results_dir / "tdc_classical_baselines_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256()
    for path in sorted(expected_paths) + [csv_path, args.results_dir / "tdc_classical_baselines_summary.json"]:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    atomic_text(complete_path, digest.hexdigest() + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
