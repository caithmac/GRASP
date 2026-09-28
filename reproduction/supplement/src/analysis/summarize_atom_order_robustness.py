#!/usr/bin/env python3
"""Validate and summarize atom-renumbering robustness from OpenADMET runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

from moljepa_benchmark_config import ENDPOINTS


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def mae(y_true, y_pred) -> float:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("invalid prediction arrays")
    return sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / len(y_true)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("results_*.json"))
    if len(files) != 23:
        raise ValueError(f"expected 23 result files, found {len(files)}")
    rows = []
    endpoint_records = {}
    checkpoint_hashes = set()
    runner_hashes = set()
    expected_tasks = {endpoint.slug for endpoint in ENDPOINTS}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_hashes.add(data["checkpoint_sha256"])
        runner_hashes.add(data["runner_sha256"])
        if set(data["method_summary"]) != {"full_mix"}:
            raise ValueError(f"{path}: expected only full_mix")
        task = data["task"]
        if task not in expected_tasks or task in endpoint_records:
            raise ValueError(f"{path}: unexpected or duplicate task {task}")
        split_rows = []
        for split_name in ("split1", "split2", "split3"):
            cell = data["splits"][split_name]["methods"]["full_mix"]
            audit = cell.get("atom_order_robustness")
            if not audit or audit.get("permutations") != 10 or len(audit.get("records", [])) != 10:
                raise ValueError(f"{task}/{split_name}: incomplete atom-order audit")
            canonical = mae(cell["y_true"], cell["y_pred"])
            if not math.isclose(canonical, float(cell["test_mae"]), abs_tol=1e-9):
                raise ValueError(f"{task}/{split_name}: canonical MAE mismatch")
            permuted_maes = []
            all_max_delta = 0.0
            permutation_predictions = []
            expected_indices = set(range(10))
            observed_indices = {int(record["permutation_index"]) for record in audit["records"]}
            if observed_indices != expected_indices:
                raise ValueError(f"{task}/{split_name}: permutation indices are incomplete")
            changed_total = 0
            for record in audit["records"]:
                observed = mae(cell["y_true"], record["y_pred"])
                if not math.isclose(observed, float(record["test_mae"]), abs_tol=1e-9):
                    raise ValueError(f"{task}/{split_name}: permuted MAE mismatch")
                permuted_maes.append(observed)
                deltas = [
                    abs(float(permuted) - float(canonical_prediction))
                    for permuted, canonical_prediction in zip(record["y_pred"], cell["y_pred"])
                ]
                observed_mean_delta = statistics.fmean(deltas)
                observed_max_delta = max(deltas)
                if not math.isclose(
                    observed_mean_delta,
                    float(record["mean_absolute_prediction_delta"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ) or not math.isclose(
                    observed_max_delta,
                    float(record["maximum_absolute_prediction_delta"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(f"{task}/{split_name}: permutation delta mismatch")
                all_max_delta = max(all_max_delta, observed_max_delta)
                changed_fraction = float(record.get("fraction_changed_smiles", -1.0))
                if not 0.0 <= changed_fraction <= 1.0:
                    raise ValueError(f"{task}/{split_name}: invalid changed-SMILES fraction")
                changed = int(record.get("changed_smiles", -1))
                if changed < 0 or not math.isclose(
                    changed_fraction,
                    changed / len(cell["y_true"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"{task}/{split_name}: changed-SMILES count mismatch")
                expected_seed = int(cell["seed"]) * 1000 + int(record["permutation_index"])
                if int(record["seed"]) != expected_seed:
                    raise ValueError(f"{task}/{split_name}: permutation seed mismatch")
                changed_total += changed
                permutation_predictions.append([float(value) for value in record["y_pred"]])
            if changed_total == 0:
                raise ValueError(f"{task}/{split_name}: atom renumbering changed no SMILES")
            per_molecule_sd = [float(value) for value in audit["per_molecule_prediction_sd"]]
            per_molecule_max = [float(value) for value in audit["per_molecule_maximum_absolute_delta"]]
            if len(per_molecule_sd) != len(cell["y_true"]) or len(per_molecule_max) != len(cell["y_true"]):
                raise ValueError(f"{task}/{split_name}: molecule robustness length mismatch")
            recomputed_sd = [
                statistics.pstdev(predictions)
                for predictions in zip(*permutation_predictions)
            ]
            recomputed_max = [
                max(
                    abs(prediction - float(canonical_prediction))
                    for prediction in predictions
                )
                for predictions, canonical_prediction in zip(
                    zip(*permutation_predictions), cell["y_pred"]
                )
            ]
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(per_molecule_sd, recomputed_sd)
            ) or any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(per_molecule_max, recomputed_max)
            ):
                raise ValueError(f"{task}/{split_name}: aggregate robustness arrays mismatch")
            if not math.isclose(
                statistics.fmean(permuted_maes),
                float(audit["mean_permuted_test_mae"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                statistics.fmean(permuted_maes) - canonical,
                float(audit["mean_mae_change"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{task}/{split_name}: aggregate robustness MAE mismatch")
            row = {
                "family": data["family"],
                "task": task,
                "split": split_name,
                "test_rows": len(cell["y_true"]),
                "canonical_mae": canonical,
                "mean_permuted_mae": statistics.fmean(permuted_maes),
                "mae_change": statistics.fmean(permuted_maes) - canonical,
                "median_prediction_sd": statistics.median(per_molecule_sd),
                "mean_prediction_sd": statistics.fmean(per_molecule_sd),
                "maximum_prediction_delta": max(per_molecule_max),
                "maximum_recorded_prediction_delta": all_max_delta,
            }
            rows.append(row)
            split_rows.append(row)
        endpoint_records[task] = {
            "family": data["family"],
            "canonical_mae": statistics.fmean(row["canonical_mae"] for row in split_rows),
            "mean_permuted_mae": statistics.fmean(row["mean_permuted_mae"] for row in split_rows),
            "mae_change": statistics.fmean(row["mae_change"] for row in split_rows),
            "mean_prediction_sd": statistics.fmean(row["mean_prediction_sd"] for row in split_rows),
            "maximum_prediction_delta": max(row["maximum_prediction_delta"] for row in split_rows),
        }

    if len(checkpoint_hashes) != 1 or len(runner_hashes) != 1:
        raise ValueError("robustness package mixes checkpoints or runners")
    if set(endpoint_records) != expected_tasks:
        raise ValueError(
            f"task set mismatch: missing={sorted(expected_tasks-set(endpoint_records))} "
            f"extra={sorted(set(endpoint_records)-expected_tasks)}"
        )
    checkpoint_sha = next(iter(checkpoint_hashes))
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("unexpected checkpoint hash")
    launch_paths = sorted(args.results_dir.glob("launch_manifest_shard_*.txt"))
    if len(launch_paths) != 8:
        raise ValueError(f"expected 8 launch manifests, found {len(launch_paths)}")
    expected_launch_names = {f"launch_manifest_shard_{index}.txt" for index in range(8)}
    if {path.name for path in launch_paths} != expected_launch_names:
        raise ValueError("launch manifest shard ids are incomplete or duplicated")
    launch_contracts = {
        tuple(path.read_text(encoding="utf-8").strip().split()) for path in launch_paths
    }
    if len(launch_contracts) != 1:
        raise ValueError("atom-order shards used mixed launch contracts")
    launch_contract = next(iter(launch_contracts))
    if len(launch_contract) != 4:
        raise ValueError("invalid atom-order launch manifest")
    if launch_contract[0] != checkpoint_sha or launch_contract[1] != next(iter(runner_hashes)):
        raise ValueError("atom-order launch/result provenance mismatch")
    out = args.results_dir / "atom_order_comparison"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "split_summary.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)

    summary = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha,
        "runner_sha256": next(iter(runner_hashes)),
        "wrapper_sha256": launch_contract[2],
        "source_bundle_sha256": launch_contract[3],
        "endpoints": 23,
        "splits": len(rows),
        "permutations_per_split": 10,
        "downstream_model_scope": (
            "fresh deterministic full_mix replicate trained from the fixed Phase-4 encoder; "
            "not the previously reported downstream head because that head was not persisted"
        ),
        "mean_endpoint_canonical_mae": statistics.fmean(
            record["canonical_mae"] for record in endpoint_records.values()
        ),
        "mean_endpoint_permuted_mae": statistics.fmean(
            record["mean_permuted_mae"] for record in endpoint_records.values()
        ),
        "mean_endpoint_mae_change": statistics.fmean(
            record["mae_change"] for record in endpoint_records.values()
        ),
        "median_endpoint_mae_change": statistics.median(
            record["mae_change"] for record in endpoint_records.values()
        ),
        "endpoints_improved_under_permutation": sum(
            record["mae_change"] < 0 for record in endpoint_records.values()
        ),
        "endpoints_degraded_under_permutation": sum(
            record["mae_change"] > 0 for record in endpoint_records.values()
        ),
        "maximum_prediction_delta": max(
            record["maximum_prediction_delta"] for record in endpoint_records.values()
        ),
        "endpoint_records": endpoint_records,
    }
    atomic_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Atom-order robustness",
        "",
        "Each held-out molecule was evaluated under 10 deterministic RDKit atom renumberings. "
        "Canonical isomeric identity was asserted before inference.",
        "The audited downstream model is a fresh deterministic full-mixing replicate from the "
        "fixed Phase-4 encoder, not the earlier reported head (which was not persisted).",
        "",
        "| Endpoint | Canonical MAE | Permuted MAE | Change | Mean prediction SD | Max prediction change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, record in sorted(endpoint_records.items()):
        lines.append(
            f"| {task} | {record['canonical_mae']:.4f} | {record['mean_permuted_mae']:.4f} | "
            f"{record['mae_change']:+.4f} | {record['mean_prediction_sd']:.4f} | "
            f"{record['maximum_prediction_delta']:.4f} |"
        )
    atomic_text(out / "RESULTS.md", "\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "mean_endpoint_canonical_mae", "mean_endpoint_permuted_mae",
        "mean_endpoint_mae_change", "maximum_prediction_delta",
    )}, indent=2))


if __name__ == "__main__":
    main()
