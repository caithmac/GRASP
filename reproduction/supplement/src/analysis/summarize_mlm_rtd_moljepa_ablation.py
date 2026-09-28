#!/usr/bin/env python3
"""Summarize the paired MLM-S2 versus RTD-25%-S2 MolJEPA-style ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path


MODELS = ("mlm_s2", "rtd25_s2")
METHODS = ("frozen_mix", "full_mix")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.results_dir
    records = {}
    result_hashes = {}
    checkpoint_hashes = {model: set() for model in MODELS}
    runner_hashes = set()
    for model in MODELS:
        files = sorted((root / model).glob("results_*.json"))
        if len(files) != 23:
            raise ValueError(f"{model}: expected 23 result files, found {len(files)}")
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            if set(data.get("method_summary", {})) != set(METHODS):
                raise ValueError(f"{path}: ablation must use exactly {METHODS}")
            if data["task"] in {task for seen_model, task in records if seen_model == model}:
                raise ValueError(f"{model}: duplicate task {data['task']}")
            records[(model, data["task"])] = data
            result_hashes[f"{model}/{path.name}"] = file_sha256(path)
            checkpoint_hashes[model].add(data["checkpoint_sha256"])
            runner_hashes.add(data["runner_sha256"])

    if any(len(values) != 1 for values in checkpoint_hashes.values()):
        raise ValueError(f"each objective must use exactly one checkpoint: {checkpoint_hashes}")
    if checkpoint_hashes[MODELS[0]] == checkpoint_hashes[MODELS[1]]:
        raise ValueError("MLM and RTD result sets use the same checkpoint hash")
    if len(runner_hashes) != 1:
        raise ValueError(f"result files use multiple runner hashes: {runner_hashes}")

    launch_paths = sorted(root.glob("launch_manifest_shard_*.json"))
    if len(launch_paths) != 12:
        raise ValueError(f"expected 12 launch manifests, found {len(launch_paths)}")
    launch_records = [json.loads(path.read_text(encoding="utf-8")) for path in launch_paths]
    expected_shards = set(range(12))
    if {record["shard_id"] for record in launch_records} != expected_shards:
        raise ValueError("launch manifest shard ids are incomplete or duplicated")
    for record in launch_records:
        if record["runner_sha256"] not in runner_hashes:
            raise ValueError("launch/result runner hash mismatch")
        for model in MODELS:
            if record["checkpoints"][model]["sha256"] not in checkpoint_hashes[model]:
                raise ValueError(f"launch/result checkpoint mismatch for {model}")
    if len({record["wrapper_sha256"] for record in launch_records}) != 1:
        raise ValueError("launch manifests disagree on wrapper hash")
    if len({record["job_yaml_sha256"] for record in launch_records}) != 1:
        raise ValueError("launch manifests disagree on job YAML hash")
    if len({record.get("source_bundle_sha256") for record in launch_records}) != 1:
        raise ValueError("launch manifests disagree on source bundle hash")
    audit_path = root / "objective_checkpoint_audit.json"
    if not audit_path.is_file():
        raise ValueError("missing objective checkpoint tensor audit")
    audit_sha = file_sha256(audit_path)
    if {record.get("checkpoint_audit_sha256") for record in launch_records} != {audit_sha}:
        raise ValueError("launch manifests disagree with checkpoint tensor audit")
    checkpoint_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        checkpoint_audit.get("mlm_file_sha256")
        != next(iter(checkpoint_hashes["mlm_s2"]))
        or checkpoint_audit.get("rtd_file_sha256")
        != next(iter(checkpoint_hashes["rtd25_s2"]))
    ):
        raise ValueError("checkpoint tensor audit does not describe the result checkpoints")
    if (
        checkpoint_audit.get("mlm_canonical_tensor_sha256")
        == checkpoint_audit.get("rtd_canonical_tensor_sha256")
        or int(checkpoint_audit.get("parameter_tensors", 0)) <= 0
        or int(checkpoint_audit.get("tensors_differing", 0)) <= 0
        or float(checkpoint_audit.get("global_relative_l2_delta", 0.0)) <= 0.0
    ):
        raise ValueError("checkpoint tensor audit does not prove distinct encoders")

    task_sets = {
        model: {task for seen_model, task in records if seen_model == model}
        for model in MODELS
    }
    if task_sets[MODELS[0]] != task_sets[MODELS[1]]:
        raise ValueError("MLM and RTD task sets differ")
    tasks = sorted(task_sets[MODELS[0]])
    rows = []
    for task in tasks:
        mlm = records[("mlm_s2", task)]
        rtd = records[("rtd25_s2", task)]
        if mlm["family"] != rtd["family"]:
            raise ValueError(f"{task}: family mismatch")
        if mlm["data_protocol"] != rtd["data_protocol"]:
            raise ValueError(f"{task}: data protocol mismatch")
        if mlm["run_configs"] != rtd["run_configs"]:
            raise ValueError(f"{task}: downstream configuration mismatch")
        for method in METHODS:
            mlm_scores = mlm["method_summary"][method]["split_test_mae"]
            rtd_scores = rtd["method_summary"][method]["split_test_mae"]
            if len(mlm_scores) != 3 or len(rtd_scores) != 3:
                raise ValueError(f"{task}/{method}: expected three paired test splits")
            for split_name in ("split1", "split2", "split3"):
                mlm_cell = mlm["splits"][split_name]["methods"][method]
                rtd_cell = rtd["splits"][split_name]["methods"][method]
                for key in (
                    "data_sha256", "seed", "run_config", "train_rows", "validation_rows",
                    "test_rows", "train_smiles_sha256", "validation_smiles_sha256",
                    "test_smiles_sha256",
                ):
                    if mlm_cell[key] != rtd_cell[key]:
                        raise ValueError(f"{task}/{split_name}/{method}: paired field mismatch: {key}")
                if mlm_cell["y_true"] != rtd_cell["y_true"]:
                    raise ValueError(f"{task}/{split_name}/{method}: paired test labels differ")
                for label, cell in (("MLM", mlm_cell), ("RTD", rtd_cell)):
                    if len(cell["y_true"]) != len(cell["y_pred"]) or not cell["y_true"]:
                        raise ValueError(f"{task}/{split_name}/{method}/{label}: invalid predictions")
                    recomputed = sum(abs(float(a) - float(b)) for a, b in zip(cell["y_true"], cell["y_pred"])) / len(cell["y_true"])
                    if not math.isclose(recomputed, float(cell["test_mae"]), rel_tol=0.0, abs_tol=1e-9):
                        raise ValueError(f"{task}/{split_name}/{method}/{label}: stored MAE mismatch")
            delta = (sum(mlm_scores) - sum(rtd_scores)) / 3
            winner = "Tie" if math.isclose(delta, 0.0, abs_tol=1e-12) else ("RTD-25%-S2" if delta > 0 else "MLM-S2")
            rows.append({
                "method": method,
                "family": mlm["family"],
                "task": task,
                "mlm_s2_mean_mae": statistics.fmean(mlm_scores),
                "mlm_s2_sd_mae": statistics.pstdev(mlm_scores),
                "rtd25_s2_mean_mae": statistics.fmean(rtd_scores),
                "rtd25_s2_sd_mae": statistics.pstdev(rtd_scores),
                "mlm_minus_rtd_mae": delta,
                "winner": winner,
                "mlm_split_mae": json.dumps(mlm_scores),
                "rtd_split_mae": json.dumps(rtd_scores),
                "exact_table3_split_reproduction": mlm["exact_table3_split_reproduction"],
            })

    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "mlm_vs_rtd_moljepa23.csv"
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_csv, csv_path)

    method_summaries = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        deltas = [row["mlm_minus_rtd_mae"] for row in selected]
        rng = random.Random(20260917)
        bootstrap = sorted(
            statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(10000)
        )
        nonzero = [value for value in deltas if not math.isclose(value, 0.0, abs_tol=1e-12)]
        rtd_wins = sum(value > 0 for value in nonzero)
        n = len(nonzero)
        tail = sum(math.comb(n, k) for k in range(0, min(rtd_wins, n-rtd_wins) + 1)) / (2 ** n) if n else 1.0
        sign_p = min(1.0, 2.0 * tail)
        method_summaries[method] = {
            "endpoints": len(selected),
            "rtd_wins": sum(row["winner"] == "RTD-25%-S2" for row in selected),
            "mlm_wins": sum(row["winner"] == "MLM-S2" for row in selected),
            "ties": sum(row["winner"] == "Tie" for row in selected),
            "mean_endpoint_delta_mlm_minus_rtd": statistics.fmean(deltas),
            "bootstrap_95pct_ci": [bootstrap[249], bootstrap[9749]],
            "paired_sign_test_two_sided_p": sign_p,
        }
    summary = {
        "comparison": "MLM-r2-S2 versus RTD-25%-S2",
        "downstream_methods": list(METHODS),
        "paired_splits_per_endpoint": 3,
        "endpoints": len(tasks),
        "method_summaries": method_summaries,
        "checkpoint_sha256": {
            model: next(iter(values)) for model, values in checkpoint_hashes.items()
        },
        "runner_sha256": next(iter(runner_hashes)),
        "wrapper_sha256": launch_records[0]["wrapper_sha256"],
        "job_yaml_sha256": launch_records[0]["job_yaml_sha256"],
        "source_bundle_sha256": launch_records[0]["source_bundle_sha256"],
        "checkpoint_audit_sha256": audit_sha,
        "checkpoint_tensor_audit": checkpoint_audit,
        "launch_manifests": len(launch_records),
        "result_file_sha256": result_hashes,
        "split_caveat": "MolJEPA-style reconstructed Taylor-Butina splits; not authors' unpublished exact SDF assignments",
    }
    atomic_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")

    lines = [
        "# MLM versus RTD on all 23 MolJEPA-style endpoints",
        "",
        "| Method | Family | Endpoint | MLM-S2 MAE | RTD-25%-S2 MAE | MLM − RTD | Winner |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['family']} | {row['task']} | "
            f"{row['mlm_s2_mean_mae']:.3f} ± {row['mlm_s2_sd_mae']:.3f} | "
            f"{row['rtd25_s2_mean_mae']:.3f} ± {row['rtd25_s2_sd_mae']:.3f} | "
            f"{row['mlm_minus_rtd_mae']:+.3f} | {row['winner']} |"
        )
    lines += [
        "",
        "Frozen mixing diagnoses representation quality without encoder updates; full mixing tests "
        "end-to-end adaptation. Positive MLM − RTD values favor RTD. Summary JSON reports "
        "paired endpoint bootstrap intervals and two-sided sign tests for each method.",
        "Both objectives use paired seeds/splits. "
        "The public-source row totals match MolJEPA for ExpansionRx and Biogen, but the split "
        "assignments are deterministic reconstructions because the authors' exact SDF files are unavailable.",
    ]
    atomic_text(out / "RESULTS.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
