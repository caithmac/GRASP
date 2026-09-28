#!/usr/bin/env python3
"""Certify and transactionally publish the v2 atom-order experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import certify_atom_order_robustness as v1


SCHEMA_VERSION = 2
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_917
EXPECTED_RDKIT_DISTRIBUTION = "2025.3.6"
EXPECTED_RDKIT_MODULE = "2025.03.6"
VALIDATION_METRIC_ROLE = (
    "training provenance only; validation predictions were not persisted, so "
    "validation_mae is not independently recertified"
)
SCOPE = v1.SCOPE
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        v1.fail(f"JSON artifact is not an object: {path.name}")
    return payload


def parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        v1.fail(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} is not an exact UTC timestamp") from exc
    return parsed


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        v1.fail(f"{label} mismatch: {actual!r} != {expected!r}")


def named_file_inventory_sha256(directory: Path, names: set[str], label: str) -> str:
    observed = {path.name for path in directory.glob("results_*.json") if path.is_file()}
    require_equal(observed, names, f"{label} inventory")
    digest = hashlib.sha256()
    for name in sorted(names):
        path = directory / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def validate_runtime(runtime: Any, receipt: dict[str, Any], label: str) -> str:
    required = {
        "python", "python_implementation", "rdkit_distribution", "rdkit_module",
        "numpy", "pandas", "scipy", "torch", "torch_geometric",
        "scikit_learn", "cuda_runtime", "cuda_available", "cudnn",
        "image_reference", "image_id",
    }
    if not isinstance(runtime, dict) or required.difference(runtime):
        v1.fail(f"{label}: incomplete runtime contract")
    for key, expected in receipt["expected_runtime"].items():
        require_equal(runtime.get(key), expected, f"{label} runtime {key}")
    require_equal(receipt.get("posthoc_runtime_required"), ["cudnn"],
                  f"{label} post-hoc runtime contract")
    require_equal(runtime["image_reference"], receipt["image_reference"],
                  f"{label} image reference")
    require_equal(runtime["image_id"], receipt["image_id"], f"{label} image ID")
    if runtime["cuda_available"] is not True or not runtime["cuda_runtime"]:
        v1.fail(f"{label}: CUDA runtime was not active")
    for key in required.difference({"cuda_available"}):
        if runtime[key] is None or str(runtime[key]).strip() == "":
            v1.fail(f"{label}: empty runtime field {key}")
    return v1.stable_hash(runtime)


def validate_provenance(
    provenance: Any, *, receipt: dict[str, Any], receipt_sha: str,
    launch: dict[str, Any], marker: dict[str, Any], label: str,
) -> datetime:
    if not isinstance(provenance, dict) or provenance.get("schema_version") != SCHEMA_VERSION:
        v1.fail(f"{label}: missing v2 artifact provenance")
    expected = {
        "run_id": receipt["run_id"], "run_started_at_utc": receipt["run_started_at_utc"],
        "job_uid": receipt["job_uid"], "job_name": receipt["job_name"],
        "pod_uid": launch["pod_uid"], "pod_name": launch["pod_name"],
        "pod_namespace": launch["pod_namespace"], "shard_id": launch["shard_id"],
        "run_receipt_sha256": receipt_sha, "source_hashes": receipt["source_hashes"],
        "pod_started_at_utc": launch["pod_started_at_utc"],
        "prelaunch_manifest_sha256": receipt["prelaunch_manifest_sha256"],
        "reference_results_sha256": receipt["reference_results_sha256"],
    }
    for key, value in expected.items():
        require_equal(provenance.get(key), value, f"{label} provenance {key}")
    artifact_time = parse_utc(provenance.get("artifact_created_at_utc"),
                              f"{label} artifact timestamp")
    launch_time = parse_utc(launch["pod_started_at_utc"], f"{label} launch timestamp")
    completion_time = parse_utc(marker["completed_at_utc"], f"{label} marker timestamp")
    if artifact_time < launch_time or artifact_time > completion_time:
        v1.fail(f"{label}: artifact timestamp lies outside its pod execution interval")
    return artifact_time


def write_transactional_output(
    output_dir: Path, files: dict[str, bytes], expected_names: set[str]
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"certified output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        for name, contents in files.items():
            path = staging / name
            path.write_bytes(contents)
        observed = {path.name for path in staging.iterdir() if path.is_file()}
        if observed != expected_names:
            v1.fail("transactional output inventory mismatch")
        for name, contents in files.items():
            if hashlib.sha256((staging / name).read_bytes()).digest() != \
                    hashlib.sha256(contents).digest():
                v1.fail(f"transactional output verification failed: {name}")
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def certify_v2(
    results_dir: Path, prepared_dir: Path, reference_dir: Path, output_dir: Path,
    *, runner_source: Path, base_runner_source: Path, wrapper_source: Path,
    certifier_source: Path, v1_certifier_source: Path, prelaunch_manifest: Path,
    job_template_source: Path,
) -> dict[str, bytes]:
    for label, directory in (("results", results_dir), ("prepared", prepared_dir),
                             ("reference", reference_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {directory}")
    if output_dir.exists():
        raise FileExistsError(f"certified output already exists: {output_dir}")

    receipt_path = results_dir / "RUN_RECEIPT.json"
    receipt = load_json(receipt_path)
    receipt_sha = v1.sha256_file(receipt_path)
    deployment = load_json(prelaunch_manifest)
    deployment_sha = v1.sha256_file(prelaunch_manifest)
    expected_receipt = {
        "schema_version": SCHEMA_VERSION, "expected_shards": 8,
        "expected_endpoints": 23, "expected_splits": 69,
        "expected_permutations": 690,
        "checkpoint_sha256": v1.EXPECTED_CHECKPOINT_SHA256,
        "posthoc_runtime_required": ["cudnn"],
    }
    for key, value in expected_receipt.items():
        require_equal(receipt.get(key), value, f"run receipt {key}")
    image_match = re.fullmatch(
        r"[^@]+@(sha256:[0-9a-f]{64})", str(receipt.get("image_reference", ""))
    )
    if image_match is None:
        v1.fail("run receipt image reference is not pinned by registry digest")
    require_equal(receipt.get("image_id"), image_match.group(1),
                  "run receipt image digest identity")
    require_equal(receipt.get("prelaunch_manifest_sha256"), deployment_sha,
                  "run receipt prelaunch manifest hash")
    require_equal(receipt.get("prepared_manifest_sha256"),
                  v1.sha256_file(prepared_dir / "manifest.json"),
                  "run receipt prepared manifest hash")
    if not isinstance(receipt.get("run_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", receipt["run_id"]
    ):
        v1.fail("run receipt has an invalid immutable run_id")
    run_started = parse_utc(receipt.get("run_started_at_utc"), "run receipt start")
    source_hashes = receipt.get("source_hashes")
    source_paths = {
        "certifiable_runner_sha256": runner_source,
        "base_runner_sha256": base_runner_source,
        "wrapper_sha256": wrapper_source,
        "certifier_sha256": certifier_source,
        "v1_certifier_sha256": v1_certifier_source,
    }
    if not isinstance(source_hashes, dict) or set(source_hashes) != {
        *source_paths, "source_bundle_sha256"
    }:
        v1.fail("run receipt source-hash contract mismatch")
    for key, path in source_paths.items():
        require_equal(v1.sha256_file(path), source_hashes[key], f"pinned source {key}")
    v1.require_hash(source_hashes["source_bundle_sha256"], "source bundle")
    deployment_expected = {
        "schema_version": 1, "run_id": receipt["run_id"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "prepared_manifest_sha256": receipt["prepared_manifest_sha256"],
        "image_reference": receipt["image_reference"], "image_id": receipt["image_id"],
        "source_bundle_sha256": source_hashes["source_bundle_sha256"],
        "reference_results_sha256": receipt["reference_results_sha256"],
        "job_template_sha256": v1.sha256_file(job_template_source),
        "source_hashes": {key: source_hashes[key] for key in source_paths},
        "expected_runtime": receipt["expected_runtime"],
        "posthoc_runtime_required": receipt["posthoc_runtime_required"],
    }
    for key, value in deployment_expected.items():
        require_equal(deployment.get(key), value, f"prelaunch manifest {key}")

    launch_names = {f"launch_manifest_shard_{index}.json" for index in range(8)}
    marker_names = {f"SHARD_{index}_COMPLETE.json" for index in range(8)}
    require_equal({p.name for p in results_dir.glob("launch_manifest_shard_*.json")},
                  launch_names, "launch inventory")
    require_equal({p.name for p in results_dir.glob("SHARD_*_COMPLETE.json")},
                  marker_names, "marker inventory")
    launches: dict[int, dict[str, Any]] = {}
    markers: dict[int, dict[str, Any]] = {}
    for shard in range(8):
        launch_path = results_dir / f"launch_manifest_shard_{shard}.json"
        launch = load_json(launch_path)
        marker = load_json(results_dir / f"SHARD_{shard}_COMPLETE.json")
        for payload, label in ((launch, "launch"), (marker, "marker")):
            require_equal(payload.get("schema_version"), SCHEMA_VERSION,
                          f"shard {shard} {label} schema")
            require_equal(payload.get("run_id"), receipt["run_id"],
                          f"shard {shard} {label} run")
            require_equal(payload.get("job_uid"), receipt["job_uid"],
                          f"shard {shard} {label} job")
            require_equal(payload.get("shard_id"), shard, f"shard {shard} {label} index")
            require_equal(payload.get("run_receipt_sha256"), receipt_sha,
                          f"shard {shard} {label} receipt hash")
            require_equal(payload.get("prelaunch_manifest_sha256"), deployment_sha,
                          f"shard {shard} {label} prelaunch hash")
        require_equal(launch.get("source_hashes"), source_hashes,
                      f"shard {shard} launch source hashes")
        require_equal(launch.get("checkpoint_sha256"), v1.EXPECTED_CHECKPOINT_SHA256,
                      f"shard {shard} checkpoint")
        require_equal(launch.get("image_reference"), receipt["image_reference"],
                      f"shard {shard} image")
        require_equal(launch.get("image_id"), receipt["image_id"],
                      f"shard {shard} image ID")
        require_equal(launch.get("reference_results_sha256"),
                      receipt["reference_results_sha256"],
                      f"shard {shard} reference results hash")
        require_equal(marker.get("reference_results_sha256"),
                      receipt["reference_results_sha256"],
                      f"shard {shard} marker reference results hash")
        require_equal(marker.get("pod_uid"), launch.get("pod_uid"),
                      f"shard {shard} pod UID")
        require_equal(marker.get("pod_name"), launch.get("pod_name"),
                      f"shard {shard} pod name")
        require_equal(marker.get("launch_manifest_sha256"), v1.sha256_file(launch_path),
                      f"shard {shard} launch hash")
        launch_time = parse_utc(launch.get("pod_started_at_utc"), f"shard {shard} launch time")
        completed_time = parse_utc(marker.get("completed_at_utc"), f"shard {shard} completion time")
        if launch_time < run_started or completed_time < launch_time:
            v1.fail(f"shard {shard}: stale or reversed execution timestamps")
        launches[shard] = launch
        markers[shard] = marker

    manifest = load_json(prepared_dir / "manifest.json")
    endpoint_manifest = manifest.get("endpoints")
    require_equal(set(endpoint_manifest or {}), set(v1.EXPECTED_ENDPOINTS),
                  "prepared endpoint inventory")
    require_equal({p.name for p in prepared_dir.glob("*.csv")},
                  {f"{e}.csv" for e in v1.EXPECTED_ENDPOINTS}, "prepared CSV inventory")
    expected_results = {f"results_{endpoint}.json" for endpoint in v1.EXPECTED_ENDPOINTS}
    require_equal({p.name for p in results_dir.glob("results_*.json")},
                  expected_results, "v2 result inventory")
    require_equal({p.name for p in reference_dir.glob("results_*.json")},
                  expected_results, "reference result inventory")
    reference_results_sha = named_file_inventory_sha256(
        reference_dir, expected_results, "canonical reference result"
    )
    require_equal(reference_results_sha, receipt["reference_results_sha256"],
                  "canonical reference result digest")
    expected_cells = {
        f"{endpoint}/cells/{split}_full_mix.json"
        for endpoint in v1.EXPECTED_ENDPOINTS for split in v1.EXPECTED_SPLITS
    }
    observed_cells = {
        p.relative_to(results_dir).as_posix() for p in results_dir.glob("*/cells/*.json")
    }
    require_equal(observed_cells, expected_cells, "v2 cell inventory")

    endpoint_rows: list[dict[str, Any]] = []
    runtime_hashes: set[str] = set()
    result_hashes: dict[str, str] = {}
    cell_hashes: dict[str, str] = {}
    for endpoint_index, endpoint in enumerate(v1.EXPECTED_ENDPOINTS):
        shard = endpoint_index % 8
        launch, marker = launches[shard], markers[shard]
        expected_tasks = list(v1.EXPECTED_ENDPOINTS[shard::8])
        require_equal(marker.get("tasks"), expected_tasks, f"shard {shard} task list")
        result_path = results_dir / f"results_{endpoint}.json"
        result = load_json(result_path)
        reference = load_json(reference_dir / result_path.name)
        result_hash = v1.sha256_file(result_path)
        result_hashes[result_path.name] = result_hash
        require_equal(marker.get("result_sha256", {}).get(result_path.name), result_hash,
                      f"{endpoint} marker result hash")
        require_equal(result.get("schema_version"), SCHEMA_VERSION, f"{endpoint} schema")
        require_equal(result.get("task"), endpoint, f"{endpoint} identity")
        require_equal(result.get("checkpoint_sha256"), v1.EXPECTED_CHECKPOINT_SHA256,
                      f"{endpoint} checkpoint")
        require_equal(result.get("runner_sha256"), source_hashes["certifiable_runner_sha256"],
                      f"{endpoint} certifiable runner")
        require_equal(result.get("validation_metric_role"), VALIDATION_METRIC_ROLE,
                      f"{endpoint} validation metric role")
        validate_provenance(result.get("certifiable_run"), receipt=receipt,
                            receipt_sha=receipt_sha, launch=launch, marker=marker,
                            label=f"{endpoint} result")
        runtime_hashes.add(validate_runtime(result.get("runtime_contract"), receipt,
                                            f"{endpoint} result"))
        require_equal(set(result.get("run_configs", {})), {"full_mix"},
                      f"{endpoint} run configs")
        require_equal(result["run_configs"]["full_mix"], v1.expected_run_config(),
                      f"{endpoint} full_mix config")
        require_equal(set(result.get("method_summary", {})), {"full_mix"},
                      f"{endpoint} method summary")
        require_equal(set(result.get("splits", {})), set(v1.EXPECTED_SPLITS),
                      f"{endpoint} split inventory")
        data_path = prepared_dir / f"{endpoint}.csv"
        data_sha = v1.sha256_file(data_path)
        require_equal(endpoint_manifest[endpoint].get("prepared_sha256"), data_sha,
                      f"{endpoint} prepared hash")
        frame = pd.read_csv(data_path)
        split_evidence = []
        canonical_split_scores: list[float] = []
        for split_index, split in enumerate(v1.EXPECTED_SPLITS, start=1):
            paper_train = frame[frame[split].eq("train")].reset_index(drop=True)
            test_frame = frame[frame[split].eq("test")].reset_index(drop=True)
            inner_idx, valid_idx = v1.cluster_validation_indices(paper_train, 7300 + split_index)
            train_frame = paper_train.iloc[inner_idx].reset_index(drop=True)
            valid_frame = paper_train.iloc[valid_idx].reset_index(drop=True)
            split_record = result["splits"][split]
            cell = split_record.get("methods", {}).get("full_mix")
            reference_cell = reference.get("splits", {}).get(split, {}).get("methods", {}).get("full_mix")
            if not isinstance(cell, dict) or not isinstance(reference_cell, dict):
                v1.fail(f"{endpoint}/{split}: missing full_mix cell/reference")
            cell_path = results_dir / endpoint / "cells" / f"{split}_full_mix.json"
            require_equal(load_json(cell_path), cell, f"{endpoint}/{split} nested cell")
            cell_hashes[cell_path.relative_to(results_dir).as_posix()] = v1.sha256_file(cell_path)
            require_equal(cell.get("validation_metric_role"), VALIDATION_METRIC_ROLE,
                          f"{endpoint}/{split} validation role")
            validate_provenance(cell.get("certifiable_run"), receipt=receipt,
                                receipt_sha=receipt_sha, launch=launch, marker=marker,
                                label=f"{endpoint}/{split} cell")
            runtime_hashes.add(validate_runtime(cell.get("runtime_contract"), receipt,
                                                f"{endpoint}/{split} cell"))
            require_equal(cell.get("runner_sha256"), source_hashes["certifiable_runner_sha256"],
                          f"{endpoint}/{split} runner")
            regenerated: dict[int, list[str]] = {}

            def exact_regeneration(values: Sequence[str], seed: int) -> list[str]:
                strings = v1.randomized_atom_order_smiles(values, seed)
                regenerated[seed] = strings
                return strings

            evidence = v1.validate_cell(
                cell, reference_cell, test_frame, train_frame, valid_frame,
                endpoint=endpoint, split=split, permutation_function=exact_regeneration,
            )
            original_smiles = tuple(map(str, test_frame["smiles"].tolist()))
            require_equal(cell.get("ordered_original_test_smiles_sha256"),
                          v1.ordered_strings_sha256(original_smiles),
                          f"{endpoint}/{split} ordered original SMILES hash")
            require_equal(cell.get("ordered_original_test_rows_sha256"),
                          v1.ordered_rows_sha256(original_smiles, cell["y_true"]),
                          f"{endpoint}/{split} ordered original row hash")
            for record in cell["atom_order_robustness"]["records"]:
                seed = int(record["seed"])
                require_equal(record.get("ordered_smiles_sha256"),
                              v1.ordered_strings_sha256(regenerated[seed]),
                              f"{endpoint}/{split}/{seed} ordered permutation hash")
            split_evidence.append(evidence)
            canonical_split_scores.append(evidence.canonical_mae)
            require_equal(split_record.get("selected_method"), "full_mix",
                          f"{endpoint}/{split} selected method")
            v1.close(float(cell["validation_mae"]),
                     split_record.get("selected_validation_mae"),
                     f"{endpoint}/{split} selected validation MAE")
            v1.close(evidence.canonical_mae, split_record.get("selected_test_mae"),
                     f"{endpoint}/{split} selected test MAE")
        summary_record = result["method_summary"]["full_mix"]
        if not v1.arrays_match_one_ulp(canonical_split_scores,
                                       summary_record.get("split_test_mae", [])):
            v1.fail(f"{endpoint}: method-summary split MAEs mismatch")
        v1.close(float(np.mean(canonical_split_scores)), summary_record.get("mean_test_mae"),
                 f"{endpoint} method-summary mean")
        v1.close(float(np.std(canonical_split_scores)), summary_record.get("std_test_mae"),
                 f"{endpoint} method-summary SD")
        if not v1.arrays_match_one_ulp(canonical_split_scores,
                                       result.get("selected_split_test_mae", [])):
            v1.fail(f"{endpoint}: selected split MAEs mismatch")
        v1.close(float(np.mean(canonical_split_scores)), result.get("selected_mean_test_mae"),
                 f"{endpoint} selected mean")
        v1.close(float(np.std(canonical_split_scores)), result.get("selected_std_test_mae"),
                 f"{endpoint} selected SD")
        canonical = statistics.fmean(item.canonical_mae for item in split_evidence)
        permuted = statistics.fmean(item.mean_permuted_mae for item in split_evidence)
        endpoint_rows.append({
            "endpoint": endpoint, "family": result["family"],
            "canonical_mae": canonical, "mean_permuted_mae": permuted,
            "mae_change": permuted - canonical,
            "mean_prediction_sd": statistics.fmean(
                item.mean_prediction_sd for item in split_evidence
            ),
            "maximum_prediction_delta": max(
                item.maximum_prediction_delta for item in split_evidence
            ),
            "test_rows_across_splits": sum(item.test_rows for item in split_evidence),
            "mean_unique_serializations": statistics.fmean(
                item.mean_unique_serializations for item in split_evidence
            ),
            "fraction_molecules_with_multiple_serializations": statistics.fmean(
                item.fraction_molecules_with_multiple_serializations for item in split_evidence
            ),
        })

    if len(runtime_hashes) != 1:
        v1.fail("artifacts report mixed runtime contracts")
    changes = np.asarray([row["mae_change"] for row in endpoint_rows], dtype=float)
    ci_low, ci_high = v1.bootstrap_mean_ci(changes, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    prediction_sd = np.asarray([row["mean_prediction_sd"] for row in endpoint_rows])
    q1, q3 = np.quantile(prediction_sd, [0.25, 0.75], method="linear")
    analysis = {
        "analysis_unit": "23 endpoint means; each endpoint first averages three locked splits",
        "macro_canonical_mae": statistics.fmean(r["canonical_mae"] for r in endpoint_rows),
        "macro_mean_permuted_mae": statistics.fmean(r["mean_permuted_mae"] for r in endpoint_rows),
        "mean_endpoint_mae_change": float(changes.mean()),
        "median_endpoint_mae_change": float(np.median(changes)),
        "bootstrap_mean_change_ci_95": [ci_low, ci_high],
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "method": "paired endpoint percentile bootstrap"},
        "endpoints_degraded": int(np.sum(changes > 0)),
        "endpoints_improved": int(np.sum(changes < 0)),
        "endpoints_tied": int(np.sum(changes == 0)),
        "exact_wilcoxon": v1.exact_wilcoxon_two_sided(changes),
        "exact_sign_test": v1.exact_sign_test_two_sided(changes),
        "multiplicity": "single prespecified atom-order contrast; no multiplicity adjustment",
        "median_endpoint_mean_prediction_sd": float(np.median(prediction_sd)),
        "endpoint_mean_prediction_sd_iqr": [float(q1), float(q3)],
        "worst_maximum_prediction_deviation": max(
            row["maximum_prediction_delta"] for row in endpoint_rows
        ),
    }
    validation = {
        "run_id": receipt["run_id"], "job_uid": receipt["job_uid"],
        "run_started_at_utc": receipt["run_started_at_utc"],
        "run_receipt_sha256": receipt_sha,
        "prelaunch_manifest_sha256": deployment_sha,
        "endpoints": 23, "splits": 69, "permutations": 690,
        "launch_manifests": 8, "completion_markers": 8,
        "checkpoint_sha256": v1.EXPECTED_CHECKPOINT_SHA256,
        "source_hashes": source_hashes,
        "runtime_contract_sha256": next(iter(runtime_hashes)),
        "result_inventory_sha256": v1.stable_hash(result_hashes),
        "cell_inventory_sha256": v1.stable_hash(cell_hashes),
        "prepared_manifest_sha256": v1.sha256_file(prepared_dir / "manifest.json"),
        "reference_results_sha256": reference_results_sha,
        "validation_mae_boundary": VALIDATION_METRIC_ROLE,
    }
    summary = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
               "analysis": analysis, "validation": validation,
               "endpoint_records": endpoint_rows}
    anonymous = summary
    md = ["# Certified v2 atom-order robustness", "", f"Scope: {SCOPE}.", "",
          f"Run `{receipt['run_id']}` passed the immutable 23-endpoint, 69-cell, "
          "690-permutation artifact contract.", "",
          f"Canonical macro MAE was {analysis['macro_canonical_mae']:.6f}; mean permuted "
          f"macro MAE was {analysis['macro_mean_permuted_mae']:.6f}. The paired endpoint "
          f"mean change was {analysis['mean_endpoint_mae_change']:+.6f} "
          f"(95% bootstrap CI {ci_low:+.6f} to {ci_high:+.6f}; "
          f"{BOOTSTRAP_REPLICATES:,} resamples, seed {BOOTSTRAP_SEED}).", "",
          "Validation MAE is retained only as training provenance because validation "
          "predictions were not persisted; it is not independently recertified.", "",
          "| Endpoint | Canonical MAE | Mean permuted MAE | Change | Mean prediction SD | Max deviation |",
          "|---|---:|---:|---:|---:|---:|"]
    for row in endpoint_rows:
        md.append(f"| {row['endpoint']} | {row['canonical_mae']:.6f} | "
                  f"{row['mean_permuted_mae']:.6f} | {row['mae_change']:+.6f} | "
                  f"{row['mean_prediction_sd']:.6f} | {row['maximum_prediction_delta']:.6f} |")
    files = {
        "summary.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        "summary_anonymous.json": (json.dumps(anonymous, indent=2, sort_keys=True) + "\n").encode(),
        "endpoint_atom_order_robustness.csv": csv_bytes(endpoint_rows, list(endpoint_rows[0])),
        "RESULTS.md": ("\n".join(md) + "\n").encode(),
    }
    for name, contents in files.items():
        if name != "summary.json":
            v1.validate_anonymous_text(contents.decode("utf-8"))
    write_transactional_output(output_dir, files, set(files))
    return files


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runner-source", type=Path,
                        default=root / "finetune_moljepa_atom_order_certifiable_v2.py")
    parser.add_argument("--base-runner-source", type=Path,
                        default=root / "finetune_moljepa_benchmarks.py")
    parser.add_argument("--wrapper-source", type=Path,
                        default=root / "run_final_phase4_atom_order_audit_v2.sh")
    parser.add_argument("--certifier-source", type=Path, default=Path(__file__).resolve())
    parser.add_argument("--v1-certifier-source", type=Path,
                        default=Path(__file__).resolve().with_name(
                            "certify_atom_order_robustness.py"
                        ))
    parser.add_argument("--prelaunch-manifest", type=Path,
                        default=root / "atom_order_v2_prelaunch_manifest.json")
    parser.add_argument("--job-template-source", type=Path,
                        default=root / "final_phase4_atom_order_job_v2.template.yml")
    args = parser.parse_args()
    files = certify_v2(
        args.results_dir, args.prepared_dir, args.reference_dir, args.output_dir,
        runner_source=args.runner_source, base_runner_source=args.base_runner_source,
        wrapper_source=args.wrapper_source, certifier_source=args.certifier_source,
        v1_certifier_source=args.v1_certifier_source,
        prelaunch_manifest=args.prelaunch_manifest,
        job_template_source=args.job_template_source,
    )
    print(json.dumps({name: hashlib.sha256(contents).hexdigest()
                      for name, contents in sorted(files.items())}, indent=2))


if __name__ == "__main__":
    main()
