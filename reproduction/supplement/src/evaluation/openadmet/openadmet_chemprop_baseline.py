#!/usr/bin/env python3
"""Chemprop v2 D-MPNN baseline on the locked OpenADMET prepared splits.

Each endpoint/split trains an actual single-task Chemprop D-MPNN from scratch.
The outer test rows are never used for fitting, preprocessing, or early stopping;
the validation partition is the same cluster-held-out partition used by the
ChemRasayan benchmark runner.  Best checkpoints and per-molecule predictions
are retained with hashes for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from moljepa_benchmark_config import ENDPOINTS, PAPER_PROTOCOL


VALID_FRAC = 0.15
METHOD = "chemprop_dmpnn"
CSV_ROUNDTRIP_ATOL = 1e-15


def smiles_sha256(values) -> str:
    """Match the neural benchmark's order-insensitive split hash."""
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def float_arrays_match_csv_roundtrip(left: np.ndarray, right: np.ndarray) -> bool:
    """Accept only sub-femtounit absolute drift from CSV parse/write/parse."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    return bool(np.all(np.abs(left - right) <= CSV_ROUNDTRIP_ATOL))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def cluster_validation_indices(train_frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = train_frame.groupby("cluster_index").indices
    cluster_ids = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)
    target = max(1, round(len(train_frame) * VALID_FRAC))
    selected: list[int] = []
    count = 0
    for cluster_id in cluster_ids:
        size = len(groups[cluster_id])
        if count < target or not selected:
            selected.append(cluster_id)
            count += size
        if count >= target:
            break
    is_valid = train_frame["cluster_index"].isin(selected).to_numpy()
    if is_valid.all() or not is_valid.any():
        raise ValueError("cluster validation split produced an empty train or validation set")
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def run_cell(
    endpoint: str,
    split_name: str,
    frame: pd.DataFrame,
    data_sha: str,
    runner_sha: str,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    accelerator: str,
    num_workers: int,
) -> dict[str, Any]:
    split_index = PAPER_PROTOCOL["splits"].index(split_name) + 1
    seed = 22000 + split_index * 100
    config = {
        "schema_version": 1,
        "method": METHOD,
        "implementation": "chemprop v2 CLI, single-task D-MPNN from scratch",
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "accelerator": accelerator,
        "num_workers": num_workers,
        "pytorch_seed": seed,
        "validation_seed": 7300 + split_index,
        "selection": "Chemprop validation-loss early stopping only",
        "outer_test_policy": "never used for fitting, preprocessing, or early stopping",
        "test_targets_blinded_in_chemprop_input": True,
    }
    config_sha = stable_hash(config)
    cell_dir = output_dir / "cells" / endpoint / f"{split_name}_{METHOD}"
    metadata_path = cell_dir / "cell.json"
    prediction_path = cell_dir / "predictions.csv"
    if metadata_path.is_file() and prediction_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = Path(existing.get("checkpoint", ""))
        if (
            existing.get("data_sha256") == data_sha
            and existing.get("runner_sha256") == runner_sha
            and existing.get("config_sha256") == config_sha
            and existing.get("predictions_sha256") == sha256_file(prediction_path)
            and checkpoint.is_file()
            and existing.get("checkpoint_sha256") == sha256_file(checkpoint)
        ):
            print(f"CURRENT {endpoint} {split_name} {METHOD}", flush=True)
            return existing

    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)
    paper_train_idx = np.flatnonzero(frame[split_name].eq("train").to_numpy())
    test_idx = np.flatnonzero(frame[split_name].eq("test").to_numpy())
    paper_train = frame.iloc[paper_train_idx].reset_index(drop=True)
    inner_rel, valid_rel = cluster_validation_indices(paper_train, seed=7300 + split_index)
    inner_idx = paper_train_idx[inner_rel]
    valid_idx = paper_train_idx[valid_rel]

    assignments = np.full(len(frame), "excluded", dtype=object)
    assignments[inner_idx] = "train"
    assignments[valid_idx] = "val"
    assignments[test_idx] = "test"
    cli_targets = frame["y"].astype(float).copy()
    cli_targets.iloc[test_idx] = np.nan
    training_frame = pd.DataFrame({
        "row_index": np.arange(len(frame)),
        "smiles": frame["smiles"],
        "y": cli_targets,
        "split": assignments,
    })
    training_frame = training_frame[training_frame["split"].ne("excluded")]
    if not np.isfinite(training_frame.loc[training_frame["split"].isin(["train", "val"]), "y"]).all():
        raise RuntimeError("Chemprop train/validation targets must be finite")
    if training_frame.loc[training_frame["split"].eq("test"), "y"].notna().any():
        raise RuntimeError("outer-test targets were not blinded before Chemprop training")
    training_path = cell_dir / "train_val_test.csv"
    atomic_csv(training_path, training_frame)
    test_input_path = cell_dir / "test_smiles.csv"
    atomic_csv(test_input_path, training_frame[training_frame["split"].eq("test")][["row_index", "smiles"]])

    model_dir = cell_dir / "model"
    log_path = cell_dir / "chemprop.log"
    train_command = [
        "chemprop", "train",
        "--data-path", str(training_path),
        "--output-dir", str(model_dir),
        "--task-type", "regression",
        "--smiles-columns", "smiles",
        "--target-columns", "y",
        "--splits-column", "split",
        "--num-replicates", "1",
        "--ensemble-size", "1",
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch-size", str(batch_size),
        "--pytorch-seed", str(seed),
        "--accelerator", accelerator,
        "--num-workers", str(num_workers),
    ]
    run_command(train_command, log_path)
    checkpoints = sorted(model_dir.glob("model_*/best.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one Chemprop best.pt under {model_dir}, found {checkpoints}")
    checkpoint = checkpoints[0]
    raw_predictions = cell_dir / "raw_predictions.csv"
    predict_command = [
        "chemprop", "predict",
        "--test-path", str(test_input_path),
        "--model-paths", str(checkpoint),
        "--preds-path", str(raw_predictions),
        "--smiles-columns", "smiles",
        "--accelerator", accelerator,
        "--num-workers", str(num_workers),
    ]
    run_command(predict_command, log_path)
    raw = pd.read_csv(raw_predictions)
    if "y" not in raw.columns or "row_index" not in raw.columns:
        raise RuntimeError(f"unexpected Chemprop prediction columns: {list(raw.columns)}")
    truth = frame[["smiles", "y"]].copy()
    truth["row_index"] = np.arange(len(frame))
    predictions = raw[["row_index", "smiles", "y"]].rename(columns={"y": "y_pred"})
    predictions = predictions.merge(
        truth.rename(columns={"smiles": "truth_smiles", "y": "y_true"}),
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    if not predictions["smiles"].eq(predictions["truth_smiles"]).all():
        raise RuntimeError("Chemprop prediction rows do not match the locked prepared data")
    predictions.insert(0, "method", METHOD)
    predictions.insert(0, "split", split_name)
    predictions.insert(0, "endpoint", endpoint)
    predictions = predictions.drop(columns="truth_smiles")
    atomic_csv(prediction_path, predictions)
    metadata = {
        "schema_version": 1,
        "endpoint": endpoint,
        "split": split_name,
        "method": METHOD,
        "data_sha256": data_sha,
        "runner_sha256": runner_sha,
        "config": config,
        "config_sha256": config_sha,
        "n_train": int(len(inner_idx)),
        "n_validation": int(len(valid_idx)),
        "n_test": int(len(test_idx)),
        "train_smiles_sha256": smiles_sha256(frame.iloc[inner_idx]["smiles"]),
        "validation_smiles_sha256": smiles_sha256(frame.iloc[valid_idx]["smiles"]),
        "test_smiles_sha256": smiles_sha256(frame.iloc[test_idx]["smiles"]),
        "test_mae": float(mean_absolute_error(predictions["y_true"], predictions["y_pred"])),
        "training_table": str(training_path),
        "training_table_sha256": sha256_file(training_path),
        "test_targets_blinded_in_training_table": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "software_versions": {
            "python": platform.python_version(),
            "chemprop": importlib.metadata.version("chemprop"),
            "torch": importlib.metadata.version("torch"),
            "lightning": importlib.metadata.version("lightning"),
            "rdkit": importlib.metadata.version("rdkit"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }
    atomic_json(metadata_path, metadata)
    print(f"DONE {endpoint} {split_name} {METHOD} test={metadata['test_mae']:.6f}", flush=True)
    return metadata


def summarize(output_dir: Path, data_dir: Path) -> None:
    records = []
    max_fitted_target_abs_delta = 0.0
    max_prediction_truth_abs_delta = 0.0
    nonzero_fitted_target_differences = 0
    nonzero_prediction_truth_differences = 0
    expected_keys = {
        (endpoint.slug, split_name)
        for endpoint in ENDPOINTS for split_name in PAPER_PROTOCOL["splits"]
    }
    paths = sorted((output_dir / "cells").glob("*/*/cell.json"))
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    runner_hashes: set[str] = set()
    software_contracts: set[str] = set()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record.get("endpoint"), record.get("split"))
        if key in indexed:
            raise ValueError(f"duplicate Chemprop cell {key}")
        indexed[key] = record
        if key not in expected_keys or record.get("method") != METHOD:
            raise ValueError(f"unexpected Chemprop cell {key}")
        if record.get("data_sha256") != manifest["endpoints"][key[0]]["prepared_sha256"]:
            raise ValueError(f"{key}: prepared data hash mismatch")
        if stable_hash(record.get("config")) != record.get("config_sha256"):
            raise ValueError(f"{key}: config hash mismatch")
        if record["config"].get("test_targets_blinded_in_chemprop_input") is not True:
            raise ValueError(f"{key}: config does not require test-target blinding")
        if not record.get("test_targets_blinded_in_training_table"):
            raise ValueError(f"{key}: test-target blinding was not recorded")
        source = pd.read_csv(data_dir / f"{key[0]}.csv")
        split_index = PAPER_PROTOCOL["splits"].index(key[1]) + 1
        paper_train_idx = np.flatnonzero(source[key[1]].eq("train").to_numpy())
        test_idx = np.flatnonzero(source[key[1]].eq("test").to_numpy())
        inner_rel, valid_rel = cluster_validation_indices(
            source.iloc[paper_train_idx].reset_index(drop=True), seed=7300 + split_index
        )
        inner_idx = paper_train_idx[inner_rel]
        valid_idx = paper_train_idx[valid_rel]
        expected_hashes = (
            smiles_sha256(source.iloc[inner_idx]["smiles"]),
            smiles_sha256(source.iloc[valid_idx]["smiles"]),
            smiles_sha256(source.iloc[test_idx]["smiles"]),
        )
        observed_hashes = (
            record.get("train_smiles_sha256"),
            record.get("validation_smiles_sha256"),
            record.get("test_smiles_sha256"),
        )
        if observed_hashes != expected_hashes:
            raise ValueError(f"{key}: split hash mismatch")
        runner_hashes.add(str(record.get("runner_sha256")))
        software_contracts.add(json.dumps(record.get("software_versions"), sort_keys=True))
        cell_dir = path.parent
        prediction_path = Path(record["predictions"])
        if not prediction_path.is_file():
            prediction_path = cell_dir / "predictions.csv"
        training_path = Path(record["training_table"])
        if not training_path.is_file():
            training_path = cell_dir / "train_val_test.csv"
        checkpoint_path = Path(record["checkpoint"])
        if not checkpoint_path.is_file():
            checkpoint_path = next((cell_dir / "model").glob("model_*/best.pt"), Path())
        if record.get("predictions_sha256") != sha256_file(prediction_path):
            raise ValueError(f"{key}: prediction hash mismatch")
        if record.get("training_table_sha256") != sha256_file(training_path):
            raise ValueError(f"{key}: training-table hash mismatch")
        if not checkpoint_path.is_file() or record.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"{key}: checkpoint hash mismatch")
        training = pd.read_csv(training_path)
        required_training_columns = {"row_index", "smiles", "y", "split"}
        if set(training.columns) != required_training_columns:
            raise ValueError(f"{key}: unexpected training-table columns")
        if not training["row_index"].is_unique:
            raise ValueError(f"{key}: duplicate rows in Chemprop training input")
        expected_indices = {
            "train": set(map(int, inner_idx)),
            "val": set(map(int, valid_idx)),
            "test": set(map(int, test_idx)),
        }
        if set(training["split"]) != set(expected_indices):
            raise ValueError(f"{key}: unexpected split labels in Chemprop training input")
        for partition, indices in expected_indices.items():
            observed = set(
                training.loc[training["split"].eq(partition), "row_index"].astype(int)
            )
            if observed != indices:
                raise ValueError(f"{key}: {partition} row assignment mismatch")
        joined_training = training.merge(
            source.reset_index(names="row_index")[["row_index", "smiles", "y"]].rename(
                columns={"smiles": "source_smiles", "y": "source_y"}
            ),
            on="row_index",
            how="left",
            validate="one_to_one",
        )
        if joined_training["source_smiles"].isna().any():
            raise ValueError(f"{key}: out-of-range row index in Chemprop training input")
        if not joined_training["smiles"].eq(joined_training["source_smiles"]).all():
            raise ValueError(f"{key}: training-table SMILES differ from locked source")
        if training.loc[training["split"].eq("test"), "y"].notna().any():
            raise ValueError(f"{key}: test targets are present in Chemprop training input")
        if training.loc[training["split"].isin(["train", "val"]), "y"].isna().any():
            raise ValueError(f"{key}: missing train/validation targets")
        fitted_rows = joined_training["split"].isin(["train", "val"])
        fitted_targets = joined_training.loc[fitted_rows, "y"].to_numpy(dtype=float)
        fitted_source_targets = joined_training.loc[fitted_rows, "source_y"].to_numpy(dtype=float)
        if not float_arrays_match_csv_roundtrip(fitted_targets, fitted_source_targets):
            raise ValueError(f"{key}: train/validation targets differ from locked source")
        fitted_delta = np.abs(fitted_targets - fitted_source_targets)
        max_fitted_target_abs_delta = max(
            max_fitted_target_abs_delta, float(fitted_delta.max(initial=0.0))
        )
        nonzero_fitted_target_differences += int(np.count_nonzero(fitted_delta))
        predictions = pd.read_csv(prediction_path)
        if len(predictions) != int(record["n_test"]) or not predictions["row_index"].is_unique:
            raise ValueError(f"{key}: invalid prediction rows")
        if set(predictions["row_index"].astype(int)) != set(map(int, test_idx)):
            raise ValueError(f"{key}: predictions do not cover the exact outer-test rows")
        joined_predictions = predictions.merge(
            source.reset_index(names="row_index")[["row_index", "smiles", "y"]].rename(
                columns={"smiles": "source_smiles", "y": "source_y"}
            ),
            on="row_index",
            how="left",
            validate="one_to_one",
        )
        prediction_truth = joined_predictions["y_true"].to_numpy(dtype=float)
        prediction_source_truth = joined_predictions["source_y"].to_numpy(dtype=float)
        if (
            joined_predictions["source_smiles"].isna().any()
            or not joined_predictions["smiles"].eq(joined_predictions["source_smiles"]).all()
            or not float_arrays_match_csv_roundtrip(prediction_truth, prediction_source_truth)
        ):
            raise ValueError(f"{key}: prediction rows/truth differ from locked source")
        prediction_delta = np.abs(prediction_truth - prediction_source_truth)
        max_prediction_truth_abs_delta = max(
            max_prediction_truth_abs_delta, float(prediction_delta.max(initial=0.0))
        )
        nonzero_prediction_truth_differences += int(np.count_nonzero(prediction_delta))
        if not np.isfinite(joined_predictions[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{key}: non-finite prediction values")
        expected_counts = (len(inner_idx), len(valid_idx), len(test_idx))
        observed_counts = (
            int(record["n_train"]), int(record["n_validation"]), int(record["n_test"])
        )
        if observed_counts != expected_counts:
            raise ValueError(f"{key}: stored split counts mismatch")
        recomputed = float(mean_absolute_error(predictions["y_true"], predictions["y_pred"]))
        if not np.isclose(recomputed, float(record["test_mae"]), rtol=0.0, atol=1e-10):
            raise ValueError(f"{key}: test MAE mismatch")
        records.append({
            "endpoint": record["endpoint"],
            "split": record["split"],
            "method": record["method"],
            "n_train": record["n_train"],
            "n_validation": record["n_validation"],
            "n_test": record["n_test"],
            "test_mae": record["test_mae"],
            "data_sha256": record["data_sha256"],
            "runner_sha256": record["runner_sha256"],
            "config_sha256": record["config_sha256"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "predictions_sha256": record["predictions_sha256"],
        })
    if set(indexed) != expected_keys:
        raise ValueError(
            f"Chemprop cell set mismatch: missing={sorted(expected_keys-set(indexed))[:10]} "
            f"extra={sorted(set(indexed)-expected_keys)[:10]}"
        )
    if len(runner_hashes) != 1 or "None" in runner_hashes:
        raise ValueError("mixed or missing Chemprop runner hashes")
    if len(software_contracts) != 1:
        raise ValueError("mixed Chemprop software environments")
    cells = pd.DataFrame(records).sort_values(["endpoint", "split"])
    atomic_csv(output_dir / "cell_summary.csv", cells)
    summary = (
        cells.groupby(["endpoint", "method"], as_index=False)
        .agg(
            mean_test_mae=("test_mae", "mean"),
            std_test_mae=("test_mae", lambda values: float(np.std(values, ddof=0))),
            n_splits=("split", "nunique"),
        )
        .sort_values("endpoint")
    )
    atomic_csv(output_dir / "summary.csv", summary)
    atomic_json(output_dir / "summary.json", {
        "schema_version": 1,
        "complete_cells": int(len(cells)),
        "expected_cells": len(ENDPOINTS) * len(PAPER_PROTOCOL["splits"]),
        "status": "PASS",
        "outer_test_targets_blinded": True,
        "target_roundtrip_check": {
            "comparison": "finite arrays with rtol=0 and bounded absolute CSV round-trip drift",
            "absolute_tolerance": CSV_ROUNDTRIP_ATOL,
            "relative_tolerance": 0.0,
            "max_fitted_target_abs_delta": max_fitted_target_abs_delta,
            "max_prediction_truth_abs_delta": max_prediction_truth_abs_delta,
            "nonzero_fitted_target_differences": nonzero_fitted_target_differences,
            "nonzero_prediction_truth_differences": nonzero_prediction_truth_differences,
        },
        "cell_runner_sha256": next(iter(runner_hashes)),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        "finalizer_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "summary_csv": str(output_dir / "summary.csv"),
        "summary_csv_sha256": sha256_file(output_dir / "summary.csv"),
        "cell_summary_csv": str(output_dir / "cell_summary.csv"),
        "cell_summary_csv_sha256": sha256_file(output_dir / "cell_summary.csv"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data/moljepa_benchmarks/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/results_openadmet_baselines/chemprop"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="do not write a potentially partial summary from a concurrent shard",
    )
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output_dir, args.data_dir)
        return
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require 0 <= shard-id < num-shards")
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("endpoints", {})) != {endpoint.slug for endpoint in ENDPOINTS}:
        raise ValueError("prepared manifest does not contain exactly the 23 locked endpoints")
    unknown = set(args.only or ()) - {endpoint.slug for endpoint in ENDPOINTS}
    if unknown:
        raise ValueError(f"unknown endpoints: {sorted(unknown)}")
    endpoints = [endpoint for endpoint in ENDPOINTS if not args.only or endpoint.slug in set(args.only)]
    runner_sha = stable_hash({
        "runner": sha256_file(Path(__file__).resolve()),
        "benchmark_config": sha256_file(Path(__file__).with_name("moljepa_benchmark_config.py")),
    })
    for index, endpoint in enumerate(endpoints):
        if index % args.num_shards != args.shard_id:
            continue
        path = args.data_dir / f"{endpoint.slug}.csv"
        expected_sha = manifest["endpoints"][endpoint.slug]["prepared_sha256"]
        if sha256_file(path) != expected_sha:
            raise ValueError(f"prepared CSV hash mismatch: {path}")
        frame = pd.read_csv(path)
        required = {"smiles", "y", "cluster_index", *PAPER_PROTOCOL["splits"]}
        if required.difference(frame.columns):
            raise ValueError(f"{path} lacks {sorted(required.difference(frame.columns))}")
        for split_name in PAPER_PROTOCOL["splits"]:
            run_cell(
                endpoint.slug,
                split_name,
                frame,
                expected_sha,
                runner_sha,
                args.output_dir,
                args.epochs,
                args.patience,
                args.batch_size,
                args.accelerator,
                args.num_workers,
            )
    if not args.skip_summary:
        summarize(args.output_dir, args.data_dir)


if __name__ == "__main__":
    main()
