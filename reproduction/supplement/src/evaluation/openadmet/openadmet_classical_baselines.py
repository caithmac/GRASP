#!/usr/bin/env python3
"""Leakage-safe classical baselines for the 23 reconstructed OpenADMET tasks.

The script consumes the exact prepared CSVs used by
``finetune_moljepa_benchmarks.py``.  For each outer split it recreates that
runner's cluster-held-out validation partition, selects hyperparameters using
validation MAE only, retrains the selected configuration on the same inner
training rows, and evaluates the untouched outer test set.  Validation labels
are therefore not used to fit the final primary model, matching the neural
runner's label budget.

Every prediction and provenance-bearing cell is retained.  Existing cells are
accepted only when their data, runner, configuration, and seed hashes match.
No model is silently substituted when an optional dependency is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import lightgbm
import numpy as np
import pandas as pd
import rdkit
import sklearn
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from moljepa_benchmark_config import ENDPOINTS, PAPER_PROTOCOL


METHODS = (
    "dummy_median",
    "ecfp4_rf",
    "ecfp4_lgbm",
    "rdkit_desc_rf",
    "rdkit_desc_lgbm",
)
VALID_FRAC = 0.15
FP_RADIUS = 2
FP_SIZE = 2048


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict[str, Any]


GRIDS = {
    # The sample median, rather than mean, minimizes absolute error.
    "dummy_median": (Candidate("training_median", {}),),
    "ecfp4_rf": tuple(
        Candidate(
            f"rf_mf-{max_features}_leaf-{min_leaf}",
            {
                "n_estimators": 500,
                "max_features": max_features,
                "min_samples_leaf": min_leaf,
            },
        )
        for max_features in ("sqrt", 0.33)
        for min_leaf in (1, 2, 4)
    ),
    "rdkit_desc_rf": tuple(
        Candidate(
            f"rf_mf-{max_features}_leaf-{min_leaf}",
            {
                "n_estimators": 500,
                "max_features": max_features,
                "min_samples_leaf": min_leaf,
            },
        )
        for max_features in ("sqrt", 0.33)
        for min_leaf in (1, 2, 4)
    ),
    "ecfp4_lgbm": tuple(
        Candidate(
            f"lgbm_leaves-{leaves}_leaf-{min_leaf}_lr-{lr}",
            {
                "n_estimators": 600,
                "learning_rate": lr,
                "num_leaves": leaves,
                "min_child_samples": min_leaf,
                "subsample": 0.9,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        )
        for leaves in (31, 63)
        for min_leaf in (10, 20)
        for lr in (0.03, 0.06)
    ),
    "rdkit_desc_lgbm": tuple(
        Candidate(
            f"lgbm_leaves-{leaves}_leaf-{min_leaf}_lr-{lr}",
            {
                "n_estimators": 600,
                "learning_rate": lr,
                "num_leaves": leaves,
                "min_child_samples": min_leaf,
                "subsample": 0.9,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        )
        for leaves in (31, 63)
        for min_leaf in (10, 20)
        for lr in (0.03, 0.06)
    ),
}


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


def smiles_sha256(values) -> str:
    """Match ``finetune_moljepa_benchmarks.py`` split hashing exactly."""
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cluster_validation_indices(train_frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Byte-for-byte algorithmic match to the neural benchmark runner."""
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


def ecfp4(smiles: list[str], *, counts: bool) -> np.ndarray:
    """Return radius-2 Morgan features with an explicit count/bit contract.

    The RF arm uses the 2,048-bit vector in the released Mol-JEPA RF helper.
    The LightGBM arm uses 2,048-bin counts, matching the Pat Walters
    ExpansionRx/Biogen benchmark implementation bundled with this repository.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    values = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"invalid prepared SMILES: {smi}")
        values.append(
            generator.GetCountFingerprintAsNumPy(mol)
            if counts else generator.GetFingerprintAsNumPy(mol)
        )
    return np.asarray(values, dtype=np.float32)


def rdkit_descriptors(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [name for name, _ in Descriptors._descList]
    functions = [function for _, function in Descriptors._descList]
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"invalid prepared SMILES: {smi}")
        row = []
        for function in functions:
            try:
                value = float(function(mol))
            except Exception:
                value = math.nan
            row.append(value if math.isfinite(value) else math.nan)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64), names


def make_model(method: str, params: dict[str, Any], seed: int):
    if method in {"ecfp4_rf", "rdkit_desc_rf"}:
        return RandomForestRegressor(
            **params, random_state=seed, n_jobs=-1, criterion="squared_error"
        )
    if method in {"ecfp4_lgbm", "rdkit_desc_lgbm"}:
        return LGBMRegressor(
            **params,
            objective="regression_l1",
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1,
        )
    raise ValueError(method)


def fit_predict(
    method: str,
    candidate: Candidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "dummy_median":
        return np.full(len(x_eval), float(np.median(y_train))), {"imputer": None}
    preprocessing: dict[str, Any] = {"imputer": None}
    if method.startswith("rdkit_desc_"):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(x_train)
        x_eval = imputer.transform(x_eval)
        preprocessing["imputer"] = {
            "strategy": "median",
            "fit_rows": int(len(y_train)),
            "n_features": int(x_train.shape[1]),
        }
    model = make_model(method, candidate.params, seed)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_eval), dtype=float), preprocessing


def run_cell(
    *,
    endpoint: str,
    split_name: str,
    method: str,
    frame: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str] | None,
    data_sha: str,
    runner_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    split_index = PAPER_PROTOCOL["splits"].index(split_name) + 1
    seed = 12000 + split_index * 100 + METHODS.index(method)
    paper_train_idx = np.flatnonzero(frame[split_name].eq("train").to_numpy())
    test_idx = np.flatnonzero(frame[split_name].eq("test").to_numpy())
    paper_train = frame.iloc[paper_train_idx].reset_index(drop=True)
    inner_rel, valid_rel = cluster_validation_indices(paper_train, seed=7300 + split_index)
    inner_idx = paper_train_idx[inner_rel]
    valid_idx = paper_train_idx[valid_rel]
    y = frame["y"].to_numpy(dtype=float)
    config = {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "validation_seed": 7300 + split_index,
        "selection_metric": "validation_mae",
        "final_fit_policy": (
            "selected configuration retrained on inner-train only; validation labels "
            "are used for selection but not fitted model weights"
        ),
        "fingerprint": (
            {
                "kind": "Morgan bit vector" if method == "ecfp4_rf" else "Morgan count",
                "radius": FP_RADIUS,
                "n_bits": FP_SIZE,
            }
            if method.startswith("ecfp4_") else None
        ),
        "descriptor_names": feature_names if method.startswith("rdkit_desc_") else None,
        "candidates": [asdict(candidate) for candidate in GRIDS[method]],
    }
    config_sha = stable_hash(config)
    cell_path = output_dir / "cells" / endpoint / f"{split_name}_{method}.json"
    prediction_path = output_dir / "predictions" / endpoint / f"{split_name}_{method}.csv"
    validation_prediction_path = (
        output_dir / "validation_predictions" / endpoint / f"{split_name}_{method}.csv"
    )
    if cell_path.is_file() and prediction_path.is_file() and validation_prediction_path.is_file():
        existing = json.loads(cell_path.read_text(encoding="utf-8"))
        if (
            existing.get("data_sha256") == data_sha
            and existing.get("runner_sha256") == runner_sha
            and existing.get("config_sha256") == config_sha
            and existing.get("predictions_sha256") == sha256_file(prediction_path)
            and existing.get("validation_predictions_sha256")
            == sha256_file(validation_prediction_path)
        ):
            print(f"CURRENT {endpoint} {split_name} {method}", flush=True)
            return existing

    validations = []
    validation_predictions: list[np.ndarray] = []
    for candidate_index, candidate in enumerate(GRIDS[method]):
        pred, _ = fit_predict(
            method,
            candidate,
            features[inner_idx],
            y[inner_idx],
            features[valid_idx],
            seed + candidate_index,
        )
        validations.append({
            "candidate_index": candidate_index,
            "candidate": candidate.name,
            "params": candidate.params,
            "validation_mae": float(mean_absolute_error(y[valid_idx], pred)),
        })
        validation_predictions.append(pred)
    selected_index = min(
        range(len(validations)),
        key=lambda index: (validations[index]["validation_mae"], validations[index]["candidate"]),
    )
    selected = GRIDS[method][selected_index]
    # Retrain on exactly the inner training rows.  This matches the neural
    # benchmark's final checkpoint label budget; the validation rows select
    # configuration only and never enter final model fitting.
    test_pred, preprocessing = fit_predict(
        method,
        selected,
        features[inner_idx],
        y[inner_idx],
        features[test_idx],
        seed + selected_index,
    )
    selected_validation_pred = validation_predictions[selected_index]
    validation_table = pd.DataFrame({
        "endpoint": endpoint,
        "split": split_name,
        "method": method,
        "row_index": valid_idx,
        "smiles": frame.iloc[valid_idx]["smiles"].to_numpy(),
        "y_true": y[valid_idx],
        "y_pred": selected_validation_pred,
    })
    atomic_csv(validation_prediction_path, validation_table)
    predictions = pd.DataFrame({
        "endpoint": endpoint,
        "split": split_name,
        "method": method,
        "row_index": test_idx,
        "smiles": frame.iloc[test_idx]["smiles"].to_numpy(),
        "y_true": y[test_idx],
        "y_pred": test_pred,
    })
    atomic_csv(prediction_path, predictions)
    cell = {
        "schema_version": 1,
        "endpoint": endpoint,
        "split": split_name,
        "method": method,
        "data_sha256": data_sha,
        "runner_sha256": runner_sha,
        "config": config,
        "config_sha256": config_sha,
        "seed": seed,
        "n_inner_train": int(len(inner_idx)),
        "n_validation": int(len(valid_idx)),
        "n_final_fit": int(len(inner_idx)),
        "n_test": int(len(test_idx)),
        "inner_train_smiles_sha256": smiles_sha256(frame.iloc[inner_idx]["smiles"]),
        "validation_smiles_sha256": smiles_sha256(frame.iloc[valid_idx]["smiles"]),
        "test_smiles_sha256": smiles_sha256(frame.iloc[test_idx]["smiles"]),
        "selected_candidate_index": selected_index,
        "selected_candidate": selected.name,
        "selected_params": selected.params,
        "validation_candidates": validations,
        "selected_validation_mae": validations[selected_index]["validation_mae"],
        "test_mae": float(mean_absolute_error(y[test_idx], test_pred)),
        "preprocessing": preprocessing,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "validation_predictions": str(validation_prediction_path),
        "validation_predictions_sha256": sha256_file(validation_prediction_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
    }
    atomic_json(cell_path, cell)
    print(
        f"DONE {endpoint} {split_name} {method} "
        f"val={cell['selected_validation_mae']:.6f} test={cell['test_mae']:.6f}",
        flush=True,
    )
    return cell


def summarize(output_dir: Path) -> None:
    rows = []
    for cell_path in sorted((output_dir / "cells").glob("*/*.json")):
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        rows.append({key: cell[key] for key in (
            "endpoint", "split", "method", "n_final_fit", "n_validation", "n_test",
            "selected_candidate", "selected_validation_mae", "test_mae",
            "data_sha256", "runner_sha256", "config_sha256", "predictions_sha256",
        )})
    if not rows:
        return
    cells = pd.DataFrame(rows).sort_values(["endpoint", "method", "split"])
    atomic_csv(output_dir / "cell_summary.csv", cells)
    summary = (
        cells.groupby(["endpoint", "method"], as_index=False)
        .agg(
            mean_test_mae=("test_mae", "mean"),
            std_test_mae=("test_mae", lambda values: float(np.std(values, ddof=0))),
            n_splits=("split", "nunique"),
        )
        .sort_values(["endpoint", "mean_test_mae", "method"])
    )
    atomic_csv(output_dir / "summary.csv", summary)
    atomic_json(output_dir / "summary.json", {
        "schema_version": 1,
        "complete_cells": int(len(cells)),
        "expected_cells": len(ENDPOINTS) * len(PAPER_PROTOCOL["splits"]) * len(METHODS),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "summary_csv": str(output_dir / "summary.csv"),
        "summary_csv_sha256": sha256_file(output_dir / "summary.csv"),
        "cell_summary_csv": str(output_dir / "cell_summary.csv"),
        "cell_summary_csv_sha256": sha256_file(output_dir / "cell_summary.csv"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data/moljepa_benchmarks/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/results_openadmet_baselines/classical"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="do not emit a possibly partial summary (use for concurrent shards)",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require 0 <= shard-id < num-shards")

    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("endpoints", {})) != {endpoint.slug for endpoint in ENDPOINTS}:
        raise ValueError("prepared manifest does not contain exactly the 23 locked endpoints")
    runner_sha = stable_hash({
        "runner": sha256_file(Path(__file__).resolve()),
        "benchmark_config": sha256_file(Path(__file__).with_name("moljepa_benchmark_config.py")),
    })
    selected = [endpoint for endpoint in ENDPOINTS if not args.only or endpoint.slug in set(args.only)]
    unknown = set(args.only or ()) - {endpoint.slug for endpoint in ENDPOINTS}
    if unknown:
        raise ValueError(f"unknown endpoints: {sorted(unknown)}")

    for index, endpoint in enumerate(selected):
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
        feature_cache: dict[str, tuple[np.ndarray, list[str] | None]] = {}
        for method in args.method:
            if method == "dummy_median":
                feature_cache[method] = (np.empty((len(frame), 0)), None)
            elif method == "ecfp4_rf":
                feature_cache[method] = (ecfp4(frame["smiles"].tolist(), counts=False), None)
            elif method == "ecfp4_lgbm":
                feature_cache[method] = (ecfp4(frame["smiles"].tolist(), counts=True), None)
            else:
                feature_cache[method] = rdkit_descriptors(frame["smiles"].tolist())
            features, names = feature_cache[method]
            for split_name in PAPER_PROTOCOL["splits"]:
                run_cell(
                    endpoint=endpoint.slug,
                    split_name=split_name,
                    method=method,
                    frame=frame,
                    features=features,
                    feature_names=names,
                    data_sha=expected_sha,
                    runner_sha=runner_sha,
                    output_dir=args.output_dir,
                )
    if not args.skip_summary:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
