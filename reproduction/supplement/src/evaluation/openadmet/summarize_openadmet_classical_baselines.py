#!/usr/bin/env python3
"""Fail-closed audit and summary for same-split OpenADMET baselines.

No summary is emitted unless every endpoint/method/split cell is present and
passes prediction, selection, provenance, and (when supplied) neural split
parity checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from moljepa_benchmark_config import ENDPOINTS, ENDPOINT_BY_SLUG, PAPER_PROTOCOL


TOLERANCE = 1e-10
METHODS = (
    "dummy_median",
    "ecfp4_rf",
    "ecfp4_lgbm",
    "rdkit_desc_rf",
    "rdkit_desc_lgbm",
)
PREDICTION_COLUMNS = (
    "endpoint", "split", "method", "row_index", "smiles", "y_true", "y_pred"
)


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
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float, context: str) -> None:
    require(
        math.isfinite(left) and math.isfinite(right) and abs(left - right) <= TOLERANCE,
        f"{context}: {left!r} != {right!r}",
    )


def resolve_artifact(output_dir: Path, stored: Any, relative: Path) -> Path:
    """Resolve both in-place cluster paths and copied result packages."""
    candidate = Path(str(stored)) if stored else Path()
    if stored and candidate.is_file():
        return candidate
    local = output_dir / relative
    require(local.is_file(), f"missing artifact: {local} (stored path was {stored!r})")
    return local


def read_prediction_table(
    path: Path,
    *,
    endpoint: str,
    split_name: str,
    method: str,
    expected_rows: int,
) -> pd.DataFrame:
    # The prediction writer emits enough decimal digits to round-trip binary64
    # values exactly.  Pandas' default "high" converter can nevertheless move
    # some values by one or two ULPs when reading those decimals, which creates
    # a false failure against the same labels stored in neural-result JSON.
    # Use the C parser's round-trip converter so exact label comparisons audit
    # the serialized values rather than a parser approximation.
    frame = pd.read_csv(path, float_precision="round_trip")
    require(tuple(frame.columns) == PREDICTION_COLUMNS, f"{path}: unexpected columns")
    require(len(frame) == expected_rows, f"{path}: expected {expected_rows} rows, got {len(frame)}")
    require(frame["row_index"].is_unique, f"{path}: duplicate row_index")
    require(frame["smiles"].notna().all(), f"{path}: null SMILES")
    require(frame["endpoint"].eq(endpoint).all(), f"{path}: endpoint mismatch")
    require(frame["split"].eq(split_name).all(), f"{path}: split mismatch")
    require(frame["method"].eq(method).all(), f"{path}: method mismatch")
    for column in ("y_true", "y_pred"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"{path}: non-finite {column}")
    return frame


def load_neural_results(neural_dir: Path | None) -> dict[str, dict[str, Any]]:
    if neural_dir is None:
        return {}
    expected = {endpoint.slug for endpoint in ENDPOINTS}
    paths = list(neural_dir.glob("results_*.json"))
    results: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = payload.get("task")
        if task in expected:
            require(task not in results, f"duplicate neural result for {task}")
            results[task] = payload
    require(set(results) == expected, f"neural result set mismatch: missing={sorted(expected-set(results))}")
    return results


def neural_split_contract(
    neural: dict[str, Any], split_name: str
) -> tuple[str, str, str, str, np.ndarray]:
    split = neural.get("splits", {}).get(split_name)
    require(isinstance(split, dict), f"{neural.get('task')} missing neural {split_name}")
    methods = split.get("methods", {})
    require(bool(methods), f"{neural.get('task')} {split_name}: no neural methods")
    contracts = set()
    truth_arrays = []
    for record in methods.values():
        contracts.add((
            record.get("data_sha256"),
            record.get("train_smiles_sha256"),
            record.get("validation_smiles_sha256"),
            record.get("test_smiles_sha256"),
        ))
        truth_arrays.append(np.asarray(record.get("y_true"), dtype=float))
    require(len(contracts) == 1, f"{neural.get('task')} {split_name}: neural split hashes disagree")
    require(
        all(np.array_equal(truth_arrays[0], values) for values in truth_arrays[1:]),
        f"{neural.get('task')} {split_name}: neural y_true arrays disagree",
    )
    data_sha, train_sha, validation_sha, test_sha = next(iter(contracts))
    require(all((data_sha, train_sha, validation_sha, test_sha)), f"incomplete neural hashes")
    return data_sha, train_sha, validation_sha, test_sha, truth_arrays[0]


def audit(output_dir: Path, neural_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    endpoints = tuple(endpoint.slug for endpoint in ENDPOINTS)
    splits = tuple(PAPER_PROTOCOL["splits"])
    expected_keys = {
        (endpoint, split_name, method)
        for endpoint in endpoints for split_name in splits for method in METHODS
    }
    cell_paths = list((output_dir / "cells").glob("*/*.json"))
    indexed: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    for path in cell_paths:
        cell = json.loads(path.read_text(encoding="utf-8"))
        key = (cell.get("endpoint"), cell.get("split"), cell.get("method"))
        require(key not in indexed, f"duplicate cell key {key}")
        indexed[key] = (path, cell)
    require(
        set(indexed) == expected_keys,
        "cell set mismatch: "
        f"missing={sorted(expected_keys-set(indexed), key=str)[:20]} "
        f"extra={sorted(set(indexed)-expected_keys, key=str)[:20]}",
    )
    neural_results = load_neural_results(neural_dir)
    rows: list[dict[str, Any]] = []
    endpoint_data_hashes: dict[str, set[str]] = {endpoint: set() for endpoint in endpoints}
    runner_hashes: set[str] = set()
    software_contracts: set[str] = set()

    for endpoint in endpoints:
        for split_name in splits:
            neural_contract = (
                neural_split_contract(neural_results[endpoint], split_name)
                if neural_results else None
            )
            split_test_indices: set[int] | None = None
            split_validation_indices: set[int] | None = None
            split_test_truth: np.ndarray | None = None
            split_validation_truth: np.ndarray | None = None
            for method in METHODS:
                path, cell = indexed[(endpoint, split_name, method)]
                prefix = f"{endpoint}/{split_name}/{method}"
                require(cell.get("schema_version") == 1, f"{prefix}: unsupported schema")
                config = cell.get("config")
                require(isinstance(config, dict), f"{prefix}: missing config")
                require(stable_hash(config) == cell.get("config_sha256"), f"{prefix}: config hash mismatch")
                require(config.get("method") == method, f"{prefix}: config method mismatch")
                require(config.get("selection_metric") == "validation_mae", f"{prefix}: selection is not validation MAE")
                expected_fingerprint = None
                if method == "ecfp4_rf":
                    expected_fingerprint = {"kind": "Morgan bit vector", "radius": 2, "n_bits": 2048}
                elif method == "ecfp4_lgbm":
                    expected_fingerprint = {"kind": "Morgan count", "radius": 2, "n_bits": 2048}
                require(config.get("fingerprint") == expected_fingerprint, f"{prefix}: feature contract mismatch")
                if method.startswith("rdkit_desc_"):
                    require(
                        isinstance(config.get("descriptor_names"), list)
                        and bool(config["descriptor_names"]),
                        f"{prefix}: descriptor contract is empty",
                    )
                else:
                    require(config.get("descriptor_names") is None, f"{prefix}: unexpected descriptors")
                require(
                    config.get("final_fit_policy") == (
                        "selected configuration retrained on inner-train only; validation labels "
                        "are used for selection but not fitted model weights"
                    ),
                    f"{prefix}: final fit policy is not the primary no-refit protocol",
                )
                candidates = cell.get("validation_candidates")
                require(isinstance(candidates, list) and candidates, f"{prefix}: no validation candidates")
                declared_candidates = config.get("candidates")
                require(isinstance(declared_candidates, list), f"{prefix}: config has no candidate grid")
                require(len(declared_candidates) == len(candidates), f"{prefix}: candidate grid length mismatch")
                for index, candidate in enumerate(candidates):
                    require(candidate.get("candidate_index") == index, f"{prefix}: candidate index mismatch")
                    require(
                        set(candidate) == {"candidate_index", "candidate", "params", "validation_mae"},
                        f"{prefix}: candidate record contains non-validation fields",
                    )
                    require(math.isfinite(float(candidate["validation_mae"])), f"{prefix}: invalid validation MAE")
                    require(
                        declared_candidates[index] == {
                            "name": candidate["candidate"], "params": candidate["params"]
                        },
                        f"{prefix}: evaluated candidate differs from declared grid",
                    )
                selected_index = min(
                    range(len(candidates)),
                    key=lambda index: (
                        float(candidates[index]["validation_mae"]),
                        str(candidates[index]["candidate"]),
                    ),
                )
                require(cell.get("selected_candidate_index") == selected_index, f"{prefix}: selected index is not validation optimum")
                require(cell.get("selected_candidate") == candidates[selected_index]["candidate"], f"{prefix}: selected candidate mismatch")
                require(cell.get("selected_params") == candidates[selected_index]["params"], f"{prefix}: selected params mismatch")
                close(
                    float(cell["selected_validation_mae"]),
                    float(candidates[selected_index]["validation_mae"]),
                    f"{prefix} selected validation MAE",
                )
                require(cell.get("n_final_fit") == cell.get("n_inner_train"), f"{prefix}: final fit used non-training rows")

                test_path = resolve_artifact(
                    output_dir, cell.get("predictions"),
                    Path("predictions") / endpoint / f"{split_name}_{method}.csv",
                )
                validation_path = resolve_artifact(
                    output_dir, cell.get("validation_predictions"),
                    Path("validation_predictions") / endpoint / f"{split_name}_{method}.csv",
                )
                require(sha256_file(test_path) == cell.get("predictions_sha256"), f"{prefix}: test prediction hash mismatch")
                require(
                    sha256_file(validation_path) == cell.get("validation_predictions_sha256"),
                    f"{prefix}: validation prediction hash mismatch",
                )
                test = read_prediction_table(
                    test_path, endpoint=endpoint, split_name=split_name, method=method,
                    expected_rows=int(cell["n_test"]),
                )
                validation = read_prediction_table(
                    validation_path, endpoint=endpoint, split_name=split_name, method=method,
                    expected_rows=int(cell["n_validation"]),
                )
                require(
                    set(test["row_index"]).isdisjoint(set(validation["row_index"])),
                    f"{prefix}: validation/test row overlap",
                )
                current_test_indices = set(test["row_index"].astype(int))
                current_validation_indices = set(validation["row_index"].astype(int))
                if split_test_indices is None:
                    split_test_indices = current_test_indices
                    split_validation_indices = current_validation_indices
                    split_test_truth = test["y_true"].to_numpy(dtype=float)
                    split_validation_truth = validation["y_true"].to_numpy(dtype=float)
                else:
                    require(current_test_indices == split_test_indices, f"{prefix}: methods use different test rows")
                    require(current_validation_indices == split_validation_indices, f"{prefix}: methods use different validation rows")
                    require(
                        np.array_equal(test["y_true"].to_numpy(dtype=float), split_test_truth),
                        f"{prefix}: methods use different test labels/order",
                    )
                    require(
                        np.array_equal(validation["y_true"].to_numpy(dtype=float), split_validation_truth),
                        f"{prefix}: methods use different validation labels/order",
                    )
                require(
                    smiles_sha256(test["smiles"]) == cell.get("test_smiles_sha256"),
                    f"{prefix}: test SMILES hash does not match predictions",
                )
                require(
                    smiles_sha256(validation["smiles"]) == cell.get("validation_smiles_sha256"),
                    f"{prefix}: validation SMILES hash does not match predictions",
                )
                recomputed_validation = float(np.mean(np.abs(validation["y_true"] - validation["y_pred"])))
                recomputed_test = float(np.mean(np.abs(test["y_true"] - test["y_pred"])))
                close(recomputed_validation, float(cell["selected_validation_mae"]), f"{prefix} validation prediction MAE")
                close(recomputed_test, float(cell["test_mae"]), f"{prefix} test prediction MAE")
                endpoint_data_hashes[endpoint].add(str(cell.get("data_sha256")))
                runner_hashes.add(str(cell.get("runner_sha256")))
                software = cell.get("software_versions")
                require(isinstance(software, dict) and len(software) == 6, f"{prefix}: incomplete software versions")
                software_contracts.add(json.dumps(software, sort_keys=True))

                if neural_contract is not None:
                    data_sha, train_sha, validation_sha, test_sha, neural_truth = neural_contract
                    require(cell.get("data_sha256") == data_sha, f"{prefix}: prepared data hash differs from neural")
                    require(cell.get("inner_train_smiles_sha256") == train_sha, f"{prefix}: train split hash differs from neural")
                    require(cell.get("validation_smiles_sha256") == validation_sha, f"{prefix}: validation split hash differs from neural")
                    require(cell.get("test_smiles_sha256") == test_sha, f"{prefix}: test split hash differs from neural")
                    require(
                        np.array_equal(test["y_true"].to_numpy(dtype=float), neural_truth),
                        f"{prefix}: ordered test labels differ from neural",
                    )
                rows.append({
                    "endpoint": endpoint,
                    "family": ENDPOINT_BY_SLUG[endpoint].family,
                    "split": split_name,
                    "method": method,
                    "n_inner_train": int(cell["n_inner_train"]),
                    "n_validation": int(cell["n_validation"]),
                    "n_test": int(cell["n_test"]),
                    "selected_candidate": cell["selected_candidate"],
                    "validation_mae": recomputed_validation,
                    "test_mae": recomputed_test,
                    "data_sha256": cell["data_sha256"],
                    "inner_train_smiles_sha256": cell["inner_train_smiles_sha256"],
                    "validation_smiles_sha256": cell["validation_smiles_sha256"],
                    "test_smiles_sha256": cell["test_smiles_sha256"],
                    "runner_sha256": cell["runner_sha256"],
                    "config_sha256": cell["config_sha256"],
                    "validation_predictions_sha256": cell["validation_predictions_sha256"],
                    "predictions_sha256": cell["predictions_sha256"],
                })

    for endpoint, hashes in endpoint_data_hashes.items():
        require(len(hashes) == 1 and "None" not in hashes, f"{endpoint}: inconsistent data hashes")
    require(len(runner_hashes) == 1 and "None" not in runner_hashes, "mixed or missing runner hashes")
    require(len(software_contracts) == 1, "mixed software environments across cells")
    cells = pd.DataFrame(rows).sort_values(["endpoint", "method", "split"]).reset_index(drop=True)
    summaries = []
    for (endpoint, family, method), group in cells.groupby(["endpoint", "family", "method"], sort=False):
        require(set(group["split"]) == set(splits), f"{endpoint}/{method}: incomplete split set")
        values = group["test_mae"].to_numpy(dtype=float)
        summaries.append({
            "endpoint": endpoint,
            "family": family,
            "method": method,
            "mean_test_mae": float(np.mean(values)),
            "std_test_mae": float(np.std(values, ddof=0)),
            "n_splits": len(values),
        })
    summary = pd.DataFrame(summaries).sort_values(["endpoint", "mean_test_mae", "method"]).reset_index(drop=True)
    audit_record = {
        "schema_version": 1,
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_endpoints": len(endpoints),
        "expected_methods": list(METHODS),
        "expected_splits": list(splits),
        "validated_cells": len(cells),
        "validation_only_selection_verified": True,
        "prediction_maes_recomputed": True,
        "neural_split_hashes_verified": bool(neural_results),
        "primary_fit_policy": "inner-train only after validation-only configuration selection",
        "software_versions": json.loads(next(iter(software_contracts))),
        "feature_contract": {
            "ecfp4_rf": "2,048-bit binary Morgan fingerprint, radius 2 (Mol-JEPA reference helper convention)",
            "ecfp4_lgbm": "2,048-bin Morgan count fingerprint, radius 2 (Pat Walters benchmark convention)",
            "rdkit_desc_rf": "RDKit descriptor vector; median imputation fitted on inner-train only",
            "rdkit_desc_lgbm": "RDKit descriptor vector; median imputation fitted on inner-train only",
            "dummy_median": "inner-train median; the constant minimizing MAE",
        },
    }
    return cells, summary, audit_record


def markdown(summary: pd.DataFrame, audit_record: dict[str, Any]) -> str:
    lookup = {
        (row.endpoint, row.method): (row.mean_test_mae, row.std_test_mae)
        for row in summary.itertuples(index=False)
    }
    lines = [
        "# Audited OpenADMET same-split classical baselines",
        "",
        f"Audit status: **{audit_record['status']}**. "
        f"All {audit_record['validated_cells']} cells (23 endpoints × 3 splits × 5 methods) passed.",
        "",
        "Hyperparameters were selected using cluster-held-out validation MAE only. "
        "The selected configuration was retrained on inner-train rows only, matching the neural "
        "benchmark label budget; test predictions were used only for final evaluation.",
        "",
        "Fingerprint RF uses binary radius-2 Morgan fingerprints (2,048 bits), following the released "
        "Mol-JEPA RF helper. Fingerprint LightGBM uses radius-2 Morgan counts (2,048 bins), following the "
        "bundled Pat Walters benchmark. Both RF and LightGBM are also evaluated on median-imputed "
        "RDKit descriptors. The dummy is the training median because MAE is the target metric.",
        "",
        "| Endpoint | Median dummy | ECFP4 RF | ECFP4 LightGBM | RDKit-2D RF | RDKit-2D LightGBM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = (
        "dummy_median",
        "ecfp4_rf",
        "ecfp4_lgbm",
        "rdkit_desc_rf",
        "rdkit_desc_lgbm",
    )
    for endpoint in (item.slug for item in ENDPOINTS):
        values = [lookup[(endpoint, method)] for method in labels]
        rendered = [f"{mean:.4f} ± {std:.4f}" for mean, std in values]
        lines.append(f"| {endpoint} | " + " | ".join(rendered) + " |")
    lines.extend([
        "",
        "Values are mean test MAE ± population standard deviation across the three locked splits.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--neural-results-dir", type=Path, default=None,
        help="directory containing all 23 results_<endpoint>.json neural results",
    )
    args = parser.parse_args()
    destination = args.output_dir or (args.results_dir / "audited_summary")
    cells, summary, audit_record = audit(args.results_dir, args.neural_results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_csv(destination / "cells.csv", cells)
    atomic_csv(destination / "endpoint_summary.csv", summary)
    audit_record["cells_csv_sha256"] = sha256_file(destination / "cells.csv")
    audit_record["endpoint_summary_csv_sha256"] = sha256_file(destination / "endpoint_summary.csv")
    atomic_text(destination / "audit.json", json.dumps(audit_record, indent=2, sort_keys=True) + "\n")
    atomic_text(destination / "summary.md", markdown(summary, audit_record))
    print(json.dumps(audit_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
