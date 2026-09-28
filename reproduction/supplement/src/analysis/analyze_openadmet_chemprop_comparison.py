#!/usr/bin/env python3
"""Fail-closed same-split OpenADMET comparison of ChemRasayan and Chemprop.

The inferential unit is an endpoint.  Each endpoint score first averages the
same three locked cluster splits; the 23 paired endpoint means are then used
for the bootstrap and exact tests.  The script also reloads the five audited
classical contrasts and recomputes Holm adjustment across the resulting six
same-split baseline contrasts, separately for Wilcoxon and sign-test p-values.

No output is written until every Chemprop cell, prediction payload, summary,
held-out target, row count, and split-membership hash has passed validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_ENDPOINTS = 23
EXPECTED_SPLITS = ("split1", "split2", "split3")
CHEMPROP_METHOD = "chemprop_dmpnn"
CLASSICAL_METHODS = (
    "dummy_median",
    "ecfp4_rf",
    "ecfp4_lgbm",
    "rdkit_desc_rf",
    "rdkit_desc_lgbm",
)
HASH_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FixedFTCell:
    endpoint: str
    split: str
    family: str
    data_sha256: str
    train_smiles_sha256: str
    validation_smiles_sha256: str
    test_smiles_sha256: str
    n_test: int
    y_true: np.ndarray
    y_pred: np.ndarray
    test_mae: float


@dataclass(frozen=True)
class ChempropCell:
    endpoint: str
    split: str
    data_sha256: str
    train_smiles_sha256: str
    validation_smiles_sha256: str
    test_smiles_sha256: str
    n_test: int
    row_index: np.ndarray
    smiles: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    test_mae: float
    cell_json_sha256: str
    predictions_sha256: str


def fail(message: str) -> None:
    raise ValueError(message)


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


def smiles_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_anonymous_payload(payload: Any) -> None:
    """Reject common private path, account, host, and infrastructure disclosures."""
    serialized = json.dumps(payload, sort_keys=True)
    forbidden = {
        "Windows absolute path": re.compile(r"[A-Za-z]:[\\/]") ,
        "private storage path": re.compile(r"/(?:mnt|nfs|home|Users)/", re.IGNORECASE),
        "cluster account or namespace": re.compile(r"\bs\d{6,}-(?:eidf\d+|[a-z]+ns\d+)\b", re.IGNORECASE),
        "private IPv4 address": re.compile(
            r"(?<!\d)(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}(?!\d)"
        ),
    }
    for label, pattern in forbidden.items():
        if pattern.search(serialized):
            fail(f"Anonymous summary contains a {label}")


def float_arrays_match_one_ulp(left: np.ndarray, right: np.ndarray) -> bool:
    """Accept equality or at most one binary64 ULP at either finite operand."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    tolerance = np.maximum(np.abs(np.spacing(left)), np.abs(np.spacing(right)))
    return bool(np.all(np.abs(left - right) <= tolerance))


def float_arrays_match_bounded(left: np.ndarray, right: np.ndarray, *, atol: float) -> bool:
    """Match finite binary64 arrays under an explicit absolute-only CSV contract."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.shape != right.shape
        or not math.isfinite(atol)
        or atol < 0.0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        return False
    return bool(np.all(np.abs(left - right) <= atol))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def exact_wilcoxon_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    """Exact sign-randomization Wilcoxon test with exact-zero pairs omitted."""
    raw = np.asarray(list(differences), dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        fail("Wilcoxon differences must be a finite one-dimensional array")
    nonzero = raw[raw != 0.0]
    if len(nonzero) == 0:
        return {
            "n_nonzero": 0,
            "zero_pairs": int(len(raw)),
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_two_sided_exact": 1.0,
            "definition": "exact sign-randomization distribution; doubled smaller tail",
        }
    doubled_ranks = np.rint(average_ranks(np.abs(nonzero)) * 2).astype(int)
    counts: dict[int, int] = {0: 1}
    for rank in doubled_ranks:
        updated = counts.copy()
        for subtotal, count in counts.items():
            key = subtotal + int(rank)
            updated[key] = updated.get(key, 0) + count
        counts = updated
    observed_plus = int(doubled_ranks[nonzero > 0].sum())
    total_rank = int(doubled_ranks.sum())
    lower_boundary = min(observed_plus, total_rank - observed_plus)
    lower_count = sum(count for score, count in counts.items() if score <= lower_boundary)
    p_value = min(1.0, 2.0 * lower_count / (2 ** len(nonzero)))
    return {
        "n_nonzero": int(len(nonzero)),
        "zero_pairs": int(len(raw) - len(nonzero)),
        "w_plus": observed_plus / 2.0,
        "w_minus": (total_rank - observed_plus) / 2.0,
        "statistic": lower_boundary / 2.0,
        "p_two_sided_exact": float(p_value),
        "definition": "exact sign-randomization distribution; doubled smaller tail",
    }


def exact_sign_test_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    raw = np.asarray(list(differences), dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        fail("Sign-test differences must be a finite one-dimensional array")
    positives = int(np.sum(raw > 0.0))
    negatives = int(np.sum(raw < 0.0))
    nonzero = positives + negatives
    if nonzero == 0:
        p_value = 1.0
    else:
        smaller = min(positives, negatives)
        lower = sum(math.comb(nonzero, index) for index in range(smaller + 1)) / (2 ** nonzero)
        p_value = min(1.0, 2.0 * lower)
    return {
        "n_nonzero": nonzero,
        "positive_differences": positives,
        "negative_differences": negatives,
        "zero_pairs": int(len(raw) - nonzero),
        "p_two_sided_exact": float(p_value),
        "definition": "exact Binomial(n, 0.5); doubled smaller tail; zero pairs omitted",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values or any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values):
        fail("Holm adjustment requires one or more finite p-values in [0, 1]")
    count = len(p_values)
    ordered = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def percentile_bootstrap_mean_ci(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        fail("Bootstrap values must be a nonempty finite one-dimensional array")
    if replicates < 1:
        fail("Bootstrap replicate count must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high)


def load_fixed_ft(directory: Path) -> tuple[dict[tuple[str, str], FixedFTCell], dict[str, str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"ChemRasayan result directory is missing: {directory}")
    files = sorted(directory.glob("results_*.json"))
    if len(files) != EXPECTED_ENDPOINTS:
        fail(f"Expected {EXPECTED_ENDPOINTS} ChemRasayan endpoint JSONs, found {len(files)}")
    cells: dict[tuple[str, str], FixedFTCell] = {}
    hashes: dict[str, str] = {}
    endpoints: set[str] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        endpoint = payload.get("task")
        if not isinstance(endpoint, str) or not endpoint or endpoint in endpoints:
            fail(f"Invalid or duplicate ChemRasayan endpoint in {path}")
        if path.name != f"results_{endpoint}.json":
            fail(f"ChemRasayan filename/task mismatch: {path.name} versus {endpoint}")
        endpoints.add(endpoint)
        hashes[path.name] = sha256_file(path)
        if payload.get("primary_metric") != "mae":
            fail(f"ChemRasayan primary metric is not MAE: {endpoint}")
        if payload.get("data_protocol") != "reconstructed_butina_public_snapshot":
            fail(f"Unexpected ChemRasayan data protocol: {endpoint}")
        if tuple(payload.get("splits", {}).keys()) != EXPECTED_SPLITS:
            fail(f"ChemRasayan split order/set mismatch: {endpoint}")
        family = payload.get("family")
        for split in EXPECTED_SPLITS:
            record = payload["splits"][split]["methods"].get("full_mix")
            if not isinstance(record, dict):
                fail(f"Missing ChemRasayan full_mix cell: {endpoint}/{split}")
            if record.get("method") != "full_mix" or record.get("task") != endpoint or record.get("split") != split:
                fail(f"ChemRasayan cell identity mismatch: {endpoint}/{split}")
            y_true = np.asarray(record.get("y_true"), dtype=float)
            y_pred = np.asarray(record.get("y_pred"), dtype=float)
            n_test = int(record.get("test_rows", -1))
            if (
                y_true.ndim != 1
                or y_pred.shape != y_true.shape
                or len(y_true) != n_test
                or not np.isfinite(y_true).all()
                or not np.isfinite(y_pred).all()
            ):
                fail(f"Invalid ChemRasayan held-out arrays: {endpoint}/{split}")
            recomputed = float(np.mean(np.abs(y_true - y_pred)))
            if not math.isclose(recomputed, float(record.get("test_mae")), rel_tol=0.0, abs_tol=1e-12):
                fail(f"ChemRasayan MAE mismatch: {endpoint}/{split}")
            key = (endpoint, split)
            cells[key] = FixedFTCell(
                endpoint=endpoint,
                split=split,
                family=str(family),
                data_sha256=require_hash(record.get("data_sha256"), f"{key} data hash"),
                train_smiles_sha256=require_hash(record.get("train_smiles_sha256"), f"{key} train hash"),
                validation_smiles_sha256=require_hash(record.get("validation_smiles_sha256"), f"{key} validation hash"),
                test_smiles_sha256=require_hash(record.get("test_smiles_sha256"), f"{key} test hash"),
                n_test=n_test,
                y_true=y_true,
                y_pred=y_pred,
                test_mae=recomputed,
            )
    if len(cells) != EXPECTED_ENDPOINTS * len(EXPECTED_SPLITS):
        fail("ChemRasayan endpoint/split grid is incomplete")
    return cells, hashes


def _local_payload_path(cell_json: Path, recorded: Any, fallback: str) -> Path:
    if isinstance(recorded, str):
        candidate = Path(recorded)
        if candidate.is_file():
            return candidate
    candidate = cell_json.parent / fallback
    if not candidate.is_file():
        raise FileNotFoundError(f"Required Chemprop payload is missing: {candidate}")
    return candidate


def load_chemprop(
    directory: Path,
) -> tuple[dict[tuple[str, str], ChempropCell], dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Retrieved Chemprop result directory is missing: {directory}. "
            "Pass --chemprop-dir after the certified result has been retrieved."
        )
    complete = directory / "COMPLETE"
    if not complete.is_file() or not complete.read_text(encoding="utf-8").strip():
        fail(f"Chemprop COMPLETE marker is missing or empty: {complete}")
    summary_path = directory / "summary.json"
    summary_csv_path = directory / "summary.csv"
    cell_summary_path = directory / "cell_summary.csv"
    for path in (summary_path, summary_csv_path, cell_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Chemprop finalizer output is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "PASS"
        or summary.get("complete_cells") != EXPECTED_ENDPOINTS * len(EXPECTED_SPLITS)
        or summary.get("expected_cells") != EXPECTED_ENDPOINTS * len(EXPECTED_SPLITS)
        or summary.get("outer_test_targets_blinded") is not True
    ):
        fail("Chemprop summary does not certify all 69 blinded-test cells")
    roundtrip = summary.get("target_roundtrip_check")
    if not isinstance(roundtrip, dict):
        fail("Chemprop summary lacks its target round-trip contract")
    target_atol = float(roundtrip.get("absolute_tolerance", math.nan))
    if (
        not math.isfinite(target_atol)
        or target_atol < 0.0
        or target_atol > 1e-15
        or float(roundtrip.get("relative_tolerance", math.nan)) != 0.0
    ):
        fail("Chemprop target round-trip tolerance is absent or broader than 1e-15 absolute")
    if sha256_file(summary_csv_path) != summary.get("summary_csv_sha256"):
        fail("Chemprop summary.csv hash differs from summary.json")
    if sha256_file(cell_summary_path) != summary.get("cell_summary_csv_sha256"):
        fail("Chemprop cell_summary.csv hash differs from summary.json")

    paths = sorted((directory / "cells").glob("*/*/cell.json"))
    if len(paths) != EXPECTED_ENDPOINTS * len(EXPECTED_SPLITS):
        fail(f"Expected 69 Chemprop cell.json files, found {len(paths)}")
    cells: dict[tuple[str, str], ChempropCell] = {}
    runner_hashes: set[str] = set()
    input_records: list[tuple[str, str]] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        endpoint = record.get("endpoint")
        split = record.get("split")
        key = (endpoint, split)
        if not isinstance(endpoint, str) or split not in EXPECTED_SPLITS or key in cells:
            fail(f"Invalid or duplicate Chemprop cell identity: {key}")
        if path.parent.parent.name != endpoint or path.parent.name != f"{split}_{CHEMPROP_METHOD}":
            fail(f"Chemprop cell path/identity mismatch: {path}")
        if record.get("method") != CHEMPROP_METHOD:
            fail(f"Unexpected Chemprop method: {key}")
        if stable_hash(record.get("config")) != record.get("config_sha256"):
            fail(f"Chemprop config hash mismatch: {key}")
        if (
            record.get("test_targets_blinded_in_training_table") is not True
            or record.get("config", {}).get("test_targets_blinded_in_chemprop_input") is not True
        ):
            fail(f"Chemprop test-target blinding is not certified: {key}")
        runner_hashes.add(require_hash(record.get("runner_sha256"), f"{key} runner hash"))
        prediction_path = _local_payload_path(path, record.get("predictions"), "predictions.csv")
        prediction_hash = sha256_file(prediction_path)
        if prediction_hash != record.get("predictions_sha256"):
            fail(f"Chemprop prediction hash mismatch: {key}")
        predictions = pd.read_csv(prediction_path)
        required = {"endpoint", "split", "method", "row_index", "smiles", "y_true", "y_pred"}
        if not required.issubset(predictions.columns):
            fail(f"Chemprop prediction columns are incomplete: {key}")
        if (
            len(predictions) != int(record.get("n_test", -1))
            or predictions["row_index"].isna().any()
            or not predictions["row_index"].is_unique
        ):
            fail(f"Chemprop prediction row coverage is invalid: {key}")
        for column, expected in (("endpoint", endpoint), ("split", split), ("method", CHEMPROP_METHOD)):
            if set(predictions[column].astype(str)) != {expected}:
                fail(f"Chemprop prediction {column} mismatch: {key}")
        predictions = predictions.sort_values("row_index", kind="mergesort").reset_index(drop=True)
        row_index = predictions["row_index"].to_numpy(dtype=np.int64)
        if len(row_index) and (np.diff(row_index) <= 0).any():
            fail(f"Chemprop row indices are not strictly increasing after sort: {key}")
        y_true = predictions["y_true"].to_numpy(dtype=float)
        y_pred = predictions["y_pred"].to_numpy(dtype=float)
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            fail(f"Chemprop predictions contain non-finite values: {key}")
        smiles = tuple(predictions["smiles"].astype(str))
        direct_test_hash = smiles_sha256(smiles)
        recorded_test_hash = require_hash(record.get("test_smiles_sha256"), f"{key} test hash")
        if direct_test_hash != recorded_test_hash:
            fail(f"Chemprop prediction SMILES do not match its test-membership hash: {key}")
        recomputed = float(np.mean(np.abs(y_true - y_pred)))
        if not math.isclose(recomputed, float(record.get("test_mae")), rel_tol=0.0, abs_tol=1e-12):
            fail(f"Chemprop prediction MAE mismatch: {key}")
        cell_hash = sha256_file(path)
        cells[key] = ChempropCell(
            endpoint=endpoint,
            split=str(split),
            data_sha256=require_hash(record.get("data_sha256"), f"{key} data hash"),
            train_smiles_sha256=require_hash(record.get("train_smiles_sha256"), f"{key} train hash"),
            validation_smiles_sha256=require_hash(record.get("validation_smiles_sha256"), f"{key} validation hash"),
            test_smiles_sha256=recorded_test_hash,
            n_test=int(record["n_test"]),
            row_index=row_index,
            smiles=smiles,
            y_true=y_true,
            y_pred=y_pred,
            test_mae=recomputed,
            cell_json_sha256=cell_hash,
            predictions_sha256=prediction_hash,
        )
        input_records.extend(
            [
                (path.relative_to(directory).as_posix(), cell_hash),
                (prediction_path.relative_to(directory).as_posix(), prediction_hash),
            ]
        )

    endpoints = sorted({endpoint for endpoint, _ in cells})
    expected_grid = {(endpoint, split) for endpoint in endpoints for split in EXPECTED_SPLITS}
    if len(endpoints) != EXPECTED_ENDPOINTS or set(cells) != expected_grid:
        fail("Chemprop endpoint/split grid is not exactly 23 x 3")
    if len(runner_hashes) != 1 or next(iter(runner_hashes)) != summary.get("cell_runner_sha256"):
        fail("Chemprop cells contain mixed runners or disagree with summary.json")

    cell_summary = pd.read_csv(cell_summary_path)
    required_cell_summary = {
        "endpoint", "split", "method", "n_test", "test_mae", "data_sha256",
        "runner_sha256", "config_sha256", "checkpoint_sha256", "predictions_sha256",
    }
    if not required_cell_summary.issubset(cell_summary.columns) or len(cell_summary) != len(cells):
        fail("Chemprop cell_summary.csv is incomplete")
    summary_keys = Counter(zip(cell_summary["endpoint"], cell_summary["split"], strict=True))
    if set(summary_keys) != set(cells) or any(count != 1 for count in summary_keys.values()):
        fail("Chemprop cell_summary.csv keys differ from cell payloads")
    for row in cell_summary.itertuples(index=False):
        key = (str(row.endpoint), str(row.split))
        cell = cells[key]
        if (
            str(row.method) != CHEMPROP_METHOD
            or int(row.n_test) != cell.n_test
            or not math.isclose(float(row.test_mae), cell.test_mae, rel_tol=0.0, abs_tol=1e-12)
            or str(row.data_sha256) != cell.data_sha256
            or str(row.predictions_sha256) != cell.predictions_sha256
        ):
            fail(f"Chemprop cell_summary.csv disagrees with payload: {key}")

    endpoint_summary = pd.read_csv(summary_csv_path)
    if len(endpoint_summary) != EXPECTED_ENDPOINTS or set(endpoint_summary["endpoint"]) != set(endpoints):
        fail("Chemprop summary.csv does not contain exactly 23 endpoints")
    for row in endpoint_summary.itertuples(index=False):
        endpoint = str(row.endpoint)
        values = [cells[(endpoint, split)].test_mae for split in EXPECTED_SPLITS]
        if (
            str(row.method) != CHEMPROP_METHOD
            or int(row.n_splits) != len(EXPECTED_SPLITS)
            or not math.isclose(float(row.mean_test_mae), float(np.mean(values)), rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(row.std_test_mae), float(np.std(values, ddof=0)), rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"Chemprop summary.csv disagrees with cells: {endpoint}")

    all_inputs = [
        ("COMPLETE", sha256_file(complete)),
        ("summary.json", sha256_file(summary_path)),
        ("summary.csv", sha256_file(summary_csv_path)),
        ("cell_summary.csv", sha256_file(cell_summary_path)),
        *input_records,
    ]
    digest = hashlib.sha256()
    for relative, value in sorted(all_inputs):
        digest.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    provenance = {
        "directory": str(directory.resolve()),
        "input_file_count": len(all_inputs),
        "canonical_input_digest_sha256": digest.hexdigest(),
        "complete_sha256": sha256_file(complete),
        "summary_json_sha256": sha256_file(summary_path),
        "summary_csv_sha256": sha256_file(summary_csv_path),
        "cell_summary_csv_sha256": sha256_file(cell_summary_path),
        "cell_runner_sha256": next(iter(runner_hashes)),
        "finalizer_sha256": summary.get("finalizer_sha256"),
        "finalizer_runtime": summary.get("finalizer_runtime"),
        "target_roundtrip_check": roundtrip,
    }
    return cells, provenance


def load_classical_contrasts(
    summary_path: Path,
    endpoint_path: Path,
    fixed_endpoint_means: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    for path in (summary_path, endpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required classical comparison evidence is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("validation", {}).get("status") != "PASS":
        fail("Classical comparison evidence is not PASS")
    by_method = {item.get("method"): item for item in summary.get("comparisons", [])}
    if set(by_method) != set(CLASSICAL_METHODS):
        fail(f"Unexpected classical contrast set: {sorted(by_method)}")
    rows = pd.read_csv(endpoint_path)
    required = {
        "endpoint", "classical_method", "chemrasayan_full_mix_mean_mae",
        "classical_mean_mae", "paired_difference_chemrasayan_minus_classical",
    }
    if not required.issubset(rows.columns) or len(rows) != EXPECTED_ENDPOINTS * len(CLASSICAL_METHODS):
        fail("Classical endpoint-pair evidence is incomplete")
    comparisons: list[dict[str, Any]] = []
    for method in CLASSICAL_METHODS:
        frame = rows.loc[rows["classical_method"].eq(method)].copy()
        if len(frame) != EXPECTED_ENDPOINTS or set(frame["endpoint"]) != set(fixed_endpoint_means):
            fail(f"Classical endpoint set mismatch: {method}")
        frame = frame.sort_values("endpoint", kind="mergesort")
        observed_fixed = frame["chemrasayan_full_mix_mean_mae"].to_numpy(dtype=float)
        expected_fixed = np.asarray([fixed_endpoint_means[value] for value in frame["endpoint"]])
        if not np.allclose(observed_fixed, expected_fixed, rtol=0.0, atol=1e-12):
            fail(f"Classical evidence uses different ChemRasayan endpoint means: {method}")
        baseline = frame["classical_mean_mae"].to_numpy(dtype=float)
        differences = observed_fixed - baseline
        recorded = frame["paired_difference_chemrasayan_minus_classical"].to_numpy(dtype=float)
        if not np.allclose(differences, recorded, rtol=0.0, atol=1e-12):
            fail(f"Classical recorded endpoint differences are inconsistent: {method}")
        wilcoxon = exact_wilcoxon_two_sided(differences)
        sign_test = exact_sign_test_two_sided(differences)
        reported = by_method[method]
        checks = (
            (float(np.mean(observed_fixed)), float(reported["chemrasayan_mean_mae"]), "ChemRasayan mean"),
            (float(np.mean(baseline)), float(reported["classical_mean_mae"]), "baseline mean"),
            (float(np.mean(differences)), float(reported["mean_difference"]), "mean difference"),
            (wilcoxon["p_two_sided_exact"], float(reported["wilcoxon"]["p_two_sided_exact"]), "Wilcoxon p"),
            (sign_test["p_two_sided_exact"], float(reported["sign_test"]["p_two_sided_exact"]), "sign p"),
        )
        for observed, expected, label in checks:
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                fail(f"Classical {label} differs from summary.json: {method}")
        comparisons.append(
            {
                "method": method,
                "kind": "classical",
                "n_endpoints": EXPECTED_ENDPOINTS,
                "chemrasayan_mean_mae": float(np.mean(observed_fixed)),
                "baseline_mean_mae": float(np.mean(baseline)),
                "mean_difference": float(np.mean(differences)),
                "bootstrap_ci_low": float(reported["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(reported["bootstrap_ci_high"]),
                "chemrasayan_point_wins": int(np.sum(differences < 0)),
                "baseline_point_wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "wilcoxon": wilcoxon,
                "sign_test": sign_test,
            }
        )
    return comparisons, {
        "summary_json_sha256": sha256_file(summary_path),
        "endpoint_paired_differences_sha256": sha256_file(endpoint_path),
    }


def run_analysis(
    *,
    chemprop_dir: Path,
    chemrasayan_dir: Path,
    classical_summary: Path,
    classical_endpoints: Path,
    output_dir: Path,
    bootstrap_replicates: int = 100_000,
    bootstrap_seed: int = 20_260_917,
) -> dict[str, Any]:
    fixed_cells, fixed_hashes = load_fixed_ft(chemrasayan_dir)
    chemprop_cells, chemprop_provenance = load_chemprop(chemprop_dir)
    target_atol = float(
        chemprop_provenance["target_roundtrip_check"]["absolute_tolerance"]
    )
    if set(fixed_cells) != set(chemprop_cells):
        fail(
            "Chemprop and ChemRasayan endpoint/split grids differ: "
            f"fixed_only={sorted(set(fixed_cells) - set(chemprop_cells))[:5]}, "
            f"chemprop_only={sorted(set(chemprop_cells) - set(fixed_cells))[:5]}"
        )

    target_max_abs_delta = 0.0
    endpoint_rows: list[dict[str, Any]] = []
    fixed_endpoint_means: dict[str, float] = {}
    chemprop_endpoint_means: dict[str, float] = {}
    endpoints = sorted({endpoint for endpoint, _ in fixed_cells})
    for endpoint in endpoints:
        fixed_values: list[float] = []
        chemprop_values: list[float] = []
        for split in EXPECTED_SPLITS:
            fixed = fixed_cells[(endpoint, split)]
            chemprop = chemprop_cells[(endpoint, split)]
            hash_pairs = (
                ("data", fixed.data_sha256, chemprop.data_sha256),
                ("train", fixed.train_smiles_sha256, chemprop.train_smiles_sha256),
                ("validation", fixed.validation_smiles_sha256, chemprop.validation_smiles_sha256),
                ("test", fixed.test_smiles_sha256, chemprop.test_smiles_sha256),
            )
            for label, left, right in hash_pairs:
                if left != right:
                    fail(f"{label} hash mismatch: {endpoint}/{split}")
            if fixed.n_test != chemprop.n_test or len(chemprop.row_index) != fixed.n_test:
                fail(f"Held-out row-count mismatch: {endpoint}/{split}")
            if not float_arrays_match_bounded(fixed.y_true, chemprop.y_true, atol=target_atol):
                fail(
                    f"Held-out target/order mismatch beyond certified absolute tolerance "
                    f"{target_atol:g}: {endpoint}/{split}"
                )
            target_max_abs_delta = max(
                target_max_abs_delta,
                float(np.max(np.abs(fixed.y_true - chemprop.y_true), initial=0.0)),
            )
            fixed_values.append(fixed.test_mae)
            chemprop_values.append(chemprop.test_mae)
        fixed_mean = float(np.mean(fixed_values))
        chemprop_mean = float(np.mean(chemprop_values))
        difference = fixed_mean - chemprop_mean
        fixed_endpoint_means[endpoint] = fixed_mean
        chemprop_endpoint_means[endpoint] = chemprop_mean
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "family": fixed_cells[(endpoint, EXPECTED_SPLITS[0])].family,
                "chemrasayan_full_mix_mean_mae": fixed_mean,
                "chemprop_dmpnn_mean_mae": chemprop_mean,
                "paired_difference_chemrasayan_minus_chemprop": difference,
                "point_estimate_winner": (
                    "ChemRasayan" if difference < 0 else "Chemprop" if difference > 0 else "tie"
                ),
            }
        )

    chemprop_differences = np.asarray(
        [fixed_endpoint_means[endpoint] - chemprop_endpoint_means[endpoint] for endpoint in endpoints]
    )
    chemprop_wilcoxon = exact_wilcoxon_two_sided(chemprop_differences)
    chemprop_sign = exact_sign_test_two_sided(chemprop_differences)
    ci_low, ci_high = percentile_bootstrap_mean_ci(
        chemprop_differences,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    chemprop_comparison = {
        "method": CHEMPROP_METHOD,
        "kind": "neural",
        "n_endpoints": EXPECTED_ENDPOINTS,
        "chemrasayan_mean_mae": float(np.mean(list(fixed_endpoint_means.values()))),
        "baseline_mean_mae": float(np.mean(list(chemprop_endpoint_means.values()))),
        "mean_difference": float(np.mean(chemprop_differences)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "chemrasayan_point_wins": int(np.sum(chemprop_differences < 0)),
        "baseline_point_wins": int(np.sum(chemprop_differences > 0)),
        "ties": int(np.sum(chemprop_differences == 0)),
        "wilcoxon": chemprop_wilcoxon,
        "sign_test": chemprop_sign,
    }

    classical, classical_provenance = load_classical_contrasts(
        classical_summary, classical_endpoints, fixed_endpoint_means
    )
    comparisons = [*classical, chemprop_comparison]
    wilcoxon_holm = holm_adjust(
        [item["wilcoxon"]["p_two_sided_exact"] for item in comparisons]
    )
    sign_holm = holm_adjust(
        [item["sign_test"]["p_two_sided_exact"] for item in comparisons]
    )
    comparison_rows: list[dict[str, Any]] = []
    for index, item in enumerate(comparisons):
        item["wilcoxon"]["p_holm_six_contrasts"] = wilcoxon_holm[index]
        item["wilcoxon"]["reject_holm_six_0_05"] = wilcoxon_holm[index] < 0.05
        item["sign_test"]["p_holm_six_contrasts"] = sign_holm[index]
        item["sign_test"]["reject_holm_six_0_05"] = sign_holm[index] < 0.05
        comparison_rows.append(
            {
                "method": item["method"],
                "kind": item["kind"],
                "n_endpoints": item["n_endpoints"],
                "chemrasayan_mean_mae": item["chemrasayan_mean_mae"],
                "baseline_mean_mae": item["baseline_mean_mae"],
                "mean_difference_chemrasayan_minus_baseline": item["mean_difference"],
                "bootstrap_95pct_ci_low": item["bootstrap_ci_low"],
                "bootstrap_95pct_ci_high": item["bootstrap_ci_high"],
                "chemrasayan_point_wins": item["chemrasayan_point_wins"],
                "baseline_point_wins": item["baseline_point_wins"],
                "ties": item["ties"],
                "wilcoxon_w": item["wilcoxon"]["statistic"],
                "wilcoxon_p_exact_two_sided": item["wilcoxon"]["p_two_sided_exact"],
                "wilcoxon_p_holm_six": item["wilcoxon"]["p_holm_six_contrasts"],
                "sign_p_exact_two_sided": item["sign_test"]["p_two_sided_exact"],
                "sign_p_holm_six": item["sign_test"]["p_holm_six_contrasts"],
            }
        )

    script_path = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "contrast_direction": (
            "ChemRasayan fixed full-fine-tuning MAE minus baseline MAE; negative favors ChemRasayan"
        ),
        "validation": {
            "endpoint_count": len(endpoints),
            "split_count_per_endpoint": len(EXPECTED_SPLITS),
            "aligned_endpoint_split_pairs": len(fixed_cells),
            "chemprop_cells_recomputed": len(chemprop_cells),
            "membership_hash_fields_matched_per_pair": 4,
            "membership_hash_comparisons_passed": len(fixed_cells) * 4,
            "held_out_row_counts_matched": len(fixed_cells),
            "held_out_target_arrays_matched": len(fixed_cells),
            "held_out_target_max_abs_delta": target_max_abs_delta,
            "held_out_target_tolerance": (
                f"rtol=0, atol={target_atol:g}; bounded by the certified finalizer contract"
            ),
            "outer_test_targets_blinded": True,
        },
        "chemprop_comparison": chemprop_comparison,
        "all_six_contrasts": comparisons,
        "multiplicity": {
            "method": "Holm step-down adjustment",
            "families": (
                "six same-split OpenADMET baseline contrasts, separately for exact paired "
                "Wilcoxon and exact sign-test p-values"
            ),
            "methods": [item["method"] for item in comparisons],
        },
        "bootstrap": {
            "unit": "23 paired endpoint means; each endpoint first averages three locked splits",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interval": "two-sided percentile interval at 2.5% and 97.5%",
        },
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "chemprop": chemprop_provenance,
            "chemrasayan_directory": str(chemrasayan_dir.resolve()),
            "chemrasayan_result_sha256": fixed_hashes,
            "classical": classical_provenance,
        },
    }

    endpoint_fields = [
        "endpoint", "family", "chemrasayan_full_mix_mean_mae", "chemprop_dmpnn_mean_mae",
        "paired_difference_chemrasayan_minus_chemprop", "point_estimate_winner",
    ]
    comparison_fields = [
        "method", "kind", "n_endpoints", "chemrasayan_mean_mae", "baseline_mean_mae",
        "mean_difference_chemrasayan_minus_baseline", "bootstrap_95pct_ci_low",
        "bootstrap_95pct_ci_high", "chemrasayan_point_wins", "baseline_point_wins", "ties",
        "wilcoxon_w", "wilcoxon_p_exact_two_sided", "wilcoxon_p_holm_six",
        "sign_p_exact_two_sided", "sign_p_holm_six",
    ]
    atomic_csv(output_dir / "chemprop_endpoint_paired_differences.csv", endpoint_rows, endpoint_fields)
    atomic_csv(output_dir / "all_six_comparisons.csv", comparison_rows, comparison_fields)
    atomic_json(output_dir / "summary.json", summary)

    anonymous_summary = {
        "schema_version": summary["schema_version"],
        "status": summary["status"],
        "contrast_direction": summary["contrast_direction"],
        "validation": summary["validation"],
        "chemprop_comparison": summary["chemprop_comparison"],
        "all_six_contrasts": summary["all_six_contrasts"],
        "multiplicity": summary["multiplicity"],
        "bootstrap": summary["bootstrap"],
        "provenance": {
            "analysis_script_sha256": summary["provenance"]["script_sha256"],
            "chemprop_input_file_count": chemprop_provenance["input_file_count"],
            "chemprop_canonical_input_digest_sha256": chemprop_provenance[
                "canonical_input_digest_sha256"
            ],
            "chemprop_complete_sha256": chemprop_provenance["complete_sha256"],
            "chemprop_summary_json_sha256": chemprop_provenance["summary_json_sha256"],
            "chemprop_summary_csv_sha256": chemprop_provenance["summary_csv_sha256"],
            "chemprop_cell_summary_csv_sha256": chemprop_provenance[
                "cell_summary_csv_sha256"
            ],
            "chemprop_cell_runner_sha256": chemprop_provenance["cell_runner_sha256"],
            "chemprop_finalizer_sha256": chemprop_provenance["finalizer_sha256"],
            "chemprop_finalizer_runtime": chemprop_provenance["finalizer_runtime"],
            "chemprop_target_roundtrip_check": chemprop_provenance[
                "target_roundtrip_check"
            ],
            "chemrasayan_result_sha256": fixed_hashes,
            "classical_summary_json_sha256": classical_provenance["summary_json_sha256"],
            "classical_endpoint_paired_differences_sha256": classical_provenance[
                "endpoint_paired_differences_sha256"
            ],
        },
    }
    validate_anonymous_payload(anonymous_summary)
    atomic_json(output_dir / "summary_anonymous.json", anonymous_summary)

    md = [
        "# Same-split OpenADMET Chemprop comparison",
        "",
        "**Validation: PASS.** All 69 Chemprop cells match ChemRasayan fixed full fine-tuning "
        "on prepared-data, inner-train, validation, and outer-test membership hashes, held-out "
        "row counts, and held-out targets.",
        "",
        "Each endpoint first averages the same three split MAEs; inference and bootstrap "
        "resampling use the 23 paired endpoint means, not the 69 split cells.",
        "",
        "| Baseline | Type | ChemRasayan | Baseline | Difference (95% endpoint bootstrap CI) | ChemR. wins | Wilcoxon raw / Holm-6 | Sign raw / Holm-6 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        md.append(
            f"| `{row['method']}` | {row['kind']} | {row['chemrasayan_mean_mae']:.4f} | "
            f"{row['baseline_mean_mae']:.4f} | "
            f"{row['mean_difference_chemrasayan_minus_baseline']:+.4f} "
            f"[{row['bootstrap_95pct_ci_low']:+.4f}, {row['bootstrap_95pct_ci_high']:+.4f}] | "
            f"{row['chemrasayan_point_wins']}/{row['n_endpoints']} | "
            f"{row['wilcoxon_p_exact_two_sided']:.6g} / {row['wilcoxon_p_holm_six']:.6g} | "
            f"{row['sign_p_exact_two_sided']:.6g} / {row['sign_p_holm_six']:.6g} |"
        )
    md.extend(
        [
            "",
            f"The Chemprop bootstrap interval uses {bootstrap_replicates:,} deterministic paired "
            f"endpoint resamples (seed {bootstrap_seed}). Classical intervals are retained from "
            "the audited five-contrast analysis; all raw exact p-values were recomputed from "
            "endpoint differences before the six-contrast Holm adjustment.",
            "",
            "Unrounded certified Chemprop difference and 95% interval: "
            f"{chemprop_comparison['mean_difference']!r} "
            f"[{chemprop_comparison['bootstrap_ci_low']!r}, "
            f"{chemprop_comparison['bootstrap_ci_high']!r}].",
            "",
            "Point-estimate wins are descriptive endpoint directions, not per-endpoint tests.",
        ]
    )
    atomic_text(output_dir / "RESULTS.md", "\n".join(md) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chemprop-dir",
        type=Path,
        default=Path("results_package/results_openadmet_chemprop_20260917_r4"),
    )
    parser.add_argument(
        "--chemrasayan-dir", type=Path, default=Path("results_rtd_phase4_95m_moljepa")
    )
    parser.add_argument(
        "--classical-summary",
        type=Path,
        default=Path("paper_iclr/evidence/openadmet_classical_comparison/summary.json"),
    )
    parser.add_argument(
        "--classical-endpoints",
        type=Path,
        default=Path(
            "paper_iclr/evidence/openadmet_classical_comparison/endpoint_paired_differences.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_iclr/evidence/openadmet_chemprop_comparison"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_917)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        chemprop_dir=args.chemprop_dir,
        chemrasayan_dir=args.chemrasayan_dir,
        classical_summary=args.classical_summary,
        classical_endpoints=args.classical_endpoints,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    comparison = result["chemprop_comparison"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "chemprop_comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
