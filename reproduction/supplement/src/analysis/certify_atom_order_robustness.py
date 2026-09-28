#!/usr/bin/env python3
"""Independently certify the OpenADMET atom-order robustness experiment.

The certifier is deliberately fail closed.  It reconstructs every prepared
train/validation/test membership, regenerates every atom-renumbered SMILES with
the experiment's pinned RDKit version, recomputes all prediction statistics,
and writes evidence only after the entire 23 x 3 x 10 contract passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


EXPECTED_ENDPOINTS = (
    "expansion_caco2_pappa", "expansion_caco2_efflux", "expansion_logd",
    "expansion_ksol", "expansion_hlm", "expansion_mlm", "expansion_mbpb",
    "expansion_mgmb", "expansion_mppb", "asap_mers", "asap_sars",
    "asap_logd", "asap_ksol", "asap_hlm", "asap_mlm", "asap_mdr1",
    "pxr", "biogen_solubility", "biogen_hlm", "biogen_rlm",
    "biogen_hppb", "biogen_rppb", "biogen_mdr1",
)
EXPECTED_SPLITS = ("split1", "split2", "split3")
EXPECTED_SHARDS = tuple(range(8))
EXPECTED_CHECKPOINT_SHA256 = (
    "7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1"
)
EXPECTED_RDKIT_VERSION = "2025.03.6"
EXPECTED_PROTOCOL = (
    "RDKit RenumberAtoms followed by non-canonical isomeric SMILES; "
    "canonical isomeric identity asserted"
)
SCOPE = (
    "fresh fixed-seed full_mix replicate measuring test-time serialization "
    "sensitivity; not the original head, not proof of invariance, and not a "
    "no-position causal ablation"
)
HASH_RE = re.compile(r"[0-9a-f]{64}")
ISO_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
FLOAT_ATOL = 1e-9


@dataclass(frozen=True)
class SplitEvidence:
    canonical_mae: float
    mean_permuted_mae: float
    mae_change: float
    mean_prediction_sd: float
    maximum_prediction_delta: float
    test_rows: int
    mean_unique_serializations: float
    fraction_molecules_with_multiple_serializations: float
    ordered_test_rows_sha256: str
    permutation_ordered_sha256: tuple[str, ...]


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
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def sorted_smiles_sha256(values: Iterable[str]) -> str:
    """The exact hash function used by finetune_moljepa_benchmarks.py."""
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def ordered_strings_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def ordered_rows_sha256(smiles: Sequence[str], targets: Sequence[float]) -> str:
    rows = [[str(smile), float(target)] for smile, target in zip(smiles, targets, strict=True)]
    return stable_hash(rows)


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


def finite_array(value: Any, label: str, expected_length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        fail(f"{label} must be a finite one-dimensional array")
    if expected_length is not None and len(array) != expected_length:
        fail(f"{label} has {len(array)} rows; expected {expected_length}")
    return array


def close(left: float, right: Any, label: str, atol: float = FLOAT_ATOL) -> None:
    try:
        observed = float(right)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric")
    if not math.isfinite(observed) or not math.isclose(
        float(left), observed, rel_tol=0.0, abs_tol=atol
    ):
        fail(f"{label} mismatch: recomputed={left!r}, stored={right!r}")


def arrays_match_one_ulp(left: Sequence[float], right: Sequence[float]) -> bool:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    tolerance = np.maximum(np.abs(np.spacing(a)), np.abs(np.spacing(b)))
    return bool(np.all(np.abs(a - b) <= tolerance))


def expected_run_config() -> dict[str, Any]:
    return {
        "method": "full_mix",
        "learning_rate": 1e-5,
        "epochs": 60,
        "patience": 12,
        "batch_size": 64,
        "validation_fraction": 0.15,
        "head_hidden": 512,
        "mix_layers": [4, 8, 10, 12],
        "weight_decay": 1e-4,
        "use_bf16": True,
        "lora_rank": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "atom_order_permutations": 10,
    }


def cluster_validation_indices(train_frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact copy of the experiment's deterministic cluster holdout logic."""
    groups = train_frame.groupby("cluster_index").indices
    cluster_ids = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)
    target = max(1, round(len(train_frame) * 0.15))
    selected: list[Any] = []
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
        fail("cluster validation split produced an empty train or validation set")
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)


def randomized_atom_order_smiles(values: Sequence[str], seed: int) -> list[str]:
    """Exact experiment algorithm, imported lazily so unit tests need no RDKit."""
    try:
        import rdkit
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for production certification; use the experiment runtime "
            f"with rdkit=={EXPECTED_RDKIT_VERSION}"
        ) from exc
    if rdkit.__version__ != EXPECTED_RDKIT_VERSION:
        raise RuntimeError(
            f"RDKit {rdkit.__version__} is installed, but exact regeneration requires "
            f"rdkit=={EXPECTED_RDKIT_VERSION}"
        )
    rng = np.random.default_rng(seed)
    randomized: list[str] = []
    for row, value in enumerate(map(str, values)):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            fail(f"invalid SMILES in atom-order audit at row {row}: {value}")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        order = rng.permutation(mol.GetNumAtoms()).astype(int).tolist()
        candidate = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, order), canonical=False, isomericSmiles=True
        )
        check = Chem.MolFromSmiles(candidate)
        if check is None or Chem.MolToSmiles(
            check, canonical=True, isomericSmiles=True
        ) != canonical:
            fail(f"atom renumbering changed canonical isomeric identity at row {row}")
        randomized.append(candidate)
    return randomized


def exact_wilcoxon_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    raw = finite_array(list(differences), "Wilcoxon differences")
    nonzero = raw[raw != 0.0]
    if not len(nonzero):
        return {"n_nonzero": 0, "zero_pairs": int(len(raw)), "statistic": 0.0,
                "w_plus": 0.0, "w_minus": 0.0, "p_two_sided_exact": 1.0,
                "definition": "exact sign-randomization distribution; doubled smaller tail"}
    absolute = np.abs(nonzero)
    order = np.argsort(absolute, kind="mergesort")
    ranks = np.empty(len(absolute), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and absolute[order[stop]] == absolute[order[start]]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    doubled = np.rint(ranks * 2).astype(int)
    counts: dict[int, int] = {0: 1}
    for rank in doubled:
        updated = counts.copy()
        for subtotal, count in counts.items():
            updated[subtotal + int(rank)] = updated.get(subtotal + int(rank), 0) + count
        counts = updated
    observed_plus = int(doubled[nonzero > 0].sum())
    total = int(doubled.sum())
    boundary = min(observed_plus, total - observed_plus)
    lower = sum(count for score, count in counts.items() if score <= boundary)
    p_value = min(1.0, 2.0 * lower / (2 ** len(nonzero)))
    return {"n_nonzero": int(len(nonzero)), "zero_pairs": int(len(raw)-len(nonzero)),
            "statistic": boundary/2.0, "w_plus": observed_plus/2.0,
            "w_minus": (total-observed_plus)/2.0, "p_two_sided_exact": float(p_value),
            "definition": "exact sign-randomization distribution; doubled smaller tail"}


def exact_sign_test_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    raw = finite_array(list(differences), "sign-test differences")
    positive = int(np.sum(raw > 0.0))
    negative = int(np.sum(raw < 0.0))
    nonzero = positive + negative
    if nonzero:
        smaller = min(positive, negative)
        lower = sum(math.comb(nonzero, k) for k in range(smaller + 1)) / (2 ** nonzero)
        p_value = min(1.0, 2.0 * lower)
    else:
        p_value = 1.0
    return {"n_nonzero": nonzero, "positive_differences": positive,
            "negative_differences": negative, "zero_pairs": int(len(raw)-nonzero),
            "p_two_sided_exact": float(p_value),
            "definition": "exact Binomial(n, 0.5); doubled smaller tail; zero pairs omitted"}


def bootstrap_mean_ci(values: Sequence[float], replicates: int, seed: int) -> tuple[float, float]:
    array = finite_array(values, "bootstrap values")
    if replicates < 1 or not len(array):
        fail("bootstrap needs a positive replicate count and nonempty values")
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high)


def validate_anonymous_text(text: str) -> None:
    forbidden = {
        "Windows absolute path": re.compile(r"[A-Za-z]:[\\/]"),
        "private storage path": re.compile(r"/(?:mnt|nfs|home|Users)/", re.I),
        "cluster account": re.compile(r"\bs\d{6,}-(?:eidf\d+|[a-z]+ns\d+)\b", re.I),
        "private IPv4 address": re.compile(
            r"(?<!\d)(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}(?!\d)"
        ),
    }
    for label, pattern in forbidden.items():
        if pattern.search(text):
            fail(f"anonymous artifact contains a {label}")


def validate_cell(
    cell: dict[str, Any], reference: dict[str, Any], test_frame: pd.DataFrame,
    train_frame: pd.DataFrame, valid_frame: pd.DataFrame, *, endpoint: str,
    split: str, permutation_function: Callable[[Sequence[str], int], list[str]],
) -> SplitEvidence:
    """Validate one cell.  The injectable permutation function enables dependency-light tests."""
    label = f"{endpoint}/{split}"
    expected_seed = 9100 + (EXPECTED_SPLITS.index(split) + 1) * 100 + 2
    for key, expected in (("task", endpoint), ("split", split), ("method", "full_mix"),
                          ("seed", expected_seed)):
        if cell.get(key) != expected:
            fail(f"{label}: {key} mismatch")
    if cell.get("run_config") != expected_run_config():
        fail(f"{label}: run configuration mismatch")
    reference_config = dict(reference.get("run_config", {}))
    reference_config.pop("atom_order_permutations", None)
    result_base_config = dict(cell["run_config"])
    result_base_config.pop("atom_order_permutations", None)
    if reference_config != result_base_config:
        fail(f"{label}: base configuration differs from canonical reference")
    for key in ("checkpoint_sha256", "data_sha256"):
        if cell.get(key) != reference.get(key):
            fail(f"{label}: {key} differs from canonical reference")
    if cell.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        fail(f"{label}: unexpected Phase-4 checkpoint")

    canonical_smiles = tuple(map(str, test_frame["smiles"].tolist()))
    test_targets = finite_array(test_frame["y"].tolist(), f"{label} prepared targets")
    y_true = finite_array(cell.get("y_true"), f"{label} y_true", len(test_frame))
    y_pred = finite_array(cell.get("y_pred"), f"{label} y_pred", len(test_frame))
    if not arrays_match_one_ulp(y_true, test_targets):
        fail(f"{label}: held-out targets differ from prepared data")
    if not arrays_match_one_ulp(y_true, reference.get("y_true", [])):
        fail(f"{label}: held-out targets differ from canonical reference")
    for role, frame, key in (
        ("train", train_frame, "train_smiles_sha256"),
        ("validation", valid_frame, "validation_smiles_sha256"),
        ("test", test_frame, "test_smiles_sha256"),
    ):
        observed = sorted_smiles_sha256(frame["smiles"])
        if cell.get(key) != observed or reference.get(key) != observed:
            fail(f"{label}: {role} membership hash mismatch")
    for key, expected in (("train_rows", len(train_frame)),
                          ("validation_rows", len(valid_frame)),
                          ("test_rows", len(test_frame))):
        if cell.get(key) != expected or reference.get(key) != expected:
            fail(f"{label}: {key} mismatch")

    canonical_mae = float(np.mean(np.abs(y_true - y_pred)))
    close(canonical_mae, cell.get("test_mae"), f"{label} canonical MAE")
    audit = cell.get("atom_order_robustness")
    if not isinstance(audit, dict):
        fail(f"{label}: atom-order audit is absent")
    if audit.get("protocol") != EXPECTED_PROTOCOL or audit.get("permutations") != 10:
        fail(f"{label}: atom-order protocol mismatch")
    close(canonical_mae, audit.get("canonical_test_mae"), f"{label} audit canonical MAE")
    records = audit.get("records")
    if not isinstance(records, list) or [r.get("permutation_index") for r in records] != list(range(10)):
        fail(f"{label}: expected ordered permutation indices 0..9")

    prediction_matrix: list[np.ndarray] = []
    regenerated_matrix: list[list[str]] = []
    ordered_hashes: list[str] = []
    permuted_maes: list[float] = []
    for index, record in enumerate(records):
        seed = expected_seed * 1000 + index
        if record.get("seed") != seed:
            fail(f"{label}/permutation-{index}: seed mismatch")
        regenerated = permutation_function(canonical_smiles, seed)
        if len(regenerated) != len(canonical_smiles):
            fail(f"{label}/permutation-{index}: regenerated row count mismatch")
        regenerated = list(map(str, regenerated))
        regenerated_matrix.append(regenerated)
        ordered_hashes.append(ordered_strings_sha256(regenerated))
        if record.get("smiles_sha256") != sorted_smiles_sha256(regenerated):
            fail(f"{label}/permutation-{index}: sorted SMILES hash mismatch")
        changed = sum(a != b for a, b in zip(canonical_smiles, regenerated, strict=True))
        if record.get("changed_smiles") != changed:
            fail(f"{label}/permutation-{index}: changed-SMILES count mismatch")
        close(changed / len(canonical_smiles), record.get("fraction_changed_smiles"),
              f"{label}/permutation-{index} changed fraction", atol=1e-12)
        permutation_prediction = finite_array(
            record.get("y_pred"), f"{label}/permutation-{index} y_pred", len(y_true)
        )
        prediction_matrix.append(permutation_prediction)
        permuted_mae = float(np.mean(np.abs(y_true - permutation_prediction)))
        permuted_maes.append(permuted_mae)
        close(permuted_mae, record.get("test_mae"), f"{label}/permutation-{index} MAE")
        absolute_delta = np.abs(permutation_prediction - y_pred)
        close(float(absolute_delta.mean()), record.get("mean_absolute_prediction_delta"),
              f"{label}/permutation-{index} mean prediction delta")
        close(float(absolute_delta.max()), record.get("maximum_absolute_prediction_delta"),
              f"{label}/permutation-{index} maximum prediction delta")

    if len(set(ordered_hashes)) != 10:
        fail(f"{label}: the ten serialized test sets are not distinct")
    unique_counts = np.asarray([
        len(set(strings)) for strings in zip(*regenerated_matrix, strict=True)
    ], dtype=float)
    if not np.any(unique_counts > 1):
        fail(f"{label}: atom renumbering produced no effective serialization diversity")
    stacked = np.stack(prediction_matrix)
    recomputed_sd = np.std(stacked, axis=0)
    recomputed_max = np.max(np.abs(stacked - y_pred[None, :]), axis=0)
    stored_sd = finite_array(audit.get("per_molecule_prediction_sd"),
                             f"{label} per-molecule SD", len(y_true))
    stored_max = finite_array(audit.get("per_molecule_maximum_absolute_delta"),
                              f"{label} per-molecule maximum", len(y_true))
    if not np.allclose(stored_sd, recomputed_sd, rtol=0.0, atol=FLOAT_ATOL):
        fail(f"{label}: per-molecule prediction SD mismatch")
    if not np.allclose(stored_max, recomputed_max, rtol=0.0, atol=FLOAT_ATOL):
        fail(f"{label}: per-molecule maximum prediction delta mismatch")
    mean_permuted = float(np.mean(permuted_maes))
    close(mean_permuted, audit.get("mean_permuted_test_mae"),
          f"{label} mean permuted MAE")
    close(mean_permuted - canonical_mae, audit.get("mean_mae_change"),
          f"{label} mean MAE change")
    return SplitEvidence(
        canonical_mae=canonical_mae,
        mean_permuted_mae=mean_permuted,
        mae_change=mean_permuted - canonical_mae,
        mean_prediction_sd=float(recomputed_sd.mean()),
        maximum_prediction_delta=float(recomputed_max.max()),
        test_rows=len(y_true),
        mean_unique_serializations=float(unique_counts.mean()),
        fraction_molecules_with_multiple_serializations=float(np.mean(unique_counts > 1)),
        ordered_test_rows_sha256=ordered_rows_sha256(canonical_smiles, y_true),
        permutation_ordered_sha256=tuple(ordered_hashes),
    )


def parse_utc_marker(path: Path) -> datetime:
    text = path.read_text(encoding="utf-8").strip()
    if ISO_UTC_RE.fullmatch(text) is None:
        fail(f"invalid UTC completion marker: {path.name}")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def certify(
    results_dir: Path, prepared_dir: Path, reference_dir: Path, output_dir: Path,
    *, runner_source: Path, wrapper_source: Path, bootstrap_replicates: int = 100_000,
    bootstrap_seed: int = 20_260_917,
    permutation_function: Callable[[Sequence[str], int], list[str]] = randomized_atom_order_smiles,
) -> dict[str, Any]:
    for label, directory in (("results", results_dir), ("prepared", prepared_dir),
                             ("reference", reference_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {directory}")
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(prepared_manifest.get("paper_split_settings", {}).get("splits", ())) != EXPECTED_SPLITS:
        fail("prepared manifest split contract mismatch")
    if prepared_manifest.get("protocol") != "reconstructed_butina_public_snapshot":
        fail("prepared manifest protocol mismatch")
    endpoint_manifest = prepared_manifest.get("endpoints")
    if not isinstance(endpoint_manifest, dict) or set(endpoint_manifest) != set(EXPECTED_ENDPOINTS):
        fail("prepared manifest endpoint set mismatch")
    csv_names = {path.name for path in prepared_dir.glob("*.csv")}
    if csv_names != {f"{endpoint}.csv" for endpoint in EXPECTED_ENDPOINTS}:
        fail("prepared CSV inventory is not exactly the expected 23 files")

    expected_result_names = {f"results_{endpoint}.json" for endpoint in EXPECTED_ENDPOINTS}
    for label, directory in (("atom-order", results_dir), ("reference", reference_dir)):
        names = {path.name for path in directory.glob("results_*.json")}
        if names != expected_result_names:
            fail(f"{label} result inventory mismatch: missing={sorted(expected_result_names-names)}, "
                 f"extra={sorted(names-expected_result_names)}")
    expected_cells = {
        f"{endpoint}/cells/{split}_full_mix.json"
        for endpoint in EXPECTED_ENDPOINTS for split in EXPECTED_SPLITS
    }
    observed_cells = {
        path.relative_to(results_dir).as_posix()
        for path in results_dir.glob("*/cells/*.json")
    }
    if observed_cells != expected_cells:
        fail(f"atom-order cell inventory is not exactly 69 files: "
             f"missing={sorted(expected_cells-observed_cells)}, "
             f"extra={sorted(observed_cells-expected_cells)}")

    launch_names = {f"launch_manifest_shard_{index}.txt" for index in EXPECTED_SHARDS}
    marker_names = {f"SHARD_{index}_COMPLETE" for index in EXPECTED_SHARDS}
    if {p.name for p in results_dir.glob("launch_manifest_shard_*.txt")} != launch_names:
        fail("launch-manifest inventory is not exactly eight indexed shards")
    if {p.name for p in results_dir.glob("SHARD_*_COMPLETE")} != marker_names:
        fail("completion-marker inventory is not exactly eight indexed shards")
    contracts = {
        tuple((results_dir / name).read_text(encoding="utf-8").strip().split())
        for name in launch_names
    }
    if len(contracts) != 1:
        fail("shards used mixed launch contracts")
    launch_contract = next(iter(contracts))
    if len(launch_contract) != 4 or any(HASH_RE.fullmatch(item) is None for item in launch_contract):
        fail("launch contract must contain four lowercase SHA-256 digests")
    checkpoint_sha, runner_sha, wrapper_sha, source_bundle_sha = launch_contract
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        fail("launch contract used an unexpected checkpoint")
    if sha256_file(runner_source) != runner_sha:
        fail("local runner source does not match the launched runner hash")
    if sha256_file(wrapper_source) != wrapper_sha:
        fail("local wrapper source does not match the launched wrapper hash")
    wrapper_text = wrapper_source.read_text(encoding="utf-8")
    if "rdkit==2025.3.6" not in wrapper_text:
        fail("launched wrapper does not pin rdkit==2025.3.6")
    shard_times = {
        index: parse_utc_marker(results_dir / f"SHARD_{index}_COMPLETE")
        for index in EXPECTED_SHARDS
    }
    launch_mtimes = {
        index: datetime.fromtimestamp(
            (results_dir / f"launch_manifest_shard_{index}.txt").stat().st_mtime,
            tz=timezone.utc,
        )
        for index in EXPECTED_SHARDS
    }
    complete_path = results_dir / "COMPLETE"
    if not complete_path.is_file():
        fail("global COMPLETE marker is absent")
    complete_time = parse_utc_marker(complete_path)
    if complete_time < max(shard_times.values()):
        fail("global COMPLETE predates a shard completion marker")

    checkpoint_hashes: set[str] = set()
    runner_hashes: set[str] = set()
    software_contracts: set[str] = set()
    result_file_hashes: dict[str, str] = {}
    cell_file_hashes: dict[str, str] = {}
    endpoint_rows: list[dict[str, Any]] = []
    split_count = 0
    permutation_count = 0
    for endpoint_index, endpoint in enumerate(EXPECTED_ENDPOINTS):
        result_path = results_dir / f"results_{endpoint}.json"
        reference_path = reference_dir / f"results_{endpoint}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        result_file_hashes[result_path.name] = sha256_file(result_path)
        if result.get("task") != endpoint or reference.get("task") != endpoint:
            fail(f"{endpoint}: result task identity mismatch")
        if result.get("schema_version") != 1 or result.get("primary_metric") != "mae":
            fail(f"{endpoint}: result schema/metric mismatch")
        for key in ("family", "display_name", "data_protocol",
                    "exact_table3_split_reproduction", "source_count_matches_paper"):
            if result.get(key) != reference.get(key):
                fail(f"{endpoint}: top-level {key} differs from canonical reference")
        if result.get("data_protocol") != prepared_manifest["protocol"]:
            fail(f"{endpoint}: result/prepared protocol mismatch")
        if result.get("exact_table3_split_reproduction") != \
                prepared_manifest.get("exact_table3_split_reproduction"):
            fail(f"{endpoint}: exact-split flag differs from prepared manifest")
        if result.get("family") != endpoint_manifest[endpoint].get("family") or \
                result.get("display_name") != endpoint_manifest[endpoint].get("display_name"):
            fail(f"{endpoint}: result/prepared endpoint metadata mismatch")
        if result.get("source_count_matches_paper") != \
                endpoint_manifest[endpoint].get("source_count_matches_paper"):
            fail(f"{endpoint}: source-row-count flag differs from prepared manifest")
        if set(result.get("run_configs", {})) != {"full_mix"} or \
                result["run_configs"]["full_mix"] != expected_run_config():
            fail(f"{endpoint}: top-level run configuration mismatch")
        if set(result.get("method_summary", {})) != {"full_mix"}:
            fail(f"{endpoint}: expected only full_mix method summary")
        if set(result.get("splits", {})) != set(EXPECTED_SPLITS):
            fail(f"{endpoint}: split inventory mismatch")
        checkpoint_hashes.add(require_hash(result.get("checkpoint_sha256"), f"{endpoint} checkpoint"))
        runner_hashes.add(require_hash(result.get("runner_sha256"), f"{endpoint} runner"))
        software_versions = result.get("software_versions")
        if not isinstance(software_versions, dict) or not {
            "torch", "cuda", "torch_geometric", "scikit_learn", "numpy", "pandas"
        }.issubset(software_versions):
            fail(f"{endpoint}: incomplete software-version metadata")
        if "rdkit" in software_versions and software_versions["rdkit"] != EXPECTED_RDKIT_VERSION:
            fail(f"{endpoint}: result metadata reports an unexpected RDKit version")
        software_contracts.add(stable_hash(software_versions))
        data_path = prepared_dir / f"{endpoint}.csv"
        endpoint_meta = endpoint_manifest[endpoint]
        data_sha = sha256_file(data_path)
        if endpoint_meta.get("prepared_sha256") != data_sha:
            fail(f"{endpoint}: prepared CSV hash mismatch")
        frame = pd.read_csv(data_path)
        required_columns = {"smiles", "y", "cluster_index", *EXPECTED_SPLITS}
        if required_columns.difference(frame.columns):
            fail(f"{endpoint}: prepared CSV lacks required columns")
        if frame["smiles"].isna().any() or not np.isfinite(pd.to_numeric(frame["y"])).all():
            fail(f"{endpoint}: invalid SMILES/target values in prepared CSV")

        split_evidence: list[SplitEvidence] = []
        for split_index, split in enumerate(EXPECTED_SPLITS, start=1):
            if set(frame[split].unique()) != {"train", "test"}:
                fail(f"{endpoint}/{split}: prepared membership values mismatch")
            paper_train = frame[frame[split].eq("train")].reset_index(drop=True)
            test_frame = frame[frame[split].eq("test")].reset_index(drop=True)
            prepared_split = endpoint_meta.get("splits", {}).get(split, {})
            if prepared_split.get("actual_train") != len(paper_train) or \
                    prepared_split.get("actual_test") != len(test_frame):
                fail(f"{endpoint}/{split}: prepared manifest row counts mismatch CSV")
            inner_idx, valid_idx = cluster_validation_indices(paper_train, 7300 + split_index)
            train_frame = paper_train.iloc[inner_idx].reset_index(drop=True)
            valid_frame = paper_train.iloc[valid_idx].reset_index(drop=True)
            split_record = result["splits"][split]
            reference_split = reference.get("splits", {}).get(split, {})
            if split_record.get("selected_method") != "full_mix" or \
                    set(split_record.get("methods", {})) != {"full_mix"}:
                fail(f"{endpoint}/{split}: only full_mix must be present and selected")
            cell = split_record["methods"]["full_mix"]
            reference_cell = reference_split.get("methods", {}).get("full_mix")
            if not isinstance(reference_cell, dict):
                fail(f"{endpoint}/{split}: canonical full_mix reference is absent")
            cell_path = results_dir / endpoint / "cells" / f"{split}_full_mix.json"
            cell_payload = json.loads(cell_path.read_text(encoding="utf-8"))
            if cell_payload != cell:
                fail(f"{endpoint}/{split}: nested cell differs from top-level result")
            relative_cell = cell_path.relative_to(results_dir).as_posix()
            cell_file_hashes[relative_cell] = sha256_file(cell_path)
            if cell.get("data_sha256") != data_sha:
                fail(f"{endpoint}/{split}: result cell prepared-data hash mismatch")
            if cell.get("runner_sha256") != runner_sha or \
                    cell.get("checkpoint_sha256") != checkpoint_sha:
                fail(f"{endpoint}/{split}: cell provenance differs from launch contract")
            for key in ("architecture", "trainable_parameters", "total_parameters"):
                if cell.get(key) != reference_cell.get(key):
                    fail(f"{endpoint}/{split}: {key} differs from canonical reference")
            if cell.get("learning_rate") != 1e-5:
                fail(f"{endpoint}/{split}: learning rate field mismatch")
            if not isinstance(cell.get("best_epoch"), int) or cell["best_epoch"] < 1 or \
                    cell["best_epoch"] > 60 or not math.isfinite(float(cell.get("validation_mae", math.nan))):
                fail(f"{endpoint}/{split}: invalid fitted-cell training metadata")
            evidence = validate_cell(
                cell, reference_cell, test_frame, train_frame, valid_frame,
                endpoint=endpoint, split=split, permutation_function=permutation_function,
            )
            close(cell["validation_mae"], split_record.get("selected_validation_mae"),
                  f"{endpoint}/{split} selected validation MAE")
            close(evidence.canonical_mae, split_record.get("selected_test_mae"),
                  f"{endpoint}/{split} selected test MAE")
            split_evidence.append(evidence)
            split_count += 1
            permutation_count += 10
        canonical_scores = np.asarray([item.canonical_mae for item in split_evidence])
        method_summary = result["method_summary"]["full_mix"]
        if not arrays_match_one_ulp(canonical_scores, method_summary.get("split_test_mae", [])):
            fail(f"{endpoint}: method split scores mismatch")
        close(float(canonical_scores.mean()), method_summary.get("mean_test_mae"),
              f"{endpoint} method mean")
        close(float(canonical_scores.std()), method_summary.get("std_test_mae"),
              f"{endpoint} method SD")
        if not arrays_match_one_ulp(canonical_scores, result.get("selected_split_test_mae", [])):
            fail(f"{endpoint}: selected split scores mismatch")
        close(float(canonical_scores.mean()), result.get("selected_mean_test_mae"),
              f"{endpoint} selected mean")
        close(float(canonical_scores.std()), result.get("selected_std_test_mae"),
              f"{endpoint} selected SD")
        canonical = statistics.fmean(item.canonical_mae for item in split_evidence)
        permuted = statistics.fmean(item.mean_permuted_mae for item in split_evidence)
        endpoint_rows.append({
            "endpoint": endpoint,
            "family": result["family"],
            "canonical_mae": canonical,
            "mean_permuted_mae": permuted,
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
            "ordered_test_rows_sha256": stable_hash([
                item.ordered_test_rows_sha256 for item in split_evidence
            ]),
            "ordered_permutation_sets_sha256": stable_hash([
                digest for item in split_evidence for digest in item.permutation_ordered_sha256
            ]),
        })
        shard = endpoint_index % len(EXPECTED_SHARDS)
        result_mtime = datetime.fromtimestamp(result_path.stat().st_mtime, tz=timezone.utc)
        # Markers have one-second resolution, whereas filesystem mtimes may retain
        # subsecond precision even when both writes occurred in the same second.
        if result_mtime + timedelta(seconds=1) < launch_mtimes[shard]:
            fail(f"{endpoint}: result predates its shard launch manifest")
        if result_mtime > shard_times[shard] + timedelta(seconds=1):
            fail(f"{endpoint}: result modification time follows its shard completion marker")

    if checkpoint_hashes != {checkpoint_sha} or runner_hashes != {runner_sha}:
        fail("result files do not share the unique launch checkpoint/runner")
    if len(software_contracts) != 1:
        fail("result files report mixed software-version contracts")
    if split_count != 69 or permutation_count != 690:
        fail("validated grid is not exactly 23 endpoints x 3 splits x 10 permutations")

    changes = np.asarray([row["mae_change"] for row in endpoint_rows], dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(changes, bootstrap_replicates, bootstrap_seed)
    endpoint_prediction_sd = np.asarray(
        [row["mean_prediction_sd"] for row in endpoint_rows], dtype=float
    )
    q1, q3 = np.quantile(endpoint_prediction_sd, [0.25, 0.75], method="linear")
    analysis = {
        "analysis_unit": "23 endpoint means; each endpoint first averages three locked splits",
        "macro_canonical_mae": statistics.fmean(row["canonical_mae"] for row in endpoint_rows),
        "macro_mean_permuted_mae": statistics.fmean(
            row["mean_permuted_mae"] for row in endpoint_rows
        ),
        "mean_endpoint_mae_change": float(changes.mean()),
        "median_endpoint_mae_change": float(np.median(changes)),
        "bootstrap_mean_change_ci_95": [ci_low, ci_high],
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed,
                      "method": "paired endpoint percentile bootstrap"},
        "endpoints_degraded": int(np.sum(changes > 0.0)),
        "endpoints_improved": int(np.sum(changes < 0.0)),
        "endpoints_tied": int(np.sum(changes == 0.0)),
        "exact_wilcoxon": exact_wilcoxon_two_sided(changes),
        "exact_sign_test": exact_sign_test_two_sided(changes),
        "multiplicity": "single prespecified atom-order contrast; no multiplicity adjustment",
        "median_endpoint_mean_prediction_sd": float(np.median(endpoint_prediction_sd)),
        "endpoint_mean_prediction_sd_iqr": [float(q1), float(q3)],
        "worst_maximum_prediction_deviation": max(
            row["maximum_prediction_delta"] for row in endpoint_rows
        ),
    }
    validation = {
        "endpoints": len(endpoint_rows), "splits": split_count,
        "prediction_cells": split_count, "permutations": permutation_count,
        "result_jsons": len(result_file_hashes), "cell_jsons": len(cell_file_hashes),
        "launch_manifests": 8, "completion_markers": 8,
        "checkpoint_sha256": checkpoint_sha, "runner_sha256": runner_sha,
        "wrapper_sha256": wrapper_sha, "source_bundle_sha256": source_bundle_sha,
        "rdkit_version_required": EXPECTED_RDKIT_VERSION,
        "rdkit_version_provenance": (
            "launch wrapper pin rdkit==2025.3.6; runtime reports 2025.03.6"
        ),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "result_inventory_sha256": stable_hash(result_file_hashes),
        "cell_inventory_sha256": stable_hash(cell_file_hashes),
        "completion_time_utc": complete_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    internal_summary = {
        "schema_version": 1, "scope": SCOPE, "analysis": analysis,
        "validation": validation, "endpoint_records": endpoint_rows,
        "inputs": {"results_dir": str(results_dir.resolve()),
                   "prepared_dir": str(prepared_dir.resolve()),
                   "reference_dir": str(reference_dir.resolve()),
                   "runner_source": str(runner_source.resolve()),
                   "wrapper_source": str(wrapper_source.resolve())},
        "input_file_sha256": {"results": result_file_hashes, "cells": cell_file_hashes},
    }
    anonymous_summary = {
        "schema_version": 1, "scope": SCOPE, "analysis": analysis,
        "validation": validation, "endpoint_records": endpoint_rows,
    }
    csv_fields = list(endpoint_rows[0])
    md_lines = [
        "# Atom-order robustness certification", "", f"Scope: {SCOPE}.", "",
        "Every held-out molecule was evaluated under ten deterministic RDKit atom "
        "renumberings after canonical isomeric identity was checked row by row.", "",
        f"Across 23 endpoint means, canonical MAE was {analysis['macro_canonical_mae']:.6f} "
        f"and mean permuted MAE was {analysis['macro_mean_permuted_mae']:.6f}; the mean "
        f"change was {analysis['mean_endpoint_mae_change']:+.6f} "
        f"(95% endpoint-bootstrap CI {ci_low:+.6f} to {ci_high:+.6f}).",
        f"Endpoints: {analysis['endpoints_degraded']} degraded, "
        f"{analysis['endpoints_improved']} improved, and {analysis['endpoints_tied']} tied. "
        f"Exact Wilcoxon p={analysis['exact_wilcoxon']['p_two_sided_exact']:.8g}; "
        f"exact sign-test p={analysis['exact_sign_test']['p_two_sided_exact']:.8g}. "
        "This is one prespecified contrast, so no multiplicity adjustment is applied.", "",
        "| Endpoint | Canonical MAE | Mean permuted MAE | Change | Mean prediction SD | Max prediction deviation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        md_lines.append(
            f"| {row['endpoint']} | {row['canonical_mae']:.6f} | "
            f"{row['mean_permuted_mae']:.6f} | {row['mae_change']:+.6f} | "
            f"{row['mean_prediction_sd']:.6f} | {row['maximum_prediction_delta']:.6f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    anonymous_json_text = json.dumps(anonymous_summary, indent=2, sort_keys=True) + "\n"
    csv_buffer_rows = endpoint_rows
    validate_anonymous_text(anonymous_json_text)
    validate_anonymous_text(md_text)
    validate_anonymous_text(json.dumps(csv_buffer_rows, sort_keys=True))

    # No output directory is touched before every validation and privacy gate passes.
    atomic_json(output_dir / "summary.json", internal_summary)
    atomic_text(output_dir / "summary_anonymous.json", anonymous_json_text)
    atomic_csv(output_dir / "endpoint_atom_order_robustness.csv", endpoint_rows, csv_fields)
    atomic_text(output_dir / "RESULTS.md", md_text)
    return anonymous_summary


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runner-source", type=Path,
                        default=root / "finetune_moljepa_benchmarks.py")
    parser.add_argument("--wrapper-source", type=Path,
                        default=root / "run_final_phase4_atom_order_audit.sh")
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_917)
    args = parser.parse_args()
    summary = certify(
        args.results_dir, args.prepared_dir, args.reference_dir, args.output_dir,
        runner_source=args.runner_source, wrapper_source=args.wrapper_source,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
