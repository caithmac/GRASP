#!/usr/bin/env python3
"""Fail-closed certification of the paired MLM-versus-RTD OpenADMET rerun.

The certifier treats the existing 23-endpoint RTD result directory as the
canonical downstream-protocol reference.  It validates every raw prediction,
every embedded and standalone cell, every aggregate, split identity, source
hash, shard record, and checkpoint-provenance record before writing anything.

The inferential unit is the endpoint.  Split MAEs are averaged within each
endpoint, and exact paired tests and endpoint bootstrap intervals are then
computed for the frozen objective effect, the full-finetuning objective effect,
and their full-minus-frozen interaction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


MODELS = ("mlm_s2", "rtd25_s2")
METHODS = ("frozen_mix", "full_mix")
SPLITS = ("split1", "split2", "split3")
ENDPOINTS = (
    "expansion_caco2_pappa",
    "expansion_caco2_efflux",
    "expansion_logd",
    "expansion_ksol",
    "expansion_hlm",
    "expansion_mlm",
    "expansion_mbpb",
    "expansion_mgmb",
    "expansion_mppb",
    "asap_mers",
    "asap_sars",
    "asap_logd",
    "asap_ksol",
    "asap_hlm",
    "asap_mlm",
    "asap_mdr1",
    "pxr",
    "biogen_solubility",
    "biogen_hlm",
    "biogen_rlm",
    "biogen_hppb",
    "biogen_rppb",
    "biogen_mdr1",
)
CONTRASTS = (
    "frozen_objective_delta",
    "full_objective_delta",
    "full_minus_frozen_interaction",
)
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_917
SHARDS = 12
HASH_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_RUNNER_SHA256 = "f06ee510bce06284e4a71636da4401fb7310d793c5a964a61942423966d2f0bb"
EXPECTED_WRAPPER_SHA256 = "ad8b8da24ba664a7ac026e31bcb44e8f443041411211e8f37898dee1a79693a4"
EXPECTED_JOB_YAML_SHA256 = "3b9cd4e8dd6a25bd5dd3d7ebe3fe0488a501bdf0215f2744b78270fe58893737"
EXPECTED_CHECKPOINT_SHA256 = {
    "mlm_s2": "1160ecaea098f0500d5b6138203f7bd7c5852ef8b34d3d5cc908aeed57c04b86",
    "rtd25_s2": "daa0d70f4a7ee44807192ccfe8735696cf26e3b6d10dc0a74bc877e586b34dba",
}
EXPECTED_JOB_UID = "3840bd50-89ca-44d6-bb0c-41bcde563490"
EXPECTED_JOB_CREATION = "2026-09-17T12:45:09Z"
EXPECTED_JOB_START = "2026-09-17T13:04:15Z"
EXPECTED_IMAGE_ID = "sha256:72d016011185c8e8c82442c87135def044f0f9707f9fd4ec1703a9e403ad4c35"
EXPECTED_ARCHITECTURE = {
    "num_hidden_layers": 12,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "mix_layers": [4, 8, 10, 12],
}
EXPECTED_CONFIGS = {
    method: {
        "method": method,
        "learning_rate": 1e-3 if method == "frozen_mix" else 1e-5,
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
        "atom_order_permutations": 0,
    }
    for method in METHODS
}
ENCODER_CONFIG_FIELDS = (
    "embedding_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "attention_head_size", "attention_probs_dropout_prob",
    "hidden_dropout_prob", "hidden_act", "initializer_range", "layer_norm_eps",
    "max_position_embeddings", "max_relative_positions", "position_buckets",
    "norm_rel_ebd", "pos_att_type", "position_biased_input", "relative_attention",
    "share_att_key", "type_vocab_size", "vocab_size",
)
PRIVATE_PATTERNS = {
    "Windows absolute path": re.compile(r"[A-Za-z]:[\\/]"),
    "private storage path": re.compile(r"/(?:mnt|nfs|home|Users)/", re.IGNORECASE),
    "cluster account or namespace": re.compile(
        r"\bs\d{6,}-(?:eidf\d+|[a-z]+ns\d+)\b", re.IGNORECASE
    ),
    "private IPv4 address": re.compile(
        r"(?<!\d)(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)"
        r"\d{1,3}\.\d{1,3}(?!\d)"
    ),
    "UNC path": re.compile(r"\\\\[^\\\s]+\\[^\\\s]+"),
    "private scratch path": re.compile(r"/(?:scratch|project|workspace|gpfs|lustre)/", re.IGNORECASE),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "cluster account": re.compile(r"\bs\d{6,}\b", re.IGNORECASE),
}


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        fail(f"{label} is not a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not a finite number")
    if not math.isfinite(result):
        fail(f"{label} is not a finite number")
    return result


def finite_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        fail(f"{label} must be a nonempty list")
    return [finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]


def close(left: float, right: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=atol):
        fail(f"{label} mismatch: {left!r} versus {right!r}")


def parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        fail(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label} is not a UTC timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        fail(f"{label} is not UTC")
    return parsed.astimezone(timezone.utc)


def require_iso_marker(path: Path) -> str:
    if not path.is_file():
        fail(f"Required completion marker is missing: {path.name}")
    value = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        fail(f"Completion marker is not a UTC timestamp: {path.name}")
    parse_utc(value, path.name)
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"Cannot read valid JSON from {path}: {error}")
    if not isinstance(payload, dict):
        fail(f"Expected a JSON object: {path}")
    return payload


def strict_json(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        fail(f"Cannot serialize strict JSON: {error}")


def normalized_config(value: Any, *, legacy_reference: bool, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label}: run config is not an object")
    result = dict(value)
    if legacy_reference and "atom_order_permutations" not in result:
        result["atom_order_permutations"] = 0
    return result


def expected_seed(split: str, method: str) -> int:
    return 9100 + (SPLITS.index(split) + 1) * 100 + (0 if method == "frozen_mix" else 2)


def expected_run_files() -> set[str]:
    files = {"COMPLETE", "objective_checkpoint_audit.json", "checkpoint_audit.lock", "summary.lock"}
    files.update(f"SHARD_{index}_COMPLETE" for index in range(SHARDS))
    files.update(f"launch_manifest_shard_{index:02d}.json" for index in range(SHARDS))
    files.update({"comparison/RESULTS.md", "comparison/summary.json", "comparison/mlm_vs_rtd_moljepa23.csv"})
    for model in MODELS:
        for endpoint in ENDPOINTS:
            files.add(f"{model}/results_{endpoint}.json")
            files.add(f"{model}/log_{endpoint}.txt")
            files.update(
                f"{model}/{endpoint}/cells/{split}_{method}.json"
                for split in SPLITS for method in METHODS
            )
    return files


def expected_run_directories() -> set[str]:
    directories = {"comparison", *MODELS}
    for model in MODELS:
        for endpoint in ENDPOINTS:
            directories.add(f"{model}/{endpoint}")
            directories.add(f"{model}/{endpoint}/cells")
    return directories


def relevant_reference_paths() -> list[str]:
    paths: list[str] = []
    for endpoint in ENDPOINTS:
        paths.append(f"results_{endpoint}.json")
        paths.extend(
            f"{endpoint}/cells/{split}_{method}.json"
            for split in SPLITS for method in METHODS
        )
    return sorted(paths)


def reference_manifest_digest(reference_dir: Path) -> str:
    records: list[dict[str, Any]] = []
    for relative in relevant_reference_paths():
        path = reference_dir / Path(relative)
        if path.is_symlink() or not path.is_file():
            fail(f"Canonical reference relevant file is missing or unsafe: {relative}")
        stat = path.stat()
        records.append({"path": relative, "sha256": sha256_file(path), "size_bytes": stat.st_size})
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_text(canonical)


def validate_attestation(
    *,
    results_dir: Path,
    attestation_path: Path,
    expected_source_bundle_sha256: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    for value, label in (
        (expected_source_bundle_sha256, "expected source-bundle hash"),
        (expected_archive_sha256, "expected archive hash"),
    ):
        require_hash(value, label)
    if results_dir.is_symlink():
        fail("Results directory must not be a symlink")
    attestation = load_json(attestation_path)
    if set(attestation) != {
        "schema_version", "job", "source_bundle_sha256", "archive_sha256",
        "generated_at", "files",
    } or attestation.get("schema_version") != 1:
        fail("Run attestation top-level schema mismatch")
    job = attestation.get("job")
    expected_job = {
        "uid": EXPECTED_JOB_UID,
        "creation_timestamp": EXPECTED_JOB_CREATION,
        "start_timestamp": EXPECTED_JOB_START,
        "image_id": EXPECTED_IMAGE_ID,
    }
    if job != expected_job:
        fail("Run attestation does not identify the pinned Kubernetes job/runtime")
    if attestation.get("source_bundle_sha256") != expected_source_bundle_sha256:
        fail("Run attestation source-bundle hash differs from the independently supplied hash")
    if attestation.get("archive_sha256") != expected_archive_sha256:
        fail("Run attestation archive hash differs from the post-completion supplied hash")
    generated_at = parse_utc(attestation.get("generated_at"), "attestation/generated_at")
    if generated_at < parse_utc(EXPECTED_JOB_START, "expected job start"):
        fail("Run attestation predates the pinned job start")
    if generated_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        fail("Run attestation timestamp is materially in the future")

    records = attestation.get("files")
    if not isinstance(records, dict):
        fail("Run attestation files must be an object keyed by relative POSIX path")
    expected_files = expected_run_files()
    if set(records) != expected_files:
        missing = sorted(expected_files.difference(records))
        extra = sorted(set(records).difference(expected_files))
        fail(f"Attested file inventory mismatch: missing={missing}, extra={extra}")

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in results_dir.rglob("*"):
        relative = path.relative_to(results_dir).as_posix()
        if path.is_symlink():
            fail(f"Results inventory contains a symlink: {relative}")
        if path.is_dir():
            observed_directories.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            fail(f"Results inventory contains a non-regular entry: {relative}")
    if observed_files != expected_files:
        missing = sorted(expected_files.difference(observed_files))
        extra = sorted(observed_files.difference(expected_files))
        fail(f"Extracted file inventory mismatch: missing={missing}, extra={extra}")
    if observed_directories != expected_run_directories():
        missing = sorted(expected_run_directories().difference(observed_directories))
        extra = sorted(observed_directories.difference(expected_run_directories()))
        fail(f"Extracted directory inventory mismatch: missing={missing}, extra={extra}")

    normalized: dict[str, dict[str, Any]] = {}
    casefolded: set[str] = set()
    for relative in sorted(expected_files):
        if relative.startswith("/") or ".." in Path(relative).parts or "\\" in relative:
            fail(f"Unsafe attested relative path: {relative}")
        folded = relative.casefold()
        if folded in casefolded:
            fail(f"Case-colliding attested relative path: {relative}")
        casefolded.add(folded)
        record = records[relative]
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes", "mtime_ns"}:
            fail(f"Malformed attestation record: {relative}")
        digest = require_hash(record.get("sha256"), f"attestation/{relative}/sha256")
        size = record.get("size_bytes")
        mtime = record.get("mtime_ns")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            fail(f"Invalid attested size: {relative}")
        if isinstance(mtime, bool) or not isinstance(mtime, int) or mtime <= 0:
            fail(f"Invalid attested mtime: {relative}")
        path = results_dir / Path(relative)
        stat = path.stat()
        if stat.st_nlink != 1:
            fail(f"Results inventory contains a hard-linked file: {relative}")
        if stat.st_size != size or stat.st_mtime_ns != mtime or sha256_file(path) != digest:
            fail(f"Extracted file differs from remote SHA/size/mtime attestation: {relative}")
        normalized[relative] = {"sha256": digest, "size_bytes": size, "mtime_ns": mtime}
    return {
        "sha256": sha256_file(attestation_path),
        "generated_at": generated_at,
        "source_bundle_sha256": expected_source_bundle_sha256,
        "archive_sha256": expected_archive_sha256,
        "files": normalized,
    }


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def exact_wilcoxon_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    raw = [finite_float(value, "Wilcoxon difference") for value in differences]
    nonzero = [value for value in raw if value != 0.0]
    if not nonzero:
        return {
            "n_nonzero": 0,
            "zero_pairs": len(raw),
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_two_sided_exact": 1.0,
            "definition": "exact sign-randomization distribution; doubled smaller tail",
        }
    rank_units = [round(rank * 2) for rank in average_ranks([abs(x) for x in nonzero])]
    counts = {0: 1}
    for units in rank_units:
        updated = counts.copy()
        for subtotal, count in counts.items():
            updated[subtotal + units] = updated.get(subtotal + units, 0) + count
        counts = updated
    observed_plus = sum(units for units, value in zip(rank_units, nonzero) if value > 0)
    total = sum(rank_units)
    boundary = min(observed_plus, total - observed_plus)
    lower_count = sum(count for score, count in counts.items() if score <= boundary)
    p_value = min(1.0, 2.0 * lower_count / (2 ** len(nonzero)))
    return {
        "n_nonzero": len(nonzero),
        "zero_pairs": len(raw) - len(nonzero),
        "w_plus": observed_plus / 2.0,
        "w_minus": (total - observed_plus) / 2.0,
        "statistic": boundary / 2.0,
        "p_two_sided_exact": p_value,
        "definition": "exact sign-randomization distribution; doubled smaller tail",
    }


def exact_sign_test_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    raw = [finite_float(value, "sign-test difference") for value in differences]
    positives = sum(value > 0.0 for value in raw)
    negatives = sum(value < 0.0 for value in raw)
    nonzero = positives + negatives
    if nonzero:
        smaller = min(positives, negatives)
        tail = sum(math.comb(nonzero, index) for index in range(smaller + 1)) / (2**nonzero)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "n_nonzero": nonzero,
        "positive_differences": positives,
        "negative_differences": negatives,
        "zero_pairs": len(raw) - nonzero,
        "p_two_sided_exact": p_value,
        "definition": "exact Binomial(n, 0.5); doubled smaller tail; zero pairs omitted",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values or any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        fail("Holm adjustment requires finite p-values in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    result = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        result[index] = running
    return result


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float], *, replicates: int, seed: int
) -> tuple[float, float]:
    if not values or replicates < 1:
        fail("Bootstrap requires nonempty values and a positive replicate count")
    values = [finite_float(value, "bootstrap value") for value in values]
    rng = random.Random(seed)
    count = len(values)
    estimates = [
        math.fsum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    ]
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def validate_raw_cell(
    cell: dict[str, Any], label: str, *, legacy_reference: bool = False
) -> dict[str, Any]:
    if cell.get("method") not in METHODS:
        fail(f"{label}: unexpected method")
    if cell.get("split") not in SPLITS or cell.get("task") not in ENDPOINTS:
        fail(f"{label}: task/split identity is invalid")
    for name in ("checkpoint_sha256", "data_sha256", "runner_sha256"):
        require_hash(cell.get(name), f"{label}/{name}")
    for name in ("train_smiles_sha256", "validation_smiles_sha256", "test_smiles_sha256"):
        require_hash(cell.get(name), f"{label}/{name}")
    y_true = finite_vector(cell.get("y_true"), f"{label}/y_true")
    y_pred = finite_vector(cell.get("y_pred"), f"{label}/y_pred")
    if len(y_true) != len(y_pred):
        fail(f"{label}: y_true/y_pred lengths differ")
    for name in ("train_rows", "validation_rows", "test_rows", "seed"):
        if isinstance(cell.get(name), bool) or not isinstance(cell.get(name), int):
            fail(f"{label}: {name} is not an integer")
    if cell["train_rows"] <= 0 or cell["validation_rows"] <= 0:
        fail(f"{label}: train/validation rows must be positive")
    if cell["test_rows"] != len(y_true):
        fail(f"{label}: test row count differs from raw targets")
    expected = EXPECTED_CONFIGS[cell["method"]]
    config = normalized_config(
        cell.get("run_config"), legacy_reference=legacy_reference, label=label
    )
    if config != expected:
        fail(f"{label}: run_config differs from the pinned downstream configuration")
    if cell["seed"] != expected_seed(cell["split"], cell["method"]):
        fail(f"{label}: seed differs from the pinned split/method seed")
    if cell.get("architecture") != EXPECTED_ARCHITECTURE:
        fail(f"{label}: architecture differs from the pinned base encoder/readout contract")
    close(
        finite_float(cell.get("learning_rate"), f"{label}/learning_rate"),
        expected["learning_rate"],
        f"{label}/learning_rate",
    )
    recomputed = math.fsum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
    recorded = finite_float(cell.get("test_mae"), f"{label}/test_mae")
    close(recomputed, recorded, f"{label}/raw prediction MAE")
    finite_float(cell.get("validation_mae"), f"{label}/validation_mae")
    return {"y_true": y_true, "test_mae": recomputed, "run_config": config}


def expected_result_names() -> set[str]:
    return {f"results_{endpoint}.json" for endpoint in ENDPOINTS}


def load_reference(reference_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Canonical reference directory is missing: {reference_dir}")
    paths = list(reference_dir.glob("results_*.json"))
    if {path.name for path in paths} != expected_result_names():
        fail("Canonical reference does not contain exactly the 23 expected result filenames")
    reference: dict[tuple[str, str, str], dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        path = reference_dir / f"results_{endpoint}.json"
        payload = load_json(path)
        if (
            payload.get("schema_version") != 1
            or payload.get("task") != endpoint
            or payload.get("primary_metric") != "mae"
        ):
            fail(f"Canonical reference identity/metric mismatch: {path.name}")
        if payload.get("data_protocol") != "reconstructed_butina_public_snapshot":
            fail(f"Canonical reference protocol mismatch: {endpoint}")
        if tuple(payload.get("splits", {})) != SPLITS:
            fail(f"Canonical reference split set/order mismatch: {endpoint}")
        configs = payload.get("run_configs")
        if not isinstance(configs, dict) or not set(METHODS).issubset(configs):
            fail(f"Canonical reference lacks required method configs: {endpoint}")
        normalized_configs = {
            method: normalized_config(
                configs[method], legacy_reference=True, label=f"reference/{endpoint}/{method}"
            )
            for method in METHODS
        }
        if normalized_configs != EXPECTED_CONFIGS:
            fail(f"Canonical reference configs differ from the pinned downstream contract: {endpoint}")
        for split in SPLITS:
            methods = payload["splits"][split].get("methods", {})
            if not set(METHODS).issubset(methods):
                fail(f"Canonical reference lacks required methods: {endpoint}/{split}")
            for method in METHODS:
                cell = methods[method]
                label = f"reference/{endpoint}/{split}/{method}"
                if cell.get("task") != endpoint or cell.get("split") != split or cell.get("method") != method:
                    fail(f"{label}: embedded identity mismatch")
                validated = validate_raw_cell(cell, label, legacy_reference=True)
                if normalized_config(
                    cell.get("run_config"), legacy_reference=True, label=label
                ) != normalized_configs[method]:
                    fail(f"{label}: cell/top-level configuration mismatch")
                standalone_path = (
                    reference_dir / endpoint / "cells" / f"{split}_{method}.json"
                )
                if load_json(standalone_path) != cell:
                    fail(f"{label}: standalone and embedded canonical cells differ")
                reference[(endpoint, split, method)] = {
                    "family": payload.get("family"),
                    "display_name": payload.get("display_name"),
                    "source_count_matches_paper": payload.get("source_count_matches_paper"),
                    "exact_table3_split_reproduction": payload.get("exact_table3_split_reproduction"),
                    "run_config": normalized_configs[method],
                    "data_sha256": cell["data_sha256"],
                    "seed": cell["seed"],
                    "train_rows": cell["train_rows"],
                    "validation_rows": cell["validation_rows"],
                    "test_rows": cell["test_rows"],
                    "train_smiles_sha256": cell["train_smiles_sha256"],
                    "validation_smiles_sha256": cell["validation_smiles_sha256"],
                    "test_smiles_sha256": cell["test_smiles_sha256"],
                    "y_true": validated["y_true"],
                }
            signatures = []
            for method in METHODS:
                item = reference[(endpoint, split, method)]
                signatures.append(
                    tuple(item[name] for name in (
                        "data_sha256", "train_rows", "validation_rows", "test_rows",
                        "train_smiles_sha256", "validation_smiles_sha256", "test_smiles_sha256",
                    )) + (tuple(item["y_true"]),)
                )
            if len(set(signatures)) != 1:
                fail(f"Canonical reference split/targets differ across methods: {endpoint}/{split}")
    if len(reference) != len(ENDPOINTS) * len(SPLITS) * len(METHODS):
        fail("Canonical reference grid is incomplete")
    return reference


def validate_top_summaries(payload: dict[str, Any], endpoint: str, model: str) -> None:
    summaries = payload.get("method_summary")
    if not isinstance(summaries, dict) or tuple(summaries) != METHODS:
        fail(f"{model}/{endpoint}: method_summary must contain exactly {METHODS} in order")
    split_test: dict[str, list[float]] = {}
    for method in METHODS:
        values = [
            finite_float(
                payload["splits"][split]["methods"][method].get("test_mae"),
                f"{model}/{endpoint}/{split}/{method}/test_mae",
            )
            for split in SPLITS
        ]
        split_test[method] = values
        summary = summaries[method]
        if not isinstance(summary, dict):
            fail(f"{model}/{endpoint}/{method}: malformed method summary")
        recorded_values = finite_vector(
            summary.get("split_test_mae"), f"{model}/{endpoint}/{method}/split_test_mae"
        )
        if len(recorded_values) != len(SPLITS):
            fail(f"{model}/{endpoint}/{method}: summary does not contain three split MAEs")
        for index, (recorded, raw) in enumerate(zip(recorded_values, values), start=1):
            close(recorded, raw, f"{model}/{endpoint}/{method}/split{index} summary")
        close(
            finite_float(summary.get("mean_test_mae"), "mean_test_mae"),
            statistics.fmean(values),
            f"{model}/{endpoint}/{method}/mean summary",
        )
        close(
            finite_float(summary.get("std_test_mae"), "std_test_mae"),
            statistics.pstdev(values),
            f"{model}/{endpoint}/{method}/std summary",
        )

    selected_values: list[float] = []
    for split in SPLITS:
        split_payload = payload["splits"][split]
        validations = [
            finite_float(split_payload["methods"][method].get("validation_mae"), "validation_mae")
            for method in METHODS
        ]
        selected_index = min(range(len(METHODS)), key=validations.__getitem__)
        selected_method = METHODS[selected_index]
        selected_cell = split_payload["methods"][selected_method]
        if split_payload.get("selected_method") != selected_method:
            fail(f"{model}/{endpoint}/{split}: selected method is not validation-optimal")
        close(
            finite_float(split_payload.get("selected_validation_mae"), "selected_validation_mae"),
            validations[selected_index],
            f"{model}/{endpoint}/{split}/selected validation",
        )
        selected_test = finite_float(selected_cell.get("test_mae"), "selected test_mae")
        close(
            finite_float(split_payload.get("selected_test_mae"), "selected_test_mae"),
            selected_test,
            f"{model}/{endpoint}/{split}/selected test",
        )
        selected_values.append(selected_test)
    recorded_selected = finite_vector(
        payload.get("selected_split_test_mae"), f"{model}/{endpoint}/selected_split_test_mae"
    )
    if len(recorded_selected) != len(SPLITS):
        fail(f"{model}/{endpoint}: selected split summary length mismatch")
    for recorded, raw in zip(recorded_selected, selected_values):
        close(recorded, raw, f"{model}/{endpoint}/selected split summary")
    close(
        finite_float(payload.get("selected_mean_test_mae"), "selected_mean_test_mae"),
        statistics.fmean(selected_values),
        f"{model}/{endpoint}/selected mean summary",
    )
    close(
        finite_float(payload.get("selected_std_test_mae"), "selected_std_test_mae"),
        statistics.pstdev(selected_values),
        f"{model}/{endpoint}/selected std summary",
    )


def validate_results(
    results_dir: Path, reference: dict[tuple[str, str, str], dict[str, Any]]
) -> dict[str, Any]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Retrieved objective result directory is missing: {results_dir}")
    complete_value = require_iso_marker(results_dir / "COMPLETE")
    observed_model_dirs = {
        path.name
        for path in results_dir.iterdir()
        if path.is_dir() and any(path.glob("results_*.json"))
    }
    if observed_model_dirs != set(MODELS):
        fail(f"Result-bearing model directories must be exactly {MODELS}")
    marker_names = {path.name for path in results_dir.glob("SHARD_*_COMPLETE")}
    expected_markers = {f"SHARD_{index}_COMPLETE" for index in range(SHARDS)}
    if marker_names != expected_markers:
        fail("Expected exactly 12 shard completion markers with IDs 0--11")
    marker_values = {
        name: require_iso_marker(results_dir / name) for name in sorted(expected_markers)
    }

    cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    result_hashes: dict[str, str] = {}
    cell_hashes: dict[str, str] = {}
    checkpoint_paths = {model: set() for model in MODELS}
    checkpoint_hashes = {model: set() for model in MODELS}
    runner_hashes: set[str] = set()

    for model in MODELS:
        model_dir = results_dir / model
        if not model_dir.is_dir():
            fail(f"Required model directory is missing: {model}")
        result_paths = list(model_dir.glob("results_*.json"))
        if {path.name for path in result_paths} != expected_result_names():
            fail(f"{model}: result filenames do not exactly match the expected 23 endpoints")
        for endpoint in ENDPOINTS:
            result_path = model_dir / f"results_{endpoint}.json"
            payload = load_json(result_path)
            result_hashes[f"{model}/{result_path.name}"] = sha256_file(result_path)
            if (
                payload.get("schema_version") != 1
                or payload.get("task") != endpoint
                or payload.get("primary_metric") != "mae"
            ):
                fail(f"{model}/{endpoint}: top-level identity/metric mismatch")
            if payload.get("data_protocol") != "reconstructed_butina_public_snapshot":
                fail(f"{model}/{endpoint}: data protocol mismatch")
            if tuple(payload.get("run_configs", {})) != METHODS:
                fail(f"{model}/{endpoint}: run_configs must contain exactly {METHODS} in order")
            if payload["run_configs"] != EXPECTED_CONFIGS:
                fail(f"{model}/{endpoint}: run_configs differ from the pinned launch contract")
            if tuple(payload.get("splits", {})) != SPLITS:
                fail(f"{model}/{endpoint}: split set/order mismatch")
            checkpoint_path = payload.get("checkpoint")
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                fail(f"{model}/{endpoint}: checkpoint path is absent")
            checkpoint_paths[model].add(checkpoint_path)
            checkpoint_hashes[model].add(
                require_hash(payload.get("checkpoint_sha256"), f"{model}/{endpoint}/checkpoint")
            )
            runner_hashes.add(require_hash(payload.get("runner_sha256"), f"{model}/{endpoint}/runner"))

            expected_family = reference[(endpoint, SPLITS[0], METHODS[0])]
            for name in (
                "family",
                "display_name",
                "source_count_matches_paper",
                "exact_table3_split_reproduction",
            ):
                if payload.get(name) != expected_family[name]:
                    fail(f"{model}/{endpoint}: {name} differs from canonical reference")
            for method in METHODS:
                if payload["run_configs"][method] != reference[(endpoint, SPLITS[0], method)]["run_config"]:
                    fail(f"{model}/{endpoint}/{method}: top-level config differs from reference")

            endpoint_cells_dir = model_dir / endpoint / "cells"
            expected_cell_names = {
                f"{split}_{method}.json" for split in SPLITS for method in METHODS
            }
            actual_cell_names = (
                {path.name for path in endpoint_cells_dir.glob("*.json")}
                if endpoint_cells_dir.is_dir()
                else set()
            )
            if actual_cell_names != expected_cell_names:
                fail(f"{model}/{endpoint}: standalone cell filenames are incomplete or unexpected")

            for split in SPLITS:
                split_payload = payload["splits"][split]
                methods = split_payload.get("methods")
                if not isinstance(methods, dict) or tuple(methods) != METHODS:
                    fail(f"{model}/{endpoint}/{split}: method set/order mismatch")
                for method in METHODS:
                    cell = methods[method]
                    label = f"{model}/{endpoint}/{split}/{method}"
                    if cell.get("task") != endpoint or cell.get("split") != split or cell.get("method") != method:
                        fail(f"{label}: embedded cell identity mismatch")
                    validated = validate_raw_cell(cell, label)
                    standalone_path = endpoint_cells_dir / f"{split}_{method}.json"
                    standalone = load_json(standalone_path)
                    if standalone != cell:
                        fail(f"{label}: standalone and embedded cell payloads differ")
                    cell_hashes[f"{model}/{endpoint}/cells/{standalone_path.name}"] = sha256_file(
                        standalone_path
                    )
                    if cell.get("checkpoint") != checkpoint_path:
                        fail(f"{label}: checkpoint path differs from top-level result")
                    if cell.get("checkpoint_sha256") != payload["checkpoint_sha256"]:
                        fail(f"{label}: checkpoint hash differs from top-level result")
                    if cell.get("runner_sha256") != payload["runner_sha256"]:
                        fail(f"{label}: runner hash differs from top-level result")
                    if cell.get("run_config") != payload["run_configs"][method]:
                        fail(f"{label}: cell/top-level configuration mismatch")

                    canonical = reference[(endpoint, split, method)]
                    for name in (
                        "data_sha256",
                        "seed",
                        "run_config",
                        "train_rows",
                        "validation_rows",
                        "test_rows",
                        "train_smiles_sha256",
                        "validation_smiles_sha256",
                        "test_smiles_sha256",
                    ):
                        if cell.get(name) != canonical[name]:
                            fail(f"{label}: {name} differs from canonical reference")
                    if validated["y_true"] != canonical["y_true"]:
                        fail(f"{label}: raw test targets differ from canonical reference")
                    cells[(model, endpoint, split, method)] = {
                        "test_mae": validated["test_mae"],
                        "validation_mae": finite_float(cell["validation_mae"], "validation_mae"),
                    }
            validate_top_summaries(payload, endpoint, model)

    if any(len(checkpoint_paths[model]) != 1 for model in MODELS):
        fail("Each model must use exactly one checkpoint path")
    if any(len(checkpoint_hashes[model]) != 1 for model in MODELS):
        fail("Each model must use exactly one checkpoint hash")
    if checkpoint_paths[MODELS[0]] == checkpoint_paths[MODELS[1]]:
        fail("MLM and RTD use the same checkpoint path")
    if checkpoint_hashes[MODELS[0]] == checkpoint_hashes[MODELS[1]]:
        fail("MLM and RTD use the same checkpoint hash")
    if len(runner_hashes) != 1:
        fail("Result files do not share exactly one runner hash")
    if next(iter(runner_hashes)) != EXPECTED_RUNNER_SHA256:
        fail("Result files do not use the pinned objective runner")
    for model in MODELS:
        if next(iter(checkpoint_hashes[model])) != EXPECTED_CHECKPOINT_SHA256[model]:
            fail(f"{model} result files do not use the pinned checkpoint")
    if len(cells) != len(MODELS) * len(ENDPOINTS) * len(SPLITS) * len(METHODS):
        fail("Validated objective cell grid is incomplete")
    return {
        "cells": cells,
        "result_hashes": dict(sorted(result_hashes.items())),
        "cell_hashes": dict(sorted(cell_hashes.items())),
        "checkpoint_paths": {model: next(iter(checkpoint_paths[model])) for model in MODELS},
        "checkpoint_hashes": {model: next(iter(checkpoint_hashes[model])) for model in MODELS},
        "runner_sha256": next(iter(runner_hashes)),
        "marker_values": marker_values,
        "complete_value": complete_value,
    }


def validate_chronology(
    *, results_dir: Path, validated: dict[str, Any], attestation: dict[str, Any]
) -> None:
    records = attestation["files"]
    creation = parse_utc(EXPECTED_JOB_CREATION, "expected job creation")
    start = parse_utc(EXPECTED_JOB_START, "expected job start")
    if creation > start:
        fail("Pinned Kubernetes creation time is after start time")
    start_ns = int(start.timestamp() * 1_000_000_000)
    complete_time = parse_utc(validated["complete_value"], "COMPLETE")
    complete_ns = records["COMPLETE"]["mtime_ns"]
    if complete_ns < int(complete_time.timestamp() * 1_000_000_000):
        fail("COMPLETE file mtime predates its timestamp content")
    if attestation["generated_at"] < complete_time:
        fail("Run attestation was generated before COMPLETE")
    if complete_ns > int(attestation["generated_at"].timestamp() * 1_000_000_000):
        fail("Attested COMPLETE mtime is after attestation generation")

    for shard in range(SHARDS):
        manifest_relative = f"launch_manifest_shard_{shard:02d}.json"
        marker_relative = f"SHARD_{shard}_COMPLETE"
        manifest_ns = records[manifest_relative]["mtime_ns"]
        marker_ns = records[marker_relative]["mtime_ns"]
        marker_time = parse_utc(validated["marker_values"][marker_relative], marker_relative)
        marker_content_ns = int(marker_time.timestamp() * 1_000_000_000)
        if not start_ns <= manifest_ns <= marker_ns <= complete_ns:
            fail(f"Launch/marker/COMPLETE mtime ordering is invalid for shard {shard}")
        if marker_ns < marker_content_ns or marker_time > complete_time:
            fail(f"Marker content ordering is invalid for shard {shard}")

    for model_index, model in enumerate(MODELS):
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            shard = (model_index * len(ENDPOINTS) + endpoint_index) % SHARDS
            manifest_ns = records[f"launch_manifest_shard_{shard:02d}.json"]["mtime_ns"]
            marker_ns = records[f"SHARD_{shard}_COMPLETE"]["mtime_ns"]
            cell_relatives = [
                f"{model}/{endpoint}/cells/{split}_{method}.json"
                for split in SPLITS for method in METHODS
            ]
            result_relative = f"{model}/results_{endpoint}.json"
            log_relative = f"{model}/log_{endpoint}.txt"
            cell_mtimes = [records[relative]["mtime_ns"] for relative in cell_relatives]
            if any(not manifest_ns <= value <= marker_ns for value in cell_mtimes):
                fail(f"Pre-launch or post-marker cell detected: {model}/{endpoint}")
            result_ns = records[result_relative]["mtime_ns"]
            log_ns = records[log_relative]["mtime_ns"]
            if max(cell_mtimes) > result_ns:
                fail(f"Result predates one of its cells: {model}/{endpoint}")
            if not manifest_ns <= result_ns <= marker_ns:
                fail(f"Result falls outside its shard launch/marker interval: {model}/{endpoint}")
            if not manifest_ns <= log_ns <= marker_ns:
                fail(f"Log falls outside its shard launch/marker interval: {model}/{endpoint}")


def independently_verify_checkpoints(
    *, mlm_checkpoint_path: Path, rtd_checkpoint_path: Path, audit: dict[str, Any]
) -> None:
    for model, path in (("mlm_s2", mlm_checkpoint_path), ("rtd25_s2", rtd_checkpoint_path)):
        if path.is_symlink() or not path.is_file():
            fail(f"Independent {model} checkpoint is missing or unsafe")
        if sha256_file(path) != EXPECTED_CHECKPOINT_SHA256[model]:
            fail(f"Independent {model} checkpoint file hash mismatch")
    try:
        import torch
        from encoder_arch import load_encoder_state, resolve_encoder_config
    except Exception as error:
        fail(f"Checkpoint tensor recomputation dependencies are unavailable: {error}")

    def canonical(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
        state = load_encoder_state(str(path))
        if not state:
            fail(f"Checkpoint has no encoder state: {path.name}")
        config = resolve_encoder_config(state, "auto")
        digest = hashlib.sha256()
        normalized: dict[str, Any] = {}
        for key in sorted(state):
            value = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            normalized[key] = value
        return normalized, config, digest.hexdigest()

    mlm, mlm_config, mlm_digest = canonical(mlm_checkpoint_path)
    rtd, rtd_config, rtd_digest = canonical(rtd_checkpoint_path)
    if mlm.keys() != rtd.keys() or mlm_config != rtd_config:
        fail("Independent checkpoint tensor architectures differ")
    differing = 0
    maximum = 0.0
    sum_sq_delta = 0.0
    sum_sq_rtd = 0.0
    for key in mlm:
        if mlm[key].shape != rtd[key].shape:
            fail(f"Independent checkpoint tensor shape mismatch: {key}")
        delta = mlm[key].float() - rtd[key].float()
        current = float(delta.abs().max()) if delta.numel() else 0.0
        differing += int(current > 0.0)
        maximum = max(maximum, current)
        sum_sq_delta += float(torch.sum(delta.double().square()))
        sum_sq_rtd += float(torch.sum(rtd[key].double().square()))
    relative_l2 = (sum_sq_delta / max(sum_sq_rtd, 1e-300)) ** 0.5
    expected = {
        "mlm_canonical_tensor_sha256": mlm_digest,
        "rtd_canonical_tensor_sha256": rtd_digest,
        "parameter_tensors": len(mlm),
        "tensors_differing": differing,
        "resolved_encoder_config": mlm_config,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            fail(f"Independent checkpoint tensor audit mismatch: {key}")
    for key, value in (
        ("maximum_absolute_tensor_delta", maximum),
        ("global_relative_l2_delta", relative_l2),
    ):
        if not math.isclose(finite_float(audit.get(key), key), value, rel_tol=1e-12, abs_tol=1e-12):
            fail(f"Independent checkpoint tensor audit mismatch: {key}")


def validate_provenance(
    results_dir: Path,
    validated: dict[str, Any],
    *,
    expected_source_bundle_sha256: str,
    mlm_checkpoint_path: Path | None = None,
    rtd_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    manifest_names = {path.name for path in results_dir.glob("launch_manifest_shard_*.json")}
    expected_names = {f"launch_manifest_shard_{index:02d}.json" for index in range(SHARDS)}
    if manifest_names != expected_names:
        fail("Expected exactly 12 launch manifests with zero-padded shard IDs 00--11")
    manifests: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    for shard in range(SHARDS):
        path = results_dir / f"launch_manifest_shard_{shard:02d}.json"
        record = load_json(path)
        manifests.append(record)
        manifest_hashes[path.name] = sha256_file(path)
        if record.get("schema_version") != 1 or record.get("shard_id") != shard:
            fail(f"Launch manifest identity mismatch: {path.name}")
        if record.get("num_shards") != SHARDS:
            fail(f"Launch manifest shard count mismatch: {path.name}")
        checkpoints = record.get("checkpoints")
        if not isinstance(checkpoints, dict) or tuple(checkpoints) != MODELS:
            fail(f"Launch manifest checkpoint models mismatch: {path.name}")
        for model in MODELS:
            checkpoint = checkpoints[model]
            if not isinstance(checkpoint, dict):
                fail(f"Malformed checkpoint record: {path.name}/{model}")
            if checkpoint.get("path") != validated["checkpoint_paths"][model]:
                fail(f"Launch/result checkpoint path mismatch: {path.name}/{model}")
            if checkpoint.get("sha256") != validated["checkpoint_hashes"][model]:
                fail(f"Launch/result checkpoint hash mismatch: {path.name}/{model}")
        if record.get("runner_sha256") != validated["runner_sha256"]:
            fail(f"Launch/result runner hash mismatch: {path.name}")
        for name in (
            "runner_sha256",
            "wrapper_sha256",
            "job_yaml_sha256",
            "checkpoint_audit_sha256",
            "source_bundle_sha256",
        ):
            require_hash(record.get(name), f"{path.name}/{name}")

    consistent_fields = (
        "checkpoints",
        "runner_sha256",
        "wrapper_sha256",
        "job_yaml_sha256",
        "checkpoint_audit_sha256",
        "source_bundle_sha256",
        "python",
    )
    first = manifests[0]
    for field in consistent_fields:
        if any(record.get(field) != first.get(field) for record in manifests[1:]):
            fail(f"Launch manifests disagree on {field}")
    if first["runner_sha256"] != EXPECTED_RUNNER_SHA256:
        fail("Launch manifest runner hash is not the pinned launch hash")
    if first["wrapper_sha256"] != EXPECTED_WRAPPER_SHA256:
        fail("Launch manifest wrapper hash is not the pinned launch hash")
    if first["job_yaml_sha256"] != EXPECTED_JOB_YAML_SHA256:
        fail("Launch manifest job YAML hash is not the pinned launch hash")
    if first["source_bundle_sha256"] != expected_source_bundle_sha256:
        fail("Launch manifest source bundle differs from the independently supplied hash")
    for model in MODELS:
        if first["checkpoints"][model]["sha256"] != EXPECTED_CHECKPOINT_SHA256[model]:
            fail(f"Launch manifest {model} checkpoint is not pinned")

    audit_path = results_dir / "objective_checkpoint_audit.json"
    audit = load_json(audit_path)
    audit_hash = sha256_file(audit_path)
    if audit_hash != first["checkpoint_audit_sha256"]:
        fail("Launch manifests do not hash the supplied checkpoint tensor audit")
    if audit.get("schema_version") != 1:
        fail("Checkpoint tensor audit schema version mismatch")
    if audit.get("mlm_file_sha256") != validated["checkpoint_hashes"]["mlm_s2"]:
        fail("Checkpoint tensor audit MLM file hash mismatch")
    if audit.get("rtd_file_sha256") != validated["checkpoint_hashes"]["rtd25_s2"]:
        fail("Checkpoint tensor audit RTD file hash mismatch")
    mlm_tensor = require_hash(audit.get("mlm_canonical_tensor_sha256"), "MLM tensor hash")
    rtd_tensor = require_hash(audit.get("rtd_canonical_tensor_sha256"), "RTD tensor hash")
    if mlm_tensor == rtd_tensor:
        fail("Checkpoint tensor audit reports identical canonical tensor payloads")
    parameter_tensors = audit.get("parameter_tensors")
    tensors_differing = audit.get("tensors_differing")
    if (
        isinstance(parameter_tensors, bool)
        or not isinstance(parameter_tensors, int)
        or isinstance(tensors_differing, bool)
        or not isinstance(tensors_differing, int)
        or not 0 < tensors_differing <= parameter_tensors
    ):
        fail("Checkpoint tensor audit does not report a valid differing-tensor count")
    if finite_float(audit.get("maximum_absolute_tensor_delta"), "maximum tensor delta") <= 0.0:
        fail("Checkpoint tensor audit maximum delta is not positive")
    if finite_float(audit.get("global_relative_l2_delta"), "relative L2 delta") <= 0.0:
        fail("Checkpoint tensor audit relative L2 delta is not positive")
    config = audit.get("resolved_encoder_config")
    if not isinstance(config, dict) or set(config) != set(ENCODER_CONFIG_FIELDS):
        fail("Checkpoint tensor attestation resolved encoder config schema mismatch")
    for key, value in config.items():
        if not isinstance(value, (str, int, float, bool)) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            fail(f"Checkpoint tensor attestation config value is invalid: {key}")
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
        if config[key] != EXPECTED_ARCHITECTURE[key]:
            fail(f"Checkpoint tensor attestation architecture mismatch: {key}")

    verification_mode = "launch_attestation_only"
    if (mlm_checkpoint_path is None) != (rtd_checkpoint_path is None):
        fail("Supply both checkpoint files for independent tensor verification, or neither")
    if mlm_checkpoint_path is not None and rtd_checkpoint_path is not None:
        independently_verify_checkpoints(
            mlm_checkpoint_path=mlm_checkpoint_path,
            rtd_checkpoint_path=rtd_checkpoint_path,
            audit=audit,
        )
        verification_mode = "independently_recomputed"
    public_audit = {
        name: audit[name]
        for name in (
            "schema_version",
            "mlm_file_sha256",
            "rtd_file_sha256",
            "mlm_canonical_tensor_sha256",
            "rtd_canonical_tensor_sha256",
            "parameter_tensors",
            "tensors_differing",
            "maximum_absolute_tensor_delta",
            "global_relative_l2_delta",
            "resolved_encoder_config",
        )
    }
    return {
        "manifest_hashes": dict(sorted(manifest_hashes.items())),
        "audit_sha256": audit_hash,
        "audit": audit,
        "public_audit": public_audit,
        "verification_mode": verification_mode,
        "source_hashes": {
            name: first[name]
            for name in ("runner_sha256", "wrapper_sha256", "job_yaml_sha256", "source_bundle_sha256")
        },
        "python": first.get("python"),
    }


def build_endpoint_rows(cells: dict[tuple[str, str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        means: dict[tuple[str, str], float] = {}
        for model in MODELS:
            for method in METHODS:
                means[(model, method)] = statistics.fmean(
                    cells[(model, endpoint, split, method)]["test_mae"] for split in SPLITS
                )
        frozen_delta = means[("mlm_s2", "frozen_mix")] - means[("rtd25_s2", "frozen_mix")]
        full_delta = means[("mlm_s2", "full_mix")] - means[("rtd25_s2", "full_mix")]
        rows.append(
            {
                "endpoint": endpoint,
                "mlm_frozen_mean_mae": means[("mlm_s2", "frozen_mix")],
                "rtd_frozen_mean_mae": means[("rtd25_s2", "frozen_mix")],
                "frozen_mlm_minus_rtd": frozen_delta,
                "mlm_full_mean_mae": means[("mlm_s2", "full_mix")],
                "rtd_full_mean_mae": means[("rtd25_s2", "full_mix")],
                "full_mlm_minus_rtd": full_delta,
                "full_minus_frozen_interaction": full_delta - frozen_delta,
            }
        )
    return rows


def analyze_rows(
    rows: list[dict[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    fields = {
        "frozen_objective_delta": "frozen_mlm_minus_rtd",
        "full_objective_delta": "full_mlm_minus_rtd",
        "full_minus_frozen_interaction": "full_minus_frozen_interaction",
    }
    definitions = {
        "frozen_objective_delta": "endpoint mean frozen MLM MAE minus frozen RTD MAE; positive favors RTD",
        "full_objective_delta": "endpoint mean full-FT MLM MAE minus full-FT RTD MAE; positive favors RTD",
        "full_minus_frozen_interaction": "full objective delta minus frozen objective delta; positive means RTD's relative advantage is larger under full fine-tuning",
    }
    analyses: list[dict[str, Any]] = []
    for name in CONTRASTS:
        values = [float(row[fields[name]]) for row in rows]
        low, high = bootstrap_mean_ci(
            values, replicates=bootstrap_replicates, seed=bootstrap_seed
        )
        analyses.append(
            {
                "contrast": name,
                "definition": definitions[name],
                "endpoints": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "positive": sum(value > 0.0 for value in values),
                "negative": sum(value < 0.0 for value in values),
                "ties": sum(value == 0.0 for value in values),
                "bootstrap_95pct_ci": [low, high],
                "wilcoxon": exact_wilcoxon_two_sided(values),
                "sign_test": exact_sign_test_two_sided(values),
            }
        )
    wilcoxon_adjusted = holm_adjust(
        [analysis["wilcoxon"]["p_two_sided_exact"] for analysis in analyses]
    )
    sign_adjusted = holm_adjust(
        [analysis["sign_test"]["p_two_sided_exact"] for analysis in analyses]
    )
    for analysis, wilcoxon_p, sign_p in zip(analyses, wilcoxon_adjusted, sign_adjusted):
        analysis["wilcoxon"]["p_holm_3"] = wilcoxon_p
        analysis["sign_test"]["p_holm_3"] = sign_p
    return {
        "endpoint_as_inferential_unit": True,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interval": "two-sided percentile interval with linear-interpolated 2.5th and 97.5th percentiles",
            "resampling": "23 endpoints with replacement; random.Random; generator reset to the recorded seed for each contrast",
        },
        "multiplicity": {
            "contrasts": list(CONTRASTS),
            "wilcoxon_family": "Holm adjustment across the three exact paired Wilcoxon tests",
            "sign_test_family": "separate Holm adjustment across the three exact paired sign tests",
            "family_size": 3,
        },
        "contrasts": analyses,
    }


CSV_FIELDS = [
    "endpoint",
    "mlm_frozen_mean_mae",
    "rtd_frozen_mean_mae",
    "frozen_mlm_minus_rtd",
    "mlm_full_mean_mae",
    "rtd_full_mean_mae",
    "full_mlm_minus_rtd",
    "full_minus_frozen_interaction",
]


def csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def results_markdown(anonymous: dict[str, Any]) -> str:
    by_name = {item["contrast"]: item for item in anonymous["analysis"]["contrasts"]}
    lines = [
        f"# {anonymous['status']} primary-protocol MLM versus RTD objective rerun",
        "",
        "All 46 objective/endpoint results, 276 split/method cells, 12 launch manifests, "
        "12 shard markers, raw predictions, canonical split identities, and distinct checkpoint "
        f"tensor provenance passed the fail-closed audit ({anonymous['checkpoint_verification_mode']}).",
        "",
        "Positive MLM minus RTD MAE favors RTD. The interaction is the full-finetuning "
        "objective delta minus the frozen objective delta.",
        "",
        "| Contrast | Mean | 95% endpoint bootstrap CI | Positive / negative / tie | Exact Wilcoxon Holm-3 p | Exact sign Holm-3 p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONTRASTS:
        item = by_name[name]
        low, high = item["bootstrap_95pct_ci"]
        lines.append(
            f"| {name.replace('_', ' ')} | {item['mean']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {item['positive']} / {item['negative']} / {item['ties']} | "
            f"{item['wilcoxon']['p_holm_3']:.8g} | {item['sign_test']['p_holm_3']:.8g} |"
        )
    bootstrap = anonymous["analysis"]["bootstrap"]
    lines += [
        "",
        f"Inference uses 23 endpoint-level paired effects. Bootstrap intervals use "
        f"{bootstrap['replicates']:,} deterministic endpoint resamples with seed "
        f"{bootstrap['seed']}. Exact Wilcoxon and exact sign-test p-values are adjusted in "
        "separate Holm families of three contrasts.",
    ]
    return "\n".join(lines) + "\n"


def privacy_scan(texts: dict[str, str]) -> None:
    for filename, text in texts.items():
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                fail(f"Anonymous output {filename} contains a {label}")


def require_disjoint_absent_output(
    *, output_dir: Path, results_dir: Path, reference_dir: Path
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        fail("Output directory must be absent; stale evidence is never overwritten")
    output = output_dir.resolve(strict=False)
    for source, label in ((results_dir, "results"), (reference_dir, "reference")):
        resolved = source.resolve()
        if output == resolved or output.is_relative_to(resolved) or resolved.is_relative_to(output):
            fail(f"Output directory must be disjoint from the {label} directory")


def run_certification(
    *,
    results_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    run_attestation_path: Path,
    expected_source_bundle_sha256: str,
    expected_reference_manifest_sha256: str,
    expected_archive_sha256: str,
    mlm_checkpoint_path: Path | None = None,
    rtd_checkpoint_path: Path | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    require_disjoint_absent_output(
        output_dir=output_dir, results_dir=results_dir, reference_dir=reference_dir
    )
    require_hash(expected_reference_manifest_sha256, "expected reference manifest hash")
    attestation = validate_attestation(
        results_dir=results_dir,
        attestation_path=run_attestation_path,
        expected_source_bundle_sha256=expected_source_bundle_sha256,
        expected_archive_sha256=expected_archive_sha256,
    )
    actual_reference_manifest_sha256 = reference_manifest_digest(reference_dir)
    if actual_reference_manifest_sha256 != expected_reference_manifest_sha256:
        fail("Canonical reference relevant-file manifest hash mismatch")
    reference = load_reference(reference_dir)
    validated = validate_results(results_dir, reference)
    validate_chronology(results_dir=results_dir, validated=validated, attestation=attestation)
    provenance = validate_provenance(
        results_dir,
        validated,
        expected_source_bundle_sha256=expected_source_bundle_sha256,
        mlm_checkpoint_path=mlm_checkpoint_path,
        rtd_checkpoint_path=rtd_checkpoint_path,
    )
    rows = build_endpoint_rows(validated["cells"])
    analysis = analyze_rows(
        rows, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed
    )
    certifier_path = Path(__file__).resolve()
    canonical_statistics = (
        bootstrap_replicates == BOOTSTRAP_REPLICATES and bootstrap_seed == BOOTSTRAP_SEED
    )
    status = "PASS" if canonical_statistics else "TEST_ONLY"
    anonymous = {
        "schema_version": 1,
        "status": status,
        "comparison": "paired primary-protocol MLM-S2 versus RTD-25%-S2 on 23 OpenADMET endpoints",
        "models": list(MODELS),
        "methods": list(METHODS),
        "splits": list(SPLITS),
        "endpoints": list(ENDPOINTS),
        "validated_counts": {
            "endpoint_results": len(MODELS) * len(ENDPOINTS),
            "split_method_cells": len(MODELS) * len(ENDPOINTS) * len(SPLITS) * len(METHODS),
            "launch_manifests": SHARDS,
            "shard_completion_markers": SHARDS,
        },
        "checkpoint_sha256": validated["checkpoint_hashes"],
        "checkpoint_tensor_audit": provenance["public_audit"],
        "checkpoint_tensor_audit_sha256": provenance["audit_sha256"],
        "checkpoint_verification_mode": provenance["verification_mode"],
        "source_hashes": provenance["source_hashes"],
        "reference_relevant_file_manifest_sha256": actual_reference_manifest_sha256,
        "remote_run_attestation_sha256": attestation["sha256"],
        "remote_archive_sha256": expected_archive_sha256,
        "certifier_sha256": sha256_file(certifier_path),
        "analysis": analysis,
        "endpoint_csv_sha256": sha256_text(csv_text(rows)),
    }
    anonymous_text = strict_json(anonymous)
    endpoint_text = csv_text(rows)
    markdown_text = results_markdown(anonymous)
    privacy_scan(
        {
            "summary_anonymous.json": anonymous_text,
            "objective_endpoint_effects.csv": endpoint_text,
            "RESULTS.md": markdown_text,
        }
    )
    internal = {
        **anonymous,
        "inputs": {
            "results_dir": str(results_dir.resolve()),
            "reference_dir": str(reference_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "input_artifact_sha256": {
            "result_files": validated["result_hashes"],
            "standalone_cell_files": validated["cell_hashes"],
            "launch_manifests": provenance["manifest_hashes"],
            "remote_run_attestation": attestation["sha256"],
            "canonical_reference_relevant_manifest": actual_reference_manifest_sha256,
            "remote_archive": expected_archive_sha256,
        },
        "completion_markers": validated["marker_values"],
        "global_completion_marker": validated["complete_value"],
        "pinned_job": {
            "uid": EXPECTED_JOB_UID,
            "creation_timestamp": EXPECTED_JOB_CREATION,
            "start_timestamp": EXPECTED_JOB_START,
            "image_id": EXPECTED_IMAGE_ID,
        },
        "runtime": {"python": provenance["python"]},
        "endpoint_rows": rows,
    }
    internal_text = strict_json(internal)

    # Publish the complete exact bundle as one directory transaction. A failed
    # certification cannot leave or preserve a PASS at the requested path.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        (staging / "summary.json").write_text(internal_text, encoding="utf-8", newline="\n")
        (staging / "summary_anonymous.json").write_text(
            anonymous_text, encoding="utf-8", newline="\n"
        )
        (staging / "objective_endpoint_effects.csv").write_text(
            endpoint_text, encoding="utf-8", newline="\n"
        )
        (staging / "RESULTS.md").write_text(markdown_text, encoding="utf-8", newline="\n")
        expected_outputs = {
            "summary.json", "summary_anonymous.json", "objective_endpoint_effects.csv", "RESULTS.md"
        }
        if {path.name for path in staging.iterdir()} != expected_outputs:
            fail("Staged certification output inventory mismatch")
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return internal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-attestation", required=True, type=Path)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-reference-manifest-sha256", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--mlm-checkpoint", type=Path)
    parser.add_argument("--rtd-checkpoint", type=Path)
    args = parser.parse_args()
    summary = run_certification(
        results_dir=args.results_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        run_attestation_path=args.run_attestation,
        expected_source_bundle_sha256=args.expected_source_bundle_sha256,
        expected_reference_manifest_sha256=args.expected_reference_manifest_sha256,
        expected_archive_sha256=args.expected_archive_sha256,
        mlm_checkpoint_path=args.mlm_checkpoint,
        rtd_checkpoint_path=args.rtd_checkpoint,
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
