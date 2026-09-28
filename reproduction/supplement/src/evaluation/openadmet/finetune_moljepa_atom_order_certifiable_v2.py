#!/usr/bin/env python3
"""Certifiable v2 adapter for the Mol-JEPA atom-order experiment.

Training and inference deliberately reuse the locked implementation in
``finetune_moljepa_benchmarks.py``.  This adapter changes the artifact
contract: it binds every cell to an immutable run receipt, captures a complete
runtime contract, and records ordered hashes that the v1 experiment omitted.
The base runner is itself pinned in the run receipt/source-bundle hash.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
import torch
import torch_geometric

import finetune_moljepa_benchmarks as base


SCHEMA_VERSION = 2
EXPECTED_RDKIT_DISTRIBUTION = "2025.3.6"
EXPECTED_RDKIT_MODULE = "2025.03.6"
VALIDATION_METRIC_ROLE = (
    "training provenance only; validation predictions were not persisted, so "
    "validation_mae is not independently recertified"
)
REQUIRED_ENV = (
    "RUN_ID", "RUN_STARTED_AT_UTC", "JOB_UID", "JOB_NAME", "POD_UID",
    "POD_NAME", "POD_NAMESPACE", "POD_STARTED_AT_UTC", "SHARD_ID", "IMAGE_REFERENCE",
    "IMAGE_ID",
    "RUN_RECEIPT_SHA256", "CERTIFIABLE_RUNNER_SHA256", "BASE_RUNNER_SHA256",
    "WRAPPER_SHA256", "CERTIFIER_SHA256", "V1_CERTIFIER_SHA256",
    "SOURCE_BUNDLE_SHA256", "REFERENCE_RESULTS_SHA256",
    "PRELAUNCH_MANIFEST", "PRELAUNCH_MANIFEST_SHA256",
)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def ordered_strings_sha256(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def ordered_rows_sha256(smiles, targets) -> str:
    return stable_hash([
        [str(smile), float(target)]
        for smile, target in zip(smiles, targets, strict=True)
    ])


def required_environment() -> dict[str, str]:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing certifiable-run environment: {missing}")
    return {name: os.environ[name] for name in REQUIRED_ENV}


def runtime_contract() -> dict[str, Any]:
    distribution_version = importlib.metadata.version("rdkit")
    if distribution_version != EXPECTED_RDKIT_DISTRIBUTION:
        raise RuntimeError(
            f"rdkit distribution {distribution_version}, expected {EXPECTED_RDKIT_DISTRIBUTION}"
        )
    if rdkit.__version__ != EXPECTED_RDKIT_MODULE:
        raise RuntimeError(
            f"rdkit module {rdkit.__version__}, expected {EXPECTED_RDKIT_MODULE}"
        )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rdkit_distribution": distribution_version,
        "rdkit_module": rdkit.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn": str(torch.backends.cudnn.version()),
        "image_reference": os.environ["IMAGE_REFERENCE"],
        "image_id": os.environ["IMAGE_ID"],
    }


def run_provenance(environment: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": environment["RUN_ID"],
        "run_started_at_utc": environment["RUN_STARTED_AT_UTC"],
        "pod_started_at_utc": environment["POD_STARTED_AT_UTC"],
        "job_uid": environment["JOB_UID"],
        "job_name": environment["JOB_NAME"],
        "pod_uid": environment["POD_UID"],
        "pod_name": environment["POD_NAME"],
        "pod_namespace": environment["POD_NAMESPACE"],
        "shard_id": int(environment["SHARD_ID"]),
        "run_receipt_sha256": environment["RUN_RECEIPT_SHA256"],
        "prelaunch_manifest_sha256": environment["PRELAUNCH_MANIFEST_SHA256"],
        "reference_results_sha256": environment["REFERENCE_RESULTS_SHA256"],
        "source_hashes": {
            "certifiable_runner_sha256": environment["CERTIFIABLE_RUNNER_SHA256"],
            "base_runner_sha256": environment["BASE_RUNNER_SHA256"],
            "wrapper_sha256": environment["WRAPPER_SHA256"],
            "certifier_sha256": environment["CERTIFIER_SHA256"],
            "v1_certifier_sha256": environment["V1_CERTIFIER_SHA256"],
            "source_bundle_sha256": environment["SOURCE_BUNDLE_SHA256"],
        },
    }


def main() -> None:
    environment = required_environment()
    runtime = runtime_contract()
    deployment = json.loads(
        Path(environment["PRELAUNCH_MANIFEST"]).read_text(encoding="utf-8")
    )
    for key, expected in deployment["expected_runtime"].items():
        if runtime.get(key) != expected:
            raise RuntimeError(
                f"runtime {key}={runtime.get(key)!r}, prelaunch contract requires {expected!r}"
            )
    provenance = run_provenance(environment)

    def artifact_provenance() -> dict[str, Any]:
        payload = dict(provenance)
        payload["source_hashes"] = dict(provenance["source_hashes"])
        payload["artifact_created_at_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return payload
    adapter_path = Path(__file__).resolve()
    base_path = Path(base.__file__).resolve()
    original_file_sha256 = base.file_sha256
    original_train_cell = base.train_cell
    original_valid_json = base.valid_json
    original_atomic_json = base.atomic_json

    if original_file_sha256(adapter_path) != environment["CERTIFIABLE_RUNNER_SHA256"]:
        raise RuntimeError("certifiable runner hash differs from run receipt")
    if original_file_sha256(base_path) != environment["BASE_RUNNER_SHA256"]:
        raise RuntimeError("base runner hash differs from run receipt")

    def receipt_bound_file_sha256(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved == base_path:
            return environment["CERTIFIABLE_RUNNER_SHA256"]
        return original_file_sha256(resolved)

    def receipt_bound_valid_json(path: Path):
        payload = original_valid_json(path)
        if payload is None:
            return None
        artifact_run = payload.get("certifiable_run", {})
        if (
            artifact_run.get("run_id") != environment["RUN_ID"]
            or artifact_run.get("job_uid") != environment["JOB_UID"]
            or artifact_run.get("run_receipt_sha256") != environment["RUN_RECEIPT_SHA256"]
            or artifact_run.get("prelaunch_manifest_sha256")
                != environment["PRELAUNCH_MANIFEST_SHA256"]
            or artifact_run.get("source_hashes") != provenance["source_hashes"]
        ):
            return None
        return payload

    def certifiable_train_cell(
        encoder_state, dictionary, train_frame, valid_frame, test_frame,
        method: str, seed: int,
    ):
        cell = original_train_cell(
            encoder_state, dictionary, train_frame, valid_frame, test_frame,
            method, seed,
        )
        if method != "full_mix" or base.ATOM_ORDER_PERMUTATIONS != 10:
            raise RuntimeError("v2 certifiable runner requires full_mix and ten permutations")
        audit = cell.get("atom_order_robustness")
        if not isinstance(audit, dict) or len(audit.get("records", [])) != 10:
            raise RuntimeError("atom-order records are incomplete")
        original_smiles = list(map(str, test_frame["smiles"].tolist()))
        original_targets = test_frame["y"].to_numpy(dtype=float).tolist()
        cell["ordered_original_test_smiles_sha256"] = ordered_strings_sha256(original_smiles)
        cell["ordered_original_test_rows_sha256"] = ordered_rows_sha256(
            original_smiles, original_targets
        )
        for record in audit["records"]:
            index = int(record["permutation_index"])
            expected_seed = seed * 1000 + index
            if int(record["seed"]) != expected_seed:
                raise RuntimeError("permutation seed drifted before v2 publication")
            permuted = base.randomized_atom_order_smiles(original_smiles, expected_seed)
            if base.smiles_sha256(permuted) != record["smiles_sha256"]:
                raise RuntimeError("regenerated legacy sorted permutation hash differs")
            record["ordered_smiles_sha256"] = ordered_strings_sha256(permuted)
        cell["validation_metric_role"] = VALIDATION_METRIC_ROLE
        cell["certifiable_run"] = artifact_provenance()
        cell["runtime_contract"] = runtime
        return cell

    def certifiable_atomic_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name.startswith("results_"):
            payload = dict(payload)
            payload["schema_version"] = SCHEMA_VERSION
            payload["validation_metric_role"] = VALIDATION_METRIC_ROLE
            payload["certifiable_run"] = artifact_provenance()
            payload["runtime_contract"] = runtime
        original_atomic_json(path, payload)

    base.file_sha256 = receipt_bound_file_sha256
    base.valid_json = receipt_bound_valid_json
    base.train_cell = certifiable_train_cell
    base.atomic_json = certifiable_atomic_json
    base.main()


if __name__ == "__main__":
    main()
