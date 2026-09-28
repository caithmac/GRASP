#!/usr/bin/env python3
"""Audited CPU-only classical baselines for the historical TDC panel.

The data path deliberately bypasses PyTDC at runtime while reproducing the
PyTDC 0.4.1 loader and scaffold splitter used by ``finetune_benchmarks.py``.
This makes the exact raw snapshot, split membership, labels, and software
versions explicit in every result.  Hyperparameters are selected using only
the fixed validation partition; the held-out test partition is evaluated
only after that selection is locked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr


TASKS = (
    "bbb_martins", "clintox", "hia_hou", "lipophilicity_astrazeneca",
    "herg", "ames", "dili", "bioavailability_ma", "caco2_wang",
    "pgp_broccatelli", "solubility_aqsoldb", "ppbr_az", "vdss_lombardo",
    "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith",
    "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels", "half_life_obach",
    "clearance_microsome_az", "clearance_hepatocyte_az", "ld50_zhu",
)

TASK_CONFIG = {
    "vdss_lombardo": ("regression", "spearman"),
    "clearance_microsome_az": ("regression", "spearman"),
    "clearance_hepatocyte_az": ("regression", "spearman"),
    "half_life_obach": ("regression", "spearman"),
    "cyp2d6_veith": ("classification", "auprc"),
    "cyp3a4_veith": ("classification", "auprc"),
    "cyp2c9_veith": ("classification", "auprc"),
    "cyp2d6_substrate_carbonmangels": ("classification", "auprc"),
    "cyp2c9_substrate_carbonmangels": ("classification", "auprc"),
    "ppbr_az": ("regression", "mae"),
    "solubility_aqsoldb": ("regression", "mae"),
    "lipophilicity_astrazeneca": ("regression", "mae"),
    "caco2_wang": ("regression", "mae"),
    "ld50_zhu": ("regression", "mae"),
}

REPRESENTATIONS = ("ecfp4", "rdkit_descriptors")
FINAL_SEEDS = (0, 1, 2)
SPLIT_SEED = 42
SPLIT_FRAC = (0.70, 0.15, 0.15)
MAX_HEAVY_ATOMS = 96
ECFP_BITS = 2048
RDLogger.DisableLog("rdApp.*")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def row_hash(frame: pd.DataFrame) -> str:
    """Hash ordered IDs, SMILES, and labels without lossy float formatting."""
    digest = hashlib.sha256()
    for row in frame[["Drug_ID", "Drug", "Y"]].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def membership_hash(frame: pd.DataFrame) -> str:
    # Drug_ID is not unique in every PyTDC snapshot.  The ID+SMILES pair is
    # the stable membership key; labels are covered by row_hash separately.
    values = sorted(
        json.dumps((str(drug_id), str(smiles)), ensure_ascii=False, separators=(",", ":"))
        for drug_id, smiles in frame[["Drug_ID", "Drug"]].itertuples(index=False, name=None)
    )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_pytdc041_snapshot(path: Path, task: str) -> pd.DataFrame:
    """Mirror PyTDC 0.4.1 ``pd_load``/``property_dataset_load``/ADME PPBR."""
    frame = pd.read_csv(path, sep="\t")
    if task == "ppbr_az":
        # ADME.__init__ rereads the raw file and selects humans after the base
        # loader; this is why this branch intentionally precedes drop_duplicates.
        frame = frame.loc[frame["Species"].eq("Homo sapiens")].copy()
    else:
        frame = frame.drop_duplicates().copy()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = frame.rename(columns={"ID": "Drug_ID", "X": "Drug"})
    frame = frame.loc[frame["Y"].notna()].reset_index(drop=True)
    required = {"Drug_ID", "Drug", "Y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks {sorted(required - set(frame.columns))}")
    return frame


def pytdc041_scaffold_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exact membership logic from PyTDC 0.4.1 create_scaffold_split."""
    scaffolds: defaultdict[str, set[int]] = defaultdict(set)
    errors = 0
    for index, smiles in enumerate(frame["Drug"].values):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].add(index)
        except Exception:
            errors += 1
    train_size = int((len(frame) - errors) * SPLIT_FRAC[0])
    valid_size = int((len(frame) - errors) * SPLIT_FRAC[1])
    test_size = len(frame) - errors - train_size - valid_size
    big, small = [], []
    for index_set in scaffolds.values():
        target = big if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2 else small
        target.append(index_set)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(big)
    rng.shuffle(small)
    indices: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for index_set in big + small:
        if len(indices["train"]) + len(index_set) <= train_size:
            indices["train"] += index_set
        elif len(indices["valid"]) + len(index_set) <= valid_size:
            indices["valid"] += index_set
        else:
            indices["test"] += index_set
    return {name: frame.iloc[index].reset_index(drop=True) for name, index in indices.items()}


def valid_molecule(smiles: Any) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or not 0 < mol.GetNumHeavyAtoms() <= MAX_HEAVY_ATOMS:
        return None
    return mol


def effective_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[Chem.Mol]]:
    keep, molecules = [], []
    for index, value in enumerate(frame["Drug"]):
        mol = valid_molecule(value)
        if mol is not None:
            keep.append(index)
            molecules.append(mol)
    return frame.iloc[keep].reset_index(drop=True), molecules


def split_audit(task: str, source_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, list[Chem.Mol]], dict[str, Any]]:
    loaded = load_pytdc041_snapshot(source_path, task)
    raw_split = pytdc041_scaffold_split(loaded)
    effective, molecules, records = {}, {}, {}
    for role in ("train", "valid", "test"):
        effective[role], molecules[role] = effective_split(raw_split[role])
        records[role] = {
            "pre_filter_rows": int(len(raw_split[role])),
            "effective_rows": int(len(effective[role])),
            "removed_invalid_or_over_96": int(len(raw_split[role]) - len(effective[role])),
            "membership_sha256": membership_hash(effective[role]),
            "ordered_rows_and_labels_sha256": row_hash(effective[role]),
        }
    memberships = [
        set(zip(part["Drug_ID"].astype(str), part["Drug"].astype(str)))
        for part in effective.values()
    ]
    if any(memberships[i] & memberships[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"{task}: overlapping split memberships")
    role_order = ("train", "valid", "test")
    canonical_sets = {
        role: {Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in molecules[role]}
        for role in role_order
    }
    scaffold_sets = {
        role: {MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) for mol in molecules[role]}
        for role in role_order
    }
    canonical_overlaps = {
        f"{left}_{right}": len(canonical_sets[left] & canonical_sets[right])
        for index, left in enumerate(role_order)
        for right in role_order[index + 1:]
    }
    scaffold_overlaps = {
        f"{left}_{right}": len(scaffold_sets[left] & scaffold_sets[right])
        for index, left in enumerate(role_order)
        for right in role_order[index + 1:]
    }
    if any(canonical_overlaps.values()) or any(scaffold_overlaps.values()):
        raise RuntimeError(
            f"{task}: chemical leakage across scaffold partitions; "
            f"canonical={canonical_overlaps}, scaffolds={scaffold_overlaps}"
        )
    audit = {
        "task": task,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "loader_rows": int(len(loaded)),
        "loader_rows_and_labels_sha256": row_hash(loaded),
        "split_method": "PyTDC 0.4.1 create_scaffold_split reproduction",
        "split_seed": SPLIT_SEED,
        "split_frac": list(SPLIT_FRAC),
        "post_split_filter": "RDKit parseable and 1..96 heavy atoms",
        "partitions": records,
        "leakage_audit": {
            "membership_key": "Drug_ID plus original SMILES",
            "canonical_smiles_overlap_counts": canonical_overlaps,
            "murcko_scaffold_overlap_counts": scaffold_overlaps,
            "status": "PASS",
        },
    }
    return effective, molecules, audit


def features(molecules: Iterable[Chem.Mol], representation: str) -> tuple[np.ndarray, list[str]]:
    molecules = list(molecules)
    if representation == "ecfp4":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_BITS)
        matrix = np.zeros((len(molecules), ECFP_BITS), dtype=np.uint8)
        for row, molecule in enumerate(molecules):
            fingerprint = generator.GetFingerprint(molecule)
            DataStructs.ConvertToNumpyArray(fingerprint, matrix[row])
        return matrix, [f"ecfp4_{i}" for i in range(ECFP_BITS)]
    if representation == "rdkit_descriptors":
        names = [name for name, _ in Descriptors.descList]
        functions = [function for _, function in Descriptors.descList]
        matrix = np.empty((len(molecules), len(functions)), dtype=np.float64)
        for row, molecule in enumerate(molecules):
            for column, function in enumerate(functions):
                try:
                    value = float(function(molecule))
                except Exception:
                    value = float("nan")
                matrix[row, column] = value if math.isfinite(value) else float("nan")
        return matrix, names
    raise ValueError(representation)


def task_type_metric(task: str, labels: np.ndarray) -> tuple[str, str]:
    if task in TASK_CONFIG:
        return TASK_CONFIG[task]
    unique = set(np.unique(labels).tolist())
    return ("classification", "auroc") if unique <= {0, 1, 0.0, 1.0} else ("regression", "mae")


def score_predictions(task_type: str, metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if task_type == "classification":
        return float(average_precision_score(labels, predictions) if metric == "auprc" else roc_auc_score(labels, predictions))
    if metric == "spearman":
        return float(spearmanr(labels, predictions).statistic)
    return float(mean_absolute_error(labels, predictions))


def utility(metric: str, score: float) -> float:
    return -score if metric == "mae" else score


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"max_features": "sqrt", "min_samples_leaf": 1},
        {"max_features": "sqrt", "min_samples_leaf": 2},
        {"max_features": 0.25, "min_samples_leaf": 1},
        {"max_features": 0.50, "min_samples_leaf": 2},
    ]


def make_estimator(task_type: str, params: dict[str, Any], seed: int, n_estimators: int):
    common = dict(
        n_estimators=n_estimators,
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        n_jobs=-1,
        random_state=seed,
    )
    if task_type == "classification":
        return ExtraTreesClassifier(class_weight="balanced", **common)
    return ExtraTreesRegressor(**common)


def predict(estimator, matrix: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        return estimator.predict_proba(matrix)[:, 1]
    return estimator.predict(matrix)


def run_representation(
    representation: str,
    task_type: str,
    metric: str,
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    n_estimators: int,
) -> dict[str, Any]:
    imputer = None
    if representation == "rdkit_descriptors":
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        matrices = {
            "train": imputer.fit_transform(matrices["train"]),
            "valid": imputer.transform(matrices["valid"]),
            "test": imputer.transform(matrices["test"]),
        }
    validation = []
    for index, params in enumerate(candidate_grid()):
        estimator = make_estimator(task_type, params, seed=1000 + index, n_estimators=n_estimators)
        estimator.fit(matrices["train"], labels["train"])
        score = score_predictions(task_type, metric, labels["valid"], predict(estimator, matrices["valid"], task_type))
        validation.append({"params": params, "validation_score": score, "validation_utility": utility(metric, score)})
    selected = max(validation, key=lambda item: (item["validation_utility"], -item["params"]["min_samples_leaf"]))
    test_runs = []
    for seed in FINAL_SEEDS:
        estimator = make_estimator(task_type, selected["params"], seed=seed, n_estimators=n_estimators)
        estimator.fit(matrices["train"], labels["train"])
        score = score_predictions(task_type, metric, labels["test"], predict(estimator, matrices["test"], task_type))
        test_runs.append({"seed": seed, "test_score": score})
    scores = np.asarray([item["test_score"] for item in test_runs], dtype=float)
    return {
        "estimator": "ExtraTreesClassifier" if task_type == "classification" else "ExtraTreesRegressor",
        "n_estimators": n_estimators,
        "selection_rule": "maximum primary-metric validation utility; test unseen until configuration locked",
        "validation_candidates": validation,
        "selected_params": selected["params"],
        "test_runs": test_runs,
        "test_mean": float(scores.mean()),
        "test_std_ddof0": float(scores.std(ddof=0)),
        "imputation": "training-partition median" if imputer is not None else None,
    }


def run_task(task: str, data_dir: Path, output_dir: Path, reference: dict[str, Any] | None, n_estimators: int, audit_only: bool) -> dict[str, Any]:
    source = data_dir / f"{task}.tab"
    if not source.is_file():
        raise FileNotFoundError(source)
    splits, molecules, audit = split_audit(task, source)
    if reference is not None:
        expected = reference["tasks"][task]
        comparable = {
            "source_sha256": audit["source_sha256"],
            "loader_rows": audit["loader_rows"],
            "partitions": audit["partitions"],
        }
        if comparable != expected:
            raise RuntimeError(f"{task}: split provenance mismatch against locked reference")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": task,
        "split_audit": audit,
        "software": {
            "python": platform.python_version(), "rdkit": rdkit.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "protocol": {
            "inference_unit": "endpoint",
            "representations": list(REPRESENTATIONS),
            "final_seeds": list(FINAL_SEEDS),
            "n_estimators": n_estimators,
            "test_policy": "not accessed for hyperparameter selection",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "reference_content_sha256": canonical_json_sha256(reference) if reference is not None else None,
        },
    }
    if not audit_only:
        labels = {role: splits[role]["Y"].to_numpy(dtype=float) for role in splits}
        task_type, metric = task_type_metric(task, labels["train"])
        payload.update({"task_type": task_type, "primary_metric": metric, "representations": {}})
        for representation in REPRESENTATIONS:
            matrices, feature_names = {}, None
            for role in ("train", "valid", "test"):
                matrices[role], names = features(molecules[role], representation)
                feature_names = names if feature_names is None else feature_names
            payload["representations"][representation] = {
                "feature_count": len(feature_names or []),
                "feature_schema_sha256": hashlib.sha256("\n".join(feature_names or []).encode()).hexdigest(),
                **run_representation(representation, task_type, metric, matrices, labels, n_estimators),
            }
    atomic_json(output_dir / f"results_{task}.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--n-estimators", type=int, default=512)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8")) if args.reference_manifest else None
    payload = run_task(args.task, args.data_dir, args.output_dir, reference, args.n_estimators, args.audit_only)
    print(json.dumps({"task": args.task, "output": str(args.output_dir), "audit_only": args.audit_only,
                      "test": None if args.audit_only else {k: v["test_mean"] for k, v in payload["representations"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
