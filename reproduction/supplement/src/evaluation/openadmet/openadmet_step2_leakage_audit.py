#!/usr/bin/env python3
"""Exact ChEMBL Step-2/OpenADMET structure and label-overlap audit.

This runner deliberately separates three resumable stages:

``prepare``
    Validate the final 511,898 x 642 ChEMBL matrix, standardize both corpora,
    retain endpoint/split membership, compute exact structural-key overlaps,
    and prepare the FPSim2 input file.
``search``
    Run an indexed shard of exact FPSim2 ECFP4 Tanimoto top-k searches.
``finalize``
    Aggregate shards, attach Step-2 labels/assay metadata, summarize each
    endpoint/split, and independently validate FPSim2 with brute-force RDKit
    BulkTanimotoSimilarity over the complete Step-2 corpus.

No approximate-nearest-neighbour method is used.  Acyclic molecules have no
Bemis--Murcko scaffold and are not collapsed into an empty-scaffold bucket.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


EXPECTED_STEP2_ROWS = 511_898
EXPECTED_ASSAYS = 642
FP_RADIUS = 2
FP_SIZE = 2048
THRESHOLDS = (0.70, 0.80, 0.90, 0.95)
NEURAL_VALIDATION_FRACTION = 0.15
STRUCTURE_KEYS = (
    "standardized_exact",
    "connectivity_inchikey",
    "bemis_murcko",
    "generic_scaffold",
)
SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


class MoleculeStandardizer:
    """Deterministic RDKit parent standardization used for both corpora."""

    def __init__(self) -> None:
        self.uncharger = rdMolStandardize.Uncharger()

    def transform(self, smiles: str) -> dict[str, object]:
        original = _clean_text(smiles).strip()
        if not original:
            return self.invalid("empty_smiles")
        try:
            mol = Chem.MolFromSmiles(original)
            if mol is None:
                return self.invalid("rdkit_parse_failed")
            mol = rdMolStandardize.Cleanup(mol)
            mol = rdMolStandardize.FragmentParent(mol)
            mol = self.uncharger.uncharge(mol)
            Chem.SanitizeMol(mol)
            standard = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if not standard:
                return self.invalid("empty_standardized_parent")
            inchi_key = Chem.MolToInchiKey(mol)
            connectivity = inchi_key.split("-")[0] if inchi_key else ""
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol is not None and scaffold_mol.GetNumAtoms() > 0:
                scaffold = Chem.MolToSmiles(
                    scaffold_mol, canonical=True, isomericSmiles=True
                )
                generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
                generic = Chem.MolToSmiles(
                    generic_mol, canonical=True, isomericSmiles=False
                )
            else:
                scaffold = ""
                generic = ""
            return {
                "standardized_smiles": standard,
                "connectivity_inchikey": connectivity,
                "bemis_murcko": scaffold,
                "generic_scaffold": generic,
                "standardization_status": "ok",
            }
        except Exception as exc:  # RDKit exceptions vary by release
            return self.invalid(f"standardization_failed:{type(exc).__name__}")

    @staticmethod
    def invalid(reason: str) -> dict[str, object]:
        return {
            "standardized_smiles": "",
            "connectivity_inchikey": "",
            "bemis_murcko": "",
            "generic_scaffold": "",
            "standardization_status": reason,
        }


def standardize_many(smiles: Iterable[str], label: str) -> pd.DataFrame:
    standardizer = MoleculeStandardizer()
    rows: list[dict[str, object]] = []
    for index, value in enumerate(smiles):
        rows.append(standardizer.transform(value))
        if (index + 1) % 25_000 == 0:
            print(f"STANDARDIZE {label}: {index + 1:,}", flush=True)
    return pd.DataFrame(rows)


def retain_standardizable_step2(step2: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Record unstandardizable source rows and return the auditable subset.

    Step-2 row IDs remain the original label-matrix row IDs.  They must not be
    renumbered when a malformed molecule is excluded from structural searches.
    """
    invalid_mask = step2["standardization_status"].ne("ok")
    invalid = step2.loc[
        invalid_mask,
        ["step2_row_id", "original_smiles", "standardization_status"],
    ].copy()
    atomic_csv_gz(out / "invalid_step2_molecules.csv.gz", invalid)
    valid = step2.loc[~invalid_mask].copy()
    if valid.empty:
        raise ValueError("Step-2 contains no molecules that could be standardized")
    if valid["step2_row_id"].duplicated().any():
        raise ValueError("Step-2 row IDs are not unique")
    return valid, invalid


def load_step2(step2_dir: Path, expected_rows: int, expected_assays: int):
    smiles_path = step2_dir / "smiles.txt"
    labels_path = step2_dir / "labels.npy"
    assays_path = step2_dir / "assay_ids.txt"
    for path in (smiles_path, labels_path, assays_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Step-2 artifact missing: {path}")
    smiles = smiles_path.read_text(encoding="utf-8").splitlines()
    assay_ids = assays_path.read_text(encoding="utf-8").splitlines()
    labels = np.load(labels_path, mmap_mode="r")
    observed = (len(smiles), tuple(labels.shape), len(assay_ids), str(labels.dtype))
    required = (expected_rows, (expected_rows, expected_assays), expected_assays)
    if observed[:3] != required:
        raise ValueError(
            "refusing non-final Step-2 data: "
            f"observed smiles={observed[0]:,}, labels={observed[1]}, "
            f"assays={observed[2]:,}, dtype={observed[3]}; expected "
            f"{expected_rows:,} x {expected_assays:,}"
        )
    return smiles, labels, assay_ids, (smiles_path, labels_path, assays_path)


def resolve_step2_dir(
    requested: Path | None, expected_rows: int, expected_assays: int
) -> Path:
    """Find exactly one dataset matching the final matrix dimensions.

    A supplied path is still validated.  Auto-detection is fail-closed: zero or
    multiple matching datasets are errors rather than opportunities to guess.
    """
    if requested is not None:
        load_step2(requested, expected_rows, expected_assays)
        return requested
    candidates: list[Path] = []
    for root in (Path("/mnt/data"), Path("/mnt")):
        if not root.is_dir():
            continue
        for labels_path in root.glob("**/labels.npy"):
            directory = labels_path.parent
            if not (directory / "smiles.txt").is_file() or not (
                directory / "assay_ids.txt"
            ).is_file():
                continue
            try:
                labels = np.load(labels_path, mmap_mode="r")
                if tuple(labels.shape) != (expected_rows, expected_assays):
                    continue
                smiles_count = len(
                    (directory / "smiles.txt").read_text(encoding="utf-8").splitlines()
                )
                assay_count = len(
                    (directory / "assay_ids.txt").read_text(encoding="utf-8").splitlines()
                )
                if smiles_count == expected_rows and assay_count == expected_assays:
                    candidates.append(directory)
            except (OSError, ValueError):
                continue
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(
            "Step-2 auto-detection requires exactly one dataset matching "
            f"{expected_rows:,} x {expected_assays:,}; found {len(unique)}: {unique}"
        )
    return unique[0]


def endpoint_files(prepared_dir: Path) -> list[Path]:
    files = sorted(path for path in prepared_dir.glob("*.csv") if path.is_file())
    if len(files) != 23:
        raise ValueError(
            f"expected exactly 23 prepared endpoint CSVs in {prepared_dir}, found {len(files)}"
        )
    return files


def load_endpoint_rows(prepared_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    records: list[pd.DataFrame] = []
    files = endpoint_files(prepared_dir)
    for path in files:
        frame = pd.read_csv(path)
        required = {"smiles", "y", "cluster_index", "split1", "split2", "split3"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        for split_index, split_name in enumerate(("split1", "split2", "split3"), start=1):
            paper_train = frame[frame[split_name].eq("train")].reset_index()
            groups = paper_train.groupby("cluster_index").indices
            cluster_ids = list(groups)
            rng = np.random.default_rng(7300 + split_index)
            rng.shuffle(cluster_ids)
            target = max(1, round(len(paper_train) * NEURAL_VALIDATION_FRACTION))
            validation_clusters: list[int] = []
            count = 0
            for cluster_id in cluster_ids:
                size = len(groups[cluster_id])
                if count < target or not validation_clusters:
                    validation_clusters.append(cluster_id)
                    count += size
                if count >= target:
                    break
            partition = np.full(len(frame), "train", dtype=object)
            partition[frame[split_name].eq("test").to_numpy()] = "test"
            is_validation = (
                frame[split_name].eq("train")
                & frame["cluster_index"].isin(validation_clusters)
            ).to_numpy()
            partition[is_validation] = "validation"
            if not set(partition) == {"train", "validation", "test"}:
                raise ValueError(
                    f"{path.stem} {split_name}: failed to reconstruct train/validation/test"
                )
            frame[f"{split_name}_neural_partition"] = partition
        keep = [
            "smiles", "y", "cluster_index",
            "split1", "split2", "split3",
            "split1_neural_partition", "split2_neural_partition",
            "split3_neural_partition",
        ]
        selected = frame[keep].copy()
        selected.insert(0, "endpoint_row_index", np.arange(len(selected), dtype=np.int64))
        selected.insert(0, "endpoint_slug", path.stem)
        selected = selected.rename(columns={"smiles": "original_smiles"})
        records.append(selected)
    combined = pd.concat(records, ignore_index=True)
    combined.insert(0, "endpoint_record_id", np.arange(len(combined), dtype=np.int64))
    return combined, files


def build_overlap_tables(
    step2: pd.DataFrame, queries: pd.DataFrame, endpoints: pd.DataFrame, out: Path
) -> dict[str, object]:
    count_rows: list[pd.DataFrame] = []
    representative_rows: list[pd.DataFrame] = []
    full_pair_rows: list[pd.DataFrame] = []
    for audit_name in STRUCTURE_KEYS:
        column = "standardized_smiles" if audit_name == "standardized_exact" else audit_name
        step_valid = step2[step2[column].fillna("").ne("")][["step2_row_id", column]]
        query_valid = queries[queries[column].fillna("").ne("")][["query_id", column]]
        counts = step_valid[column].value_counts()
        qcounts = queries[["query_id", column]].copy()
        qcounts["key_type"] = audit_name
        qcounts["step2_match_count"] = qcounts[column].map(counts).fillna(0).astype(np.int64)
        qcounts = qcounts[["query_id", "key_type", "step2_match_count"]]
        count_rows.append(qcounts)

        matched_keys = set(query_valid[column]).intersection(set(step_valid[column]))
        representatives = (
            step_valid[step_valid[column].isin(matched_keys)]
            .sort_values([column, "step2_row_id"])
            .groupby(column, sort=False)
            .head(10)
        )
        reps = query_valid.merge(representatives, on=column, how="inner")
        reps.insert(1, "key_type", audit_name)
        reps = reps[["query_id", "key_type", "step2_row_id"]]
        representative_rows.append(reps)

        if audit_name in {"standardized_exact", "connectivity_inchikey"}:
            pairs = query_valid.merge(step_valid, on=column, how="inner")
            pairs.insert(1, "key_type", audit_name)
            full_pair_rows.append(pairs[["query_id", "key_type", "step2_row_id"]])

    query_counts = pd.concat(count_rows, ignore_index=True)
    representatives = pd.concat(representative_rows, ignore_index=True)
    exact_pairs = pd.concat(full_pair_rows, ignore_index=True)
    atomic_csv_gz(out / "structure_overlap_query_counts.csv.gz", query_counts)
    atomic_csv_gz(out / "structure_overlap_representatives.csv.gz", representatives)
    atomic_csv_gz(out / "exact_and_connectivity_pairs.csv.gz", exact_pairs)

    expanded = endpoints.merge(query_counts, on="query_id", how="left")
    summary_parts: list[dict[str, object]] = []
    for split in ("all", "split1", "split2", "split3"):
        memberships = ("all",) if split == "all" else ("train", "validation", "test")
        for membership in memberships:
            partition_column = f"{split}_neural_partition"
            selected = (
                expanded if split == "all"
                else expanded[expanded[partition_column].eq(membership)]
            )
            for (endpoint, key_type), group in selected.groupby(
                ["endpoint_slug", "key_type"], sort=True
            ):
                summary_parts.append({
                    "endpoint_slug": endpoint,
                    "split_name": split,
                    "split_membership": membership,
                    "key_type": key_type,
                    "n_endpoint_rows": int(len(group)),
                    "n_rows_with_overlap": int(group["step2_match_count"].gt(0).sum()),
                    "fraction_rows_with_overlap": float(group["step2_match_count"].gt(0).mean()),
                    "total_step2_matches": int(group["step2_match_count"].sum()),
                })
    summary = pd.DataFrame(summary_parts)
    summary.to_csv(out / "structure_overlap_endpoint_split_summary.csv", index=False)
    return {
        "query_key_counts_file": "structure_overlap_query_counts.csv.gz",
        "representatives_file": "structure_overlap_representatives.csv.gz",
        "full_pairs_file": "exact_and_connectivity_pairs.csv.gz",
        "full_pair_policy": (
            "All standardized-exact and connectivity-InChIKey pairs are retained. "
            "Bemis-Murcko and generic-scaffold overlaps retain exact per-query counts "
            "and the first 10 deterministic Step-2 representatives per key; their full "
            "Cartesian pair expansion is intentionally not materialized."
        ),
    }


def find_chembl_db(explicit: Path | None) -> Path | None:
    if explicit and explicit.is_file():
        return explicit
    candidates = (
        Path("/mnt/data/chembl/chembl_36_sqlite/chembl_36.db"),
        Path("/mnt/data/chembl/chembl_35_sqlite/chembl_35.db"),
        Path("/mnt/data/chembl/chembl_34_sqlite/chembl_34.db"),
    )
    return next((path for path in candidates if path.is_file()), None)


def export_assay_metadata(
    assay_ids: list[str], db_path: Path | None, out_path: Path
) -> dict[str, object]:
    columns = [
        "assay_id", "assay_chembl_id", "description", "assay_type",
        "assay_organism", "assay_tissue", "assay_cell_type",
        "assay_subcellular_fraction", "target_chembl_id", "target_pref_name",
        "target_type", "target_organism",
    ]
    if db_path is None:
        pd.DataFrame({"assay_id": assay_ids}).to_csv(out_path, index=False)
        return {
            "status": "incomplete_missing_chembl_sqlite",
            "database": None,
            "matched_assays": 0,
            "expected_assays": len(assay_ids),
            "endpoint_equivalence_adjudicated": False,
        }
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows: list[tuple] = []
        integer_ids = [int(value) for value in assay_ids]
        for start in range(0, len(integer_ids), 500):
            batch = integer_ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            sql = f"""
                SELECT a.assay_id, a.chembl_id, a.description, a.assay_type,
                       a.assay_organism, a.assay_tissue, a.assay_cell_type,
                       a.assay_subcellular_fraction, td.chembl_id, td.pref_name,
                       td.target_type, td.organism
                FROM assays a
                LEFT JOIN target_dictionary td ON a.tid = td.tid
                WHERE a.assay_id IN ({marks})
            """
            rows.extend(con.execute(sql, batch).fetchall())
        frame = pd.DataFrame(rows, columns=columns)
        frame = pd.DataFrame({"assay_id": [int(x) for x in assay_ids]}).merge(
            frame, on="assay_id", how="left"
        )
        frame.to_csv(out_path, index=False)
        matched = int(frame["assay_chembl_id"].notna().sum())
        return {
            "status": "metadata_joined_endpoint_equivalence_not_adjudicated",
            "database": str(db_path),
            "database_sha256": sha256_file(db_path),
            "matched_assays": matched,
            "expected_assays": len(assay_ids),
            "endpoint_equivalence_adjudicated": False,
        }
    finally:
        con.close()


def command_prepare(args) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    step2_dir = resolve_step2_dir(
        args.step2_dir, args.expected_rows, args.expected_assays
    )
    smiles, labels, assay_ids, step_paths = load_step2(
        step2_dir, args.expected_rows, args.expected_assays
    )
    prepared_files = endpoint_files(args.prepared_dir)
    prepared_manifest = args.prepared_dir / "manifest.json"
    if not prepared_manifest.is_file():
        raise FileNotFoundError(f"prepared benchmark manifest missing: {prepared_manifest}")
    manifest_payload = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    expected_endpoint_names = {path.stem for path in prepared_files}
    if set(manifest_payload.get("endpoints", {})) != expected_endpoint_names:
        raise ValueError("prepared manifest endpoint set does not match the 23 CSV files")
    for path in prepared_files:
        expected_sha = manifest_payload["endpoints"][path.stem].get("prepared_sha256")
        if not expected_sha or sha256_file(path) != expected_sha:
            raise ValueError(f"prepared CSV hash mismatch: {path}")

    source_hashes = {str(path): sha256_file(path) for path in step_paths}
    source_hashes.update({str(path): sha256_file(path) for path in prepared_files})
    source_hashes[str(prepared_manifest)] = sha256_file(prepared_manifest)
    prepare_contract = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "expected_rows": args.expected_rows,
        "expected_assays": args.expected_assays,
        "standardization": [
            "MolFromSmiles", "Cleanup", "FragmentParent", "Uncharger", "SanitizeMol"
        ],
        "fingerprint": {"type": "Morgan", "radius": FP_RADIUS, "fpSize": FP_SIZE},
        "thresholds": list(THRESHOLDS),
        "source_sha256": source_hashes,
    }
    prepare_contract_sha = stable_hash(prepare_contract)
    if (out / "SETUP_INPUTS_READY").is_file():
        existing_path = out / "prepare_manifest.json"
        if not existing_path.is_file():
            raise RuntimeError("SETUP_INPUTS_READY exists without prepare_manifest.json")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        required_outputs = (
            "step2_structures.csv.gz", "endpoint_rows.csv.gz", "query_structures.csv.gz",
            "step2_standardized.smi", "invalid_step2_molecules.csv.gz",
            "structure_overlap_query_counts.csv.gz",
            "structure_overlap_representatives.csv.gz",
            "exact_and_connectivity_pairs.csv.gz",
            "structure_overlap_endpoint_split_summary.csv", "assay_metadata.csv",
        )
        observed_artifacts = existing.get("prepared_artifact_sha256", {})
        artifacts_current = all(
            (out / name).is_file()
            and observed_artifacts.get(name) == sha256_file(out / name)
            for name in required_outputs
        )
        if existing.get("prepare_contract_sha256") != prepare_contract_sha or not artifacts_current:
            raise RuntimeError(
                "stale or incomplete prepared leakage-audit state; use a new OUTPUT_DIR"
            )
        print("setup inputs already complete and provenance-current", flush=True)
        return

    endpoint_rows, _ = load_endpoint_rows(args.prepared_dir)

    step_std = standardize_many(smiles, "Step-2")
    step2 = pd.DataFrame({
        "step2_row_id": np.arange(len(smiles), dtype=np.int64),
        "original_smiles": smiles,
    })
    step2 = pd.concat([step2, step_std], axis=1)
    valid_step2, invalid_step2 = retain_standardizable_step2(step2, out)

    endpoint_std = standardize_many(endpoint_rows["original_smiles"], "OpenADMET")
    endpoints = pd.concat([endpoint_rows, endpoint_std], axis=1)
    invalid_endpoints = endpoints["standardization_status"].ne("ok")
    if invalid_endpoints.any():
        bad = endpoints.loc[
            invalid_endpoints,
            ["endpoint_slug", "endpoint_row_index", "original_smiles", "standardization_status"],
        ]
        atomic_csv_gz(out / "invalid_endpoint_molecules.csv.gz", bad)
        raise ValueError(
            f"OpenADMET contains {int(invalid_endpoints.sum())} molecules that could not be standardized"
        )

    unique_structures = endpoints[
        ["standardized_smiles", "connectivity_inchikey", "bemis_murcko", "generic_scaffold"]
    ].drop_duplicates("standardized_smiles").reset_index(drop=True)
    unique_structures.insert(0, "query_id", np.arange(len(unique_structures), dtype=np.int64))
    endpoints = endpoints.merge(
        unique_structures[["query_id", "standardized_smiles"]],
        on="standardized_smiles",
        how="left",
        validate="many_to_one",
    )
    endpoints["query_id"] = endpoints["query_id"].astype(np.int64)

    atomic_csv_gz(out / "step2_structures.csv.gz", step2)
    atomic_csv_gz(out / "endpoint_rows.csv.gz", endpoints)
    atomic_csv_gz(out / "query_structures.csv.gz", unique_structures)
    with (out / "step2_standardized.smi").open("w", encoding="utf-8", newline="\n") as handle:
        for row in valid_step2.itertuples(index=False):
            handle.write(f"{row.standardized_smiles}\t{int(row.step2_row_id) + 1}\n")

    overlap_policy = build_overlap_tables(valid_step2, unique_structures, endpoints, out)
    semantic = export_assay_metadata(
        assay_ids, find_chembl_db(args.chembl_db), out / "assay_metadata.csv"
    )

    prepare_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepared_for_fpsim2",
        "created_utc": utc_now(),
        "step2": {
            "directory": str(step2_dir),
            "rows": len(smiles),
            "standardized_rows": len(valid_step2),
            "excluded_unstandardizable_rows": len(invalid_step2),
            "invalid_rows_file": "invalid_step2_molecules.csv.gz",
            "assays": len(assay_ids),
            "labels_shape": list(labels.shape),
            "labels_dtype": str(labels.dtype),
        },
        "openadmet": {
            "prepared_directory": str(args.prepared_dir),
            "endpoints": 23,
            "endpoint_rows": len(endpoints),
            "unique_standardized_queries": len(unique_structures),
            "neural_inner_validation": {
                "fraction": NEURAL_VALIDATION_FRACTION,
                "seeds": {"split1": 7301, "split2": 7302, "split3": 7303},
                "unit": "whole cluster_index groups from each outer-train partition",
            },
        },
        "standardization": {
            "library": "RDKit MolStandardize",
            "steps": ["MolFromSmiles", "Cleanup", "FragmentParent", "Uncharger", "SanitizeMol"],
            "standardized_exact": "canonical isomeric SMILES of standardized parent",
            "connectivity_inchikey": "first block of standardized-parent InChIKey",
            "bemis_murcko": "RDKit GetScaffoldForMol; empty/acyclic excluded",
            "generic_scaffold": "RDKit MakeScaffoldGeneric on nonempty Bemis-Murcko scaffold",
            "rdkit_version": rdBase.rdkitVersion,
        },
        "fingerprint": {
            "implementation": "FPSim2 exact CPU search",
            "type": "Morgan/ECFP4",
            "radius": FP_RADIUS,
            "fpSize": FP_SIZE,
            "metric": "Tanimoto",
            "thresholds": list(THRESHOLDS),
            "nearest_neighbor_search_threshold": 0.0,
            "approximate_search": False,
        },
        "overlap_materialization": overlap_policy,
        "semantic_mapping": semantic,
        "source_sha256": source_hashes,
        "prepare_contract": prepare_contract,
        "prepare_contract_sha256": prepare_contract_sha,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    prepare_manifest["prepared_artifact_sha256"] = {
        path.name: sha256_file(path)
        for path in out.iterdir()
        if path.is_file() and path.name not in {"prepare_manifest.json", "SETUP_INPUTS_READY"}
        and not path.name.endswith((".tmp", ".lock"))
    }
    atomic_json(out / "prepare_manifest.json", prepare_manifest)
    atomic_text(out / "SETUP_INPUTS_READY", utc_now() + "\n")
    print(json.dumps(prepare_manifest["openadmet"], indent=2), flush=True)


def command_mark_index(args) -> None:
    from FPSim2 import FPSim2Engine
    import FPSim2

    db = args.output_dir / "step2_ecfp4_2048.h5"
    if not db.is_file() or db.stat().st_size < 1024:
        raise FileNotFoundError(f"FPSim2 database was not created: {db}")
    engine = FPSim2Engine(str(db), in_memory_fps=False)
    if engine.fp_type != "Morgan":
        raise ValueError(f"unexpected FPSim2 fingerprint type: {engine.fp_type}")
    observed = dict(engine.fp_params)
    radius = int(observed.get("radius", -1))
    size = int(observed.get("fpSize", observed.get("nBits", -1)))
    if radius != FP_RADIUS or size != FP_SIZE:
        raise ValueError(f"unexpected FPSim2 parameters: {observed}")
    manifest_path = args.output_dir / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage"] = "fpsim2_index_ready"
    manifest["fingerprint"].update({
        "fpsim2_version": getattr(FPSim2, "__version__", "unknown"),
        "index_file": str(db),
        "index_sha256": sha256_file(db),
        "index_fp_params": observed,
    })
    manifest["index_contract_sha256"] = stable_hash({
        "prepare_contract_sha256": manifest["prepare_contract_sha256"],
        "index_sha256": manifest["fingerprint"]["index_sha256"],
        "index_fp_params": observed,
    })
    atomic_json(manifest_path, manifest)
    atomic_text(args.output_dir / "SETUP_COMPLETE", utc_now() + "\n")
    print(str(engine), flush=True)


_ENGINE = None


def _init_worker(db_path: str) -> None:
    global _ENGINE
    from FPSim2 import FPSim2Engine
    _ENGINE = FPSim2Engine(db_path, in_memory_fps=True)


def _search_one(record: tuple[int, str]) -> dict[str, object]:
    query_id, smiles = record
    # threshold=0.0 gives a true exact nearest neighbour, including queries
    # whose best Step-2 match lies below the audit's lowest reporting cutoff.
    results = _ENGINE.top_k(smiles, k=1, threshold=0.0, n_workers=1)
    if len(results) == 0:
        raise RuntimeError(f"FPSim2 returned no nearest neighbour for query {query_id}")
    ordered = sorted(
        ((int(value["mol_id"]), float(value["coeff"])) for value in results),
        key=lambda item: (-item[1], item[0]),
    )
    mol_id, coeff = ordered[0]
    row: dict[str, object] = {
        "query_id": int(query_id),
        "nearest_step2_row_id": mol_id - 1,
        "fpsim2_mol_id": mol_id,
        "max_tanimoto": coeff,
    }
    for threshold in THRESHOLDS:
        key = f"hit_ge_{threshold:.2f}".replace(".", "_")
        row[key] = bool(coeff + 1e-7 >= threshold)
    return row


def command_search(args) -> None:
    from multiprocessing import get_context

    out = args.output_dir
    if not (out / "SETUP_COMPLETE").is_file():
        raise FileNotFoundError("SETUP_COMPLETE is missing")
    shard_path = out / f"nearest_neighbors_shard_{args.shard_id:03d}.csv.gz"
    done_path = out / f"SEARCH_SHARD_{args.shard_id:03d}_COMPLETE"
    metadata_path = out / f"nearest_neighbors_shard_{args.shard_id:03d}.json"
    prepare = json.loads((out / "prepare_manifest.json").read_text(encoding="utf-8"))
    db_path_obj = out / "step2_ecfp4_2048.h5"
    if sha256_file(db_path_obj) != prepare.get("fingerprint", {}).get("index_sha256"):
        raise RuntimeError("FPSim2 index hash differs from the marked index")
    queries = pd.read_csv(out / "query_structures.csv.gz")
    subset = queries[queries["query_id"].mod(args.num_shards).eq(args.shard_id)]
    records = list(zip(subset["query_id"].astype(int), subset["standardized_smiles"].astype(str)))
    search_contract = {
        "prepare_contract_sha256": prepare["prepare_contract_sha256"],
        "index_contract_sha256": prepare["index_contract_sha256"],
        "query_table_sha256": sha256_file(out / "query_structures.csv.gz"),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "queries": len(records),
        "nearest_neighbor_threshold": 0.0,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    search_contract_sha = stable_hash(search_contract)
    if shard_path.is_file() and done_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("search_contract_sha256") == search_contract_sha
            and existing.get("csv_sha256") == sha256_file(shard_path)
        ):
            print(f"search shard {args.shard_id} already complete and provenance-current", flush=True)
            return
        raise RuntimeError(f"stale search shard {args.shard_id}; use a new OUTPUT_DIR")
    print(
        f"SEARCH shard={args.shard_id}/{args.num_shards} queries={len(records):,} workers={args.workers}",
        flush=True,
    )
    db_path = str(db_path_obj)
    context = get_context("spawn")
    rows: list[dict[str, object]] = []
    with context.Pool(args.workers, initializer=_init_worker, initargs=(db_path,)) as pool:
        for count, row in enumerate(pool.imap_unordered(_search_one, records, chunksize=8), start=1):
            rows.append(row)
            if count % 250 == 0:
                print(f"SEARCH shard={args.shard_id} completed={count:,}/{len(records):,}", flush=True)
    result = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    if len(result) != len(records) or result["query_id"].nunique() != len(records):
        raise RuntimeError("search shard lost or duplicated query IDs")
    atomic_csv_gz(shard_path, result)
    atomic_json(
        metadata_path,
        {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "queries": len(result),
            "hits_ge_0_70": int(result["hit_ge_0_70"].sum()),
            "nearest_neighbor_threshold": 0.0,
            "search_contract": search_contract,
            "search_contract_sha256": search_contract_sha,
            "csv_sha256": sha256_file(shard_path),
            "completed_utc": utc_now(),
        },
    )
    atomic_text(done_path, utc_now() + "\n")


def brute_force_validation(
    queries: pd.DataFrame, step2: pd.DataFrame, nearest: pd.DataFrame, out: Path
) -> dict[str, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    if "standardization_status" in step2.columns:
        valid_step2 = step2[step2["standardization_status"].eq("ok")].copy()
    else:
        valid_step2 = step2[step2["standardized_smiles"].fillna("").ne("")].copy()
    if valid_step2.empty:
        raise ValueError("no standardized Step-2 molecules are available for validation")
    step2_row_ids = valid_step2["step2_row_id"].astype(int).to_numpy()
    row_id_to_position = {row_id: position for position, row_id in enumerate(step2_row_ids)}
    if len(row_id_to_position) != len(step2_row_ids):
        raise ValueError("Step-2 row IDs are not unique")
    step_fps = []
    for position, smiles in enumerate(valid_step2["standardized_smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            row_id = int(step2_row_ids[position])
            raise ValueError(f"standardized Step-2 molecule no longer parses at row {row_id}")
        step_fps.append(generator.GetFingerprint(mol))
        if (position + 1) % 50_000 == 0:
            print(f"BRUTE-FORCE fingerprinted Step-2 {position + 1:,}", flush=True)

    sample_size = min(32, len(queries))
    sample_ids = np.linspace(0, len(queries) - 1, sample_size, dtype=np.int64)
    by_query = nearest.set_index("query_id")
    validation_rows: list[dict[str, object]] = []
    failures = 0
    for query_id in sample_ids:
        smiles = str(queries.iloc[int(query_id)]["standardized_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        query_fp = generator.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, step_fps)
        exact_max = float(max(similarities))
        exact_argmax = int(step2_row_ids[int(np.argmax(similarities))])
        reported = by_query.loc[int(query_id)]
        hit = bool(reported["hit_ge_0_70"])
        reported_value = float(reported["max_tanimoto"])
        reported_id = int(reported["nearest_step2_row_id"])
        reported_position = row_id_to_position.get(reported_id)
        reported_similarity = (
            float(similarities[reported_position])
            if reported_position is not None else float("nan")
        )
        passed = (
            abs(reported_value - exact_max) <= 1e-6
            and abs(reported_similarity - exact_max) <= 1e-6
            and hit == (exact_max + 1e-7 >= THRESHOLDS[0])
        )
        failures += int(not passed)
        validation_rows.append({
            "query_id": int(query_id),
            "exact_bruteforce_max_tanimoto": exact_max,
            "exact_bruteforce_argmax_step2_row_id": exact_argmax,
            "fpsim2_hit_ge_0_70": hit,
            "fpsim2_reported_max_tanimoto": reported_value,
            "fpsim2_step2_row_id": reported_id,
            "fpsim2_row_bruteforce_tanimoto": reported_similarity,
            "pass": passed,
        })
    frame = pd.DataFrame(validation_rows)
    frame.to_csv(out / "fpsim2_bruteforce_validation.csv", index=False)
    if failures:
        raise RuntimeError(f"FPSim2 brute-force validation failed for {failures} queries")
    return {
        "sample_queries": len(frame),
        "full_step2_comparisons_per_query": len(valid_step2),
        "excluded_unstandardizable_step2_rows": len(step2) - len(valid_step2),
        "failures": failures,
        "method": "RDKit BulkTanimotoSimilarity over all Step-2 ECFP4 fingerprints",
    }


def attach_labels(
    args, nearest: pd.DataFrame, out: Path
) -> dict[str, object]:
    step2_dir = resolve_step2_dir(
        args.step2_dir, args.expected_rows, args.expected_assays
    )
    labels = np.load(step2_dir / "labels.npy", mmap_mode="r")
    assay_ids = (step2_dir / "assay_ids.txt").read_text(encoding="utf-8").splitlines()
    exact_pairs = pd.read_csv(out / "exact_and_connectivity_pairs.csv.gz")
    provenance: dict[int, set[str]] = defaultdict(set)
    inconsistent_hits = nearest[
        nearest["hit_ge_0_70"].astype(bool)
        & nearest["max_tanimoto"].lt(THRESHOLDS[0] - 1e-7)
    ]
    if not inconsistent_hits.empty:
        raise RuntimeError("nearest-neighbour threshold flags are internally inconsistent")
    for row in exact_pairs.itertuples(index=False):
        provenance[int(row.step2_row_id)].add(str(row.key_type))
    for row_id in nearest.loc[
        nearest["nearest_step2_row_id"].ge(0)
        & nearest["hit_ge_0_70"].astype(bool),
        "nearest_step2_row_id",
    ].astype(int):
        provenance[int(row_id)].add("nearest_ecfp4_ge_0.70")

    metadata = pd.read_csv(out / "assay_metadata.csv")
    metadata["assay_id"] = metadata["assay_id"].astype(str)
    meta_by_id = metadata.set_index("assay_id").to_dict(orient="index")
    output = out / "matched_step2_labels_with_assay_metadata.csv.gz"
    tmp = output.with_suffix(output.suffix + ".tmp")
    metadata_columns = [column for column in metadata.columns if column != "assay_id"]
    n_rows = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step2_row_id", "match_provenance", "assay_id", "label"]
            + metadata_columns,
        )
        writer.writeheader()
        for row_id in sorted(provenance):
            row_labels = np.asarray(labels[row_id])
            for assay_index in np.flatnonzero(~np.isnan(row_labels)):
                assay_id = str(assay_ids[int(assay_index)])
                record = {
                    "step2_row_id": row_id,
                    "match_provenance": ";".join(sorted(provenance[row_id])),
                    "assay_id": assay_id,
                    "label": float(row_labels[int(assay_index)]),
                }
                record.update(meta_by_id.get(assay_id, {}))
                writer.writerow(record)
                n_rows += 1
    os.replace(tmp, output)
    return {
        "matched_step2_molecules": len(provenance),
        "matched_nonmissing_label_records": n_rows,
        "output": output.name,
        "sha256": sha256_file(output),
    }


def endpoint_nn_summary(endpoint_nn: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    threshold_columns = ["hit_ge_0_70", "hit_ge_0_80", "hit_ge_0_90", "hit_ge_0_95"]
    for split in ("all", "split1", "split2", "split3"):
        memberships = ("all",) if split == "all" else ("train", "validation", "test")
        for membership in memberships:
            partition_column = f"{split}_neural_partition"
            selected = (
                endpoint_nn if split == "all"
                else endpoint_nn[endpoint_nn[partition_column].eq(membership)]
            )
            for endpoint, group in selected.groupby("endpoint_slug", sort=True):
                record: dict[str, object] = {
                    "endpoint_slug": endpoint,
                    "split_name": split,
                    "split_membership": membership,
                    "n_endpoint_rows": len(group),
                }
                for column in threshold_columns:
                    suffix = column.removeprefix("hit_ge_")
                    count = int(group[column].astype(bool).sum())
                    record[f"n_rows_ge_{suffix}"] = count
                    record[f"fraction_rows_ge_{suffix}"] = count / len(group) if len(group) else np.nan
                rows.append(record)
    return pd.DataFrame(rows)


def command_finalize(args) -> None:
    out = args.output_dir
    if (out / "AUDIT_COMPLETE").is_file() or (out / "STRUCTURE_AUDIT_COMPLETE").is_file():
        print("audit already finalized", flush=True)
        return
    missing = [
        shard for shard in range(args.num_shards)
        if not (out / f"SEARCH_SHARD_{shard:03d}_COMPLETE").is_file()
    ]
    if missing:
        print(f"finalize deferred; incomplete shards: {missing}", flush=True)
        return
    prepare = json.loads((out / "prepare_manifest.json").read_text(encoding="utf-8"))
    for stored_path, expected_sha in prepare.get("source_sha256", {}).items():
        current_path = Path(stored_path)
        if not current_path.is_file() or sha256_file(current_path) != expected_sha:
            raise RuntimeError(f"source changed since prepare: {stored_path}")
    if sha256_file(out / "step2_ecfp4_2048.h5") != prepare["fingerprint"]["index_sha256"]:
        raise RuntimeError("FPSim2 index changed after setup")
    shard_frames = []
    for shard in range(args.num_shards):
        shard_path = out / f"nearest_neighbors_shard_{shard:03d}.csv.gz"
        shard_meta_path = out / f"nearest_neighbors_shard_{shard:03d}.json"
        shard_meta = json.loads(shard_meta_path.read_text(encoding="utf-8"))
        if shard_meta.get("csv_sha256") != sha256_file(shard_path):
            raise RuntimeError(f"search shard {shard} CSV hash mismatch")
        contract = shard_meta.get("search_contract", {})
        if (
            stable_hash(contract) != shard_meta.get("search_contract_sha256")
            or contract.get("prepare_contract_sha256") != prepare["prepare_contract_sha256"]
            or contract.get("index_contract_sha256") != prepare["index_contract_sha256"]
            or contract.get("num_shards") != args.num_shards
            or contract.get("shard_id") != shard
        ):
            raise RuntimeError(f"search shard {shard} provenance mismatch")
        shard_frames.append(pd.read_csv(shard_path))
    nearest = pd.concat(shard_frames, ignore_index=True).sort_values("query_id")
    queries = pd.read_csv(out / "query_structures.csv.gz")
    if len(nearest) != len(queries) or nearest["query_id"].nunique() != len(queries):
        raise RuntimeError("aggregated search results do not cover every unique query exactly once")
    atomic_csv_gz(out / "nearest_neighbors_all_queries.csv.gz", nearest)

    endpoints = pd.read_csv(out / "endpoint_rows.csv.gz")
    endpoint_nn = endpoints.merge(nearest, on="query_id", how="left", validate="many_to_one")
    if endpoint_nn["hit_ge_0_70"].isna().any():
        raise RuntimeError("one or more endpoint rows lack nearest-neighbour results")
    atomic_csv_gz(out / "endpoint_rows_with_nearest_neighbors.csv.gz", endpoint_nn)
    summary = endpoint_nn_summary(endpoint_nn)
    summary.to_csv(out / "nearest_neighbor_endpoint_split_summary.csv", index=False)

    step2 = pd.read_csv(out / "step2_structures.csv.gz")
    brute = brute_force_validation(queries, step2, nearest, out)
    label_report = attach_labels(args, nearest, out)

    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and not path.name.endswith((".tmp", ".lock")):
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    semantics_complete = bool(
        prepare.get("semantic_mapping", {}).get("endpoint_equivalence_adjudicated")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete" if semantics_complete
            else "structure_audit_complete_endpoint_equivalence_pending"
        ),
        "completed_utc": utc_now(),
        "prepare": prepare,
        "search": {
            "num_shards": args.num_shards,
            "unique_queries": len(nearest),
            "exact_fpsim2": True,
            "approximate_search": False,
            "threshold_counts": {
                column: int(nearest[column].astype(bool).sum())
                for column in ("hit_ge_0_70", "hit_ge_0_80", "hit_ge_0_90", "hit_ge_0_95")
            },
        },
        "brute_force_validation": brute,
        "matched_labels": label_report,
        "semantic_mapping_status": prepare["semantic_mapping"],
        "artifacts": artifacts,
    }
    atomic_json(out / "audit_manifest.json", manifest)
    completion_marker = "AUDIT_COMPLETE" if semantics_complete else "STRUCTURE_AUDIT_COMPLETE"
    atomic_text(out / completion_marker, utc_now() + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "queries": len(nearest),
        "brute_force_failures": brute["failures"],
        "label_records": label_report["matched_nonmissing_label_records"],
    }, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--step2-dir", type=Path)
    common.add_argument("--expected-rows", type=int, default=EXPECTED_STEP2_ROWS)
    common.add_argument("--expected-assays", type=int, default=EXPECTED_ASSAYS)

    prepare = sub.add_parser("prepare", parents=[common])
    prepare.add_argument(
        "--prepared-dir", type=Path, default=Path("/mnt/data/moljepa_benchmarks/prepared")
    )
    prepare.add_argument("--chembl-db", type=Path)
    prepare.set_defaults(func=command_prepare)

    mark = sub.add_parser("mark-index", parents=[common])
    mark.set_defaults(func=command_mark_index)

    search = sub.add_parser("search", parents=[common])
    search.add_argument("--shard-id", type=int, required=True)
    search.add_argument("--num-shards", type=int, required=True)
    search.add_argument("--workers", type=int, default=8)
    search.set_defaults(func=command_search)

    finalize = sub.add_parser("finalize", parents=[common])
    finalize.add_argument("--num-shards", type=int, required=True)
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
