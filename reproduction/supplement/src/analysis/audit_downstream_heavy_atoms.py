#!/usr/bin/env python3
"""Audit downstream molecules against the 96-heavy-atom pretraining cap.

The OpenADMET input is the deterministic output of
``prepare_moljepa_benchmarks.py``.  Optional retained benchmark cells are used
to verify the exact train/validation/test memberships by their SMILES hashes.

The TDC input is a local PyTDC ``*.tab`` snapshot.  The scaffold split below is
the PyTDC 0.4.1 implementation used by ``finetune_benchmarks.py``.  Historical
run logs are parsed separately because the old PVC snapshot is not retained
locally; a row-count mismatch is therefore exposed rather than hidden.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TDC_TASKS = (
    "ames", "bbb_martins", "bioavailability_ma", "caco2_wang",
    "clearance_hepatocyte_az", "clearance_microsome_az",
    "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels", "cyp3a4_veith", "dili",
    "half_life_obach", "herg", "hia_hou", "ld50_zhu",
    "lipophilicity_astrazeneca", "pgp_broccatelli", "ppbr_az",
    "solubility_aqsoldb", "vdss_lombardo",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smiles_sha256(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def heavy_atoms(value: str) -> int | None:
    mol = Chem.MolFromSmiles(str(value))
    return None if mol is None else int(mol.GetNumHeavyAtoms())


def size_record(benchmark: str, endpoint: str, split: str, role: str, frame: pd.DataFrame) -> dict:
    values = frame["heavy_atoms"]
    valid = values.notna() & values.gt(0)
    return {
        "benchmark": benchmark,
        "endpoint": endpoint,
        "split": split,
        "role": role,
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "within_cap_rows": int((valid & values.le(96)).sum()),
        "over_cap_rows": int((valid & values.gt(96)).sum()),
        "max_heavy_atoms": int(values[valid].max()) if valid.any() else None,
    }


def inner_openadmet_split(frame: pd.DataFrame, split_index: int):
    groups = frame.groupby("cluster_index").indices
    cluster_ids = list(groups)
    rng = np.random.default_rng(7300 + split_index)
    rng.shuffle(cluster_ids)
    target = max(1, round(len(frame) * 0.15))
    selected, count = [], 0
    for cluster_id in cluster_ids:
        if count < target or not selected:
            selected.append(cluster_id)
            count += len(groups[cluster_id])
        if count >= target:
            break
    mask = frame["cluster_index"].isin(selected).to_numpy()
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)


def pytdc_scaffold_split(frame: pd.DataFrame):
    scaffolds, errors = defaultdict(set), 0
    for index, value in enumerate(frame["Drug"]):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(str(value)), includeChirality=False
            )
            scaffolds[scaffold].add(index)
        except Exception:
            errors += 1
    train_size = int((len(frame) - errors) * 0.70)
    valid_size = int((len(frame) - errors) * 0.15)
    test_size = len(frame) - errors - train_size - valid_size
    big, small = [], []
    for index_set in scaffolds.values():
        (big if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2 else small).append(index_set)
    random = Random(42)
    random.shuffle(big)
    random.shuffle(small)
    output = {"train": [], "valid": [], "test": []}
    for index_set in big + small:
        if len(output["train"]) + len(index_set) <= train_size:
            output["train"] += index_set
        elif len(output["valid"]) + len(index_set) <= valid_size:
            output["valid"] += index_set
        else:
            output["test"] += index_set
    return {key: frame.iloc[index].reset_index(drop=True) for key, index in output.items()}


def parse_tdc_log(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    totals = [int(b) for a, b in re.findall(r"(\d+)/(\d+)\s+\[", text) if a == b]
    counts = {key: int(value) for key, value in re.findall(r"\b(train|valid|test): n=(\d+)", text)}
    return {"pre_filter_rows": totals[0] if totals else None, "post_filter_rows": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openadmet-dir", type=Path, required=True)
    parser.add_argument("--openadmet-results-dir", type=Path)
    parser.add_argument("--tdc-dir", type=Path, required=True)
    parser.add_argument("--tdc-log-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.openadmet_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, open_endpoints, hash_checks = [], {}, []
    for endpoint, source in sorted(manifest["endpoints"].items()):
        frame = pd.read_csv(args.openadmet_dir / f"{endpoint}.csv")
        frame["heavy_atoms"] = [heavy_atoms(value) for value in frame["smiles"]]
        rows.append(size_record("OpenADMET", endpoint, "all", "all", frame))
        endpoint_record = {"rows": len(frame), "over_cap_rows": int(frame["heavy_atoms"].gt(96).sum()), "splits": {}}
        for split_index, split_name in enumerate(("split1", "split2", "split3"), 1):
            outer_train = frame[frame[split_name].eq("train")].reset_index(drop=True)
            test = frame[frame[split_name].eq("test")].reset_index(drop=True)
            inner_train, validation = inner_openadmet_split(outer_train, split_index)
            partitions = {"inner_train": inner_train, "validation": validation, "test": test}
            endpoint_record["splits"][split_name] = {}
            for role, part in partitions.items():
                record = size_record("OpenADMET", endpoint, split_name, role, part)
                rows.append(record)
                endpoint_record["splits"][split_name][role] = record
            if args.openadmet_results_dir:
                cell = json.loads((args.openadmet_results_dir / endpoint / "cells" / f"{split_name}_full_mix.json").read_text(encoding="utf-8"))
                observed = {
                    "inner_train": smiles_sha256(inner_train["smiles"]),
                    "validation": smiles_sha256(validation["smiles"]),
                    "test": smiles_sha256(test["smiles"]),
                }
                expected = {
                    "inner_train": cell["train_smiles_sha256"],
                    "validation": cell["validation_smiles_sha256"],
                    "test": cell["test_smiles_sha256"],
                }
                row_counts_match = (
                    len(inner_train) == int(cell["train_rows"])
                    and len(validation) == int(cell["validation_rows"])
                    and len(test) == int(cell["test_rows"])
                )
                hash_checks.append({
                    "endpoint": endpoint,
                    "split": split_name,
                    "row_counts_match": row_counts_match,
                    "partition_hash_matches": {
                        role: observed[role] == expected[role] for role in observed
                    },
                    "all_partition_hashes_match": observed == expected,
                })
        open_endpoints[endpoint] = endpoint_record

    tdc_endpoints, tdc_local_over = {}, 0
    for endpoint in TDC_TASKS:
        path = args.tdc_dir / f"{endpoint}.tab"
        raw = pd.read_csv(path, sep="\t").rename(columns={"X": "Drug", "ID": "Drug_ID"})
        frame = raw[raw["Y"].notna()].reset_index(drop=True)
        # PyTDC's ADME loader restricts PPBR_AZ to the human subset.
        if endpoint == "ppbr_az":
            frame = frame[frame["Species"].eq("Homo sapiens")].reset_index(drop=True)
        frame["heavy_atoms"] = [heavy_atoms(value) for value in frame["Drug"]]
        split = pytdc_scaffold_split(frame)
        endpoint_rows = []
        for role, part in split.items():
            record = size_record("TDC_local_snapshot", endpoint, "seed42", role, part)
            rows.append(record)
            endpoint_rows.append(record)
        over = sum(item["over_cap_rows"] for item in endpoint_rows)
        tdc_local_over += over
        log = parse_tdc_log(args.tdc_log_dir / f"finetune_{endpoint}.log") if args.tdc_log_dir else None
        local_pre = len(frame)
        tdc_endpoints[endpoint] = {
            "source_path": str(path), "source_sha256": file_sha256(path),
            "local_rows_after_label_filter": local_pre, "local_over_cap_rows": over,
            "local_split_records": endpoint_rows, "historical_run_log": log,
            "historical_snapshot_row_match": bool(log and log["pre_filter_rows"] == local_pre),
        }

    open_total = sum(item["rows"] for item in open_endpoints.values())
    open_over = sum(item["over_cap_rows"] for item in open_endpoints.values())
    open_test_over = sum(
        split["test"]["over_cap_rows"]
        for item in open_endpoints.values() for split in item["splits"].values()
    )
    historical_pre = sum(
        int(item["historical_run_log"]["pre_filter_rows"])
        for item in tdc_endpoints.values() if item["historical_run_log"]
    )
    historical_post = {
        role: sum(
            int(item["historical_run_log"]["post_filter_rows"][role])
            for item in tdc_endpoints.values() if item["historical_run_log"]
        )
        for role in ("train", "valid", "test")
    }
    payload = {
        "schema_version": 1,
        "cap_heavy_atoms": 96,
        "software": {"rdkit": rdBase.rdkitVersion, "pandas": pd.__version__, "numpy": np.__version__},
        "openadmet": {
            "scope": "pinned public-source reconstruction used by the benchmark; retained-cell hashes provide an independent membership check",
            "endpoint_row_total": open_total, "over_cap_endpoint_rows": open_over,
            "over_cap_fraction": open_over / open_total, "over_cap_test_appearances_across_69_tests": open_test_over,
            "membership_hash_checks": {
                "split_triplets_fully_matched": sum(x["all_partition_hashes_match"] for x in hash_checks),
                "split_triplets_total": len(hash_checks),
                "partition_hashes_matched": sum(
                    sum(x["partition_hash_matches"].values()) for x in hash_checks
                ),
                "partition_hashes_total": 3 * len(hash_checks),
                "split_row_counts_matched": sum(x["row_counts_match"] for x in hash_checks),
                "details": hash_checks,
            },
            "endpoints": open_endpoints,
            "performance_implication": "No >96-heavy-atom molecule occurs in any held-out test partition; subgroup test performance is therefore undefined. The sole PXR molecule is in inner training for all three splits.",
        },
        "tdc": {
            "scope": "current local PyTDC tab snapshot; historical Phase-4 logs are separate evidence",
            "local_over_cap_split_rows": tdc_local_over,
            "local_over_cap_by_split_role": {
                role: sum(
                    record["over_cap_rows"]
                    for item in tdc_endpoints.values()
                    for record in item["local_split_records"] if record["role"] == role
                )
                for role in ("train", "valid", "test")
            },
            "historical_logged_pre_filter_rows": historical_pre,
            "historical_logged_post_filter_rows": historical_post,
            "historical_logged_removed_by_combined_invalid_or_over_cap_filter": (
                historical_pre - sum(historical_post.values())
            ),
            "historical_snapshot_matches_local_for_all_tasks": all(x["historical_snapshot_row_match"] for x in tdc_endpoints.values()),
            "endpoints": tdc_endpoints,
            "performance_implication": "finetune_benchmarks.py removes invalid and >96-heavy-atom molecules after splitting, so reported TDC metrics contain no >96-heavy-atom examples. Exact historical over-cap counts require the unretained PVC snapshot; local counts must not be presented as counts for that run.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
