#!/usr/bin/env python3
"""Audit the Step-2 TDC exclusion builder against the reported TDC split.

The historical Step-2 builder in ``build_chembl_activity.py`` requests a
single-prediction scaffold split with ``seed=42`` but does not pass ``frac``.
The primary TDC evaluator in ``finetune_benchmarks.py`` explicitly requests
``frac=[0.7, 0.15, 0.15]``.  This script executes both calls using the same
PyTDC installation, canonicalizes the resulting test molecules with RDKit,
and reports exact per-endpoint set intersections.  It also executes every
dataset lookup attempted by the builder without suppressing failures.

Example (the paper's cluster environment pins PyTDC 0.4.1)::

    python scripts/audit_tdc_step2_exclusions.py \
      --tdc-data-dir /mnt/tdc_data \
      --output-dir results_package/tdc_step2_exclusion_audit

The output is evidence only.  It does not modify the Step-2 training data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem


BUILDER_DATASETS: tuple[tuple[str, str], ...] = (
    ("ADME", "BBB_Martins"),
    ("ADME", "Caco2_Wang"),
    ("ADME", "HIA_Hou"),
    ("ADME", "Pgp_Broccatelli"),
    ("ADME", "Bioavailability_Ma"),
    ("ADME", "Lipophilicity_AstraZeneca"),
    ("ADME", "Solubility_AqSolDB"),
    ("ADME", "CYP2C19_Veith"),
    ("ADME", "CYP2D6_Veith"),
    ("ADME", "CYP3A4_Veith"),
    ("ADME", "CYP1A2_Veith"),
    ("ADME", "CYP2C9_Veith"),
    ("ADME", "CYP2C9_Substrate_CarbonMangels"),
    ("ADME", "CYP2D6_Substrate_CarbonMangels"),
    ("ADME", "CYP3A4_Substrate_CarbonMangels"),
    ("ADME", "Half_Life_Obach"),
    ("ADME", "Clearance_Hepatocyte_AZ"),
    ("ADME", "Clearance_Microsome_AZ"),
    ("Tox", "hERG"),
    ("Tox", "hERG_Karim"),
    ("Tox", "AMES"),
    ("Tox", "DILI"),
    ("Tox", "Skin_Reaction"),
    ("Tox", "Carcinogens_Lagunin"),
    ("Tox", "ClinTox"),
    ("Tox", "Tox21"),
    ("HTS", "SARSCoV2_Vitro_Touret"),
    ("HTS", "SARSCoV2_3CLPro_Diamond"),
    # The builder performs this lookup separately after the loop.
    ("ADME", "BACE"),
)


EVALUATION_TDC22: tuple[str, ...] = (
    "bbb_martins",
    "hia_hou",
    "herg",
    "caco2_wang",
    "half_life_obach",
    "cyp2d6_veith",
    "dili",
    "bioavailability_ma",
    "cyp2c9_veith",
    "cyp3a4_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "ames",
    "pgp_broccatelli",
    "vdss_lombardo",
    "lipophilicity_astrazeneca",
    "ppbr_az",
    "solubility_aqsoldb",
    "ld50_zhu",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
)


BUILDER_DEFAULT_FRAC = (0.7, 0.1, 0.2)
EVALUATION_FRAC = (0.7, 0.15, 0.15)
SPLIT_METHOD = "scaffold"
SPLIT_SEED = 42


@dataclass(frozen=True)
class CanonicalSet:
    rows: int
    valid_rows: int
    invalid_rows: int
    over_max_heavy_rows: int
    unique_canonical_smiles: int
    smiles: frozenset[str]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    return name.strip().lower()


def canonicalize(values: Iterable[Any], max_heavy: int | None = None) -> CanonicalSet:
    canonical: list[str] = []
    invalid = 0
    over_max_heavy = 0
    rows = 0
    for value in values:
        rows += 1
        if not isinstance(value, str):
            invalid += 1
            continue
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            invalid += 1
            continue
        if max_heavy is not None and mol.GetNumHeavyAtoms() > max_heavy:
            over_max_heavy += 1
            continue
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    unique = frozenset(canonical)
    return CanonicalSet(
        rows=rows,
        valid_rows=len(canonical),
        invalid_rows=invalid,
        over_max_heavy_rows=over_max_heavy,
        unique_canonical_smiles=len(unique),
        smiles=unique,
    )


def frame_drugs(
    frame: pd.DataFrame,
    dataset: str,
    split: str,
    max_heavy: int | None = None,
) -> CanonicalSet:
    if "Drug" not in frame.columns:
        raise ValueError(
            f"{dataset} {split} frame lacks Drug column; columns={list(frame.columns)}"
        )
    return canonicalize(frame["Drug"].tolist(), max_heavy=max_heavy)


def load_exact(module_name: str, dataset_name: str, data_dir: Path):
    from tdc.single_pred import ADME, HTS, Tox

    classes = {"ADME": ADME, "Tox": Tox, "HTS": HTS}
    return classes[module_name](name=dataset_name, path=str(data_dir))


def load_as_evaluator(dataset_name: str, data_dir: Path):
    """Mirror finetune_benchmarks.py: ADME first, then Tox."""
    from tdc.single_pred import ADME, Tox

    errors: list[dict[str, str]] = []
    for module_name, cls in (("ADME", ADME), ("Tox", Tox)):
        try:
            return module_name, cls(name=dataset_name, path=str(data_dir)), errors
        except Exception as exc:  # deliberately records the evaluator's fallback
            errors.append(
                {
                    "module": module_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    raise RuntimeError(json.dumps(errors, sort_keys=True))


def split_signature_record(task: Any) -> dict[str, Any]:
    signature = inspect.signature(task.get_split)
    frac_default = signature.parameters["frac"].default
    return {
        "signature": str(signature),
        "frac_default": list(frac_default),
    }


def compare_sets(builder: CanonicalSet, evaluation: CanonicalSet) -> dict[str, Any]:
    intersection = builder.smiles & evaluation.smiles
    union = builder.smiles | evaluation.smiles
    evaluation_missing = evaluation.smiles - builder.smiles
    builder_only = builder.smiles - evaluation.smiles
    return {
        "intersection_unique": len(intersection),
        "evaluation_test_missing_from_builder_exclusion": len(evaluation_missing),
        "builder_exclusion_only": len(builder_only),
        "evaluation_test_coverage_fraction": (
            len(intersection) / len(evaluation.smiles) if evaluation.smiles else None
        ),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "sets_identical": builder.smiles == evaluation.smiles,
        "evaluation_missing_examples": sorted(evaluation_missing)[:10],
        "builder_only_examples": sorted(builder_only)[:10],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TDC Step-2 exclusion audit",
        "",
        "## Verdict",
        "",
        (
            "The historical Step-2 builder and the reported evaluator do **not** "
            "construct the same test split. The builder relied on PyTDC's default "
            f"fractions `{summary['builder_default_frac']}`; evaluation explicitly "
            f"used `{summary['evaluation_frac']}`."
        ),
        "",
        (
            f"Across the {summary['evaluation_endpoint_count']} reported endpoints, "
            f"{summary['evaluation_endpoints_absent_from_builder_count']} endpoints "
            "were absent from the builder's lookup list and "
            f"{summary['evaluation_unique_test_molecules_missing_from_builder_union']} "
            "unique evaluation-test molecules were not in the union of successfully "
            "constructed builder exclusions."
        ),
        "",
        "## Verified facts",
        "",
        f"- PyTDC version: `{payload['environment']['pytdc_version']}`.",
        f"- Split method/seed: `{SPLIT_METHOD}`, `{SPLIT_SEED}`.",
        f"- Builder default: `{summary['builder_default_frac']}`.",
        f"- Evaluator explicit fractions: `{summary['evaluation_frac']}`.",
        (
            "- Evaluation endpoints absent from the builder list: "
            + ", ".join(f"`{x}`" for x in summary["evaluation_endpoints_absent_from_builder"])
            + "."
        ),
        (
            f"- Builder lookups attempted: {summary['builder_lookup_count']}; "
            f"successful: {summary['builder_lookup_success_count']}; "
            f"failed (and historically suppressed): {summary['builder_lookup_failure_count']}."
        ),
        "",
        "## Endpoint comparison",
        "",
        "| Endpoint | Builder status | Builder test unique | Effective evaluation-test unique | Covered | Missing | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["evaluation_endpoints"]:
        coverage = row.get("evaluation_test_coverage_fraction")
        coverage_text = "n/a" if coverage is None else f"{coverage:.3f}"
        lines.append(
            "| {endpoint} | {builder_status} | {builder_test_unique} | "
            "{evaluation_test_unique} | {intersection_unique} | "
            "{evaluation_test_missing_from_builder_exclusion} | {coverage} |".format(
                endpoint=row["endpoint"],
                builder_status=row["builder_status"],
                builder_test_unique=row["builder_test_unique"],
                evaluation_test_unique=row["evaluation_test_unique"],
                intersection_unique=row["intersection_unique"],
                evaluation_test_missing_from_builder_exclusion=row[
                    "evaluation_test_missing_from_builder_exclusion"
                ],
                coverage=coverage_text,
            )
        )
    lines.extend(
        [
            "",
            "## Builder failures",
            "",
        ]
    )
    failures = [row for row in payload["builder_lookups"] if row["status"] != "ok"]
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['module']}:{row['dataset']}`: "
                f"`{row['error_type']}` — {row['error_message']}"
            )
    else:
        lines.append("- None in this reproduced run.")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            (
            "The primary endpoint table applies the evaluator's post-split validity "
            "filter (parseable RDKit molecule, 1--96 heavy atoms). This audit reproduces "
            "the split construction from source and TDC data. "
                "It does not prove which exclusion set was materialized into the historical "
                "511,898-by-642 Step-2 artifact because that artifact does not contain a "
                "saved exclusion manifest or raw split membership list."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tdc-data-dir",
        type=Path,
        default=Path("tmp/tdc_step2_exclusion_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_package/tdc_step2_exclusion_audit"),
    )
    parser.add_argument(
        "--builder-source", type=Path, default=Path("build_chembl_activity.py")
    )
    parser.add_argument(
        "--evaluation-source", type=Path, default=Path("finetune_benchmarks.py")
    )
    args = parser.parse_args()

    from tdc.version import __version__ as pytdc_version

    args.tdc_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder_records: list[dict[str, Any]] = []
    builder_by_name: dict[str, CanonicalSet] = {}
    signature_record: dict[str, Any] | None = None
    builder_union: set[str] = set()

    for module_name, dataset_name in BUILDER_DATASETS:
        record: dict[str, Any] = {
            "module": module_name,
            "dataset": dataset_name,
            "normalized_name": normalize_name(dataset_name),
        }
        try:
            task = load_exact(module_name, dataset_name, args.tdc_data_dir)
            if signature_record is None:
                signature_record = split_signature_record(task)
                observed_default = tuple(signature_record["frac_default"])
                if observed_default != BUILDER_DEFAULT_FRAC:
                    raise RuntimeError(
                        f"unexpected PyTDC get_split default {observed_default}; "
                        f"expected {BUILDER_DEFAULT_FRAC}"
                    )
            split = task.get_split(method=SPLIT_METHOD, seed=SPLIT_SEED)
            test_set = frame_drugs(split["test"], dataset_name, "builder-default test")
            builder_by_name[normalize_name(dataset_name)] = test_set
            builder_union.update(test_set.smiles)
            record.update(
                {
                    "status": "ok",
                    "test_rows": test_set.rows,
                    "test_valid_rows": test_set.valid_rows,
                    "test_invalid_rows": test_set.invalid_rows,
                    "test_unique_canonical_smiles": test_set.unique_canonical_smiles,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        builder_records.append(record)

    evaluation_records: list[dict[str, Any]] = []
    evaluation_union: set[str] = set()
    builder_names = {normalize_name(name) for _, name in BUILDER_DATASETS}

    for endpoint in EVALUATION_TDC22:
        record: dict[str, Any] = {
            "endpoint": endpoint,
            "builder_listed": endpoint in builder_names,
        }
        try:
            module_name, task, fallback_errors = load_as_evaluator(
                endpoint, args.tdc_data_dir
            )
            explicit_split = task.get_split(
                method=SPLIT_METHOD,
                seed=SPLIT_SEED,
                frac=list(EVALUATION_FRAC),
            )
            evaluation_raw_set = frame_drugs(
                explicit_split["test"], endpoint, "evaluation-explicit test"
            )
            # Mirror finetune_benchmarks.py:is_valid_smiles, which is applied to
            # each split after construction and before model evaluation.
            evaluation_set = frame_drugs(
                explicit_split["test"],
                endpoint,
                "evaluation-effective test",
                max_heavy=96,
            )
            evaluation_union.update(evaluation_set.smiles)
            record.update(
                {
                    "evaluation_status": "ok",
                    "evaluation_module": module_name,
                    "evaluation_loader_prior_failures": fallback_errors,
                    "evaluation_raw_test_rows": evaluation_raw_set.rows,
                    "evaluation_raw_test_valid_rows": evaluation_raw_set.valid_rows,
                    "evaluation_raw_test_invalid_rows": evaluation_raw_set.invalid_rows,
                    "evaluation_raw_test_unique": evaluation_raw_set.unique_canonical_smiles,
                    "evaluation_test_rows": evaluation_set.rows,
                    "evaluation_test_valid_rows": evaluation_set.valid_rows,
                    "evaluation_test_invalid_rows": evaluation_set.invalid_rows,
                    "evaluation_test_over_96_heavy_rows": evaluation_set.over_max_heavy_rows,
                    "evaluation_test_unique": evaluation_set.unique_canonical_smiles,
                    "raw_split_comparison": (
                        compare_sets(builder_by_name[endpoint], evaluation_raw_set)
                        if endpoint in builder_by_name
                        else None
                    ),
                }
            )
            builder_set = builder_by_name.get(endpoint)
            if builder_set is None:
                record.update(
                    {
                        "builder_status": (
                            "absent_from_builder_list"
                            if endpoint not in builder_names
                            else "builder_lookup_failed"
                        ),
                        "builder_test_unique": 0,
                        "intersection_unique": 0,
                        "evaluation_test_missing_from_builder_exclusion": evaluation_set.unique_canonical_smiles,
                        "builder_exclusion_only": 0,
                        "evaluation_test_coverage_fraction": 0.0,
                        "jaccard": 0.0,
                        "sets_identical": False,
                    }
                )
            else:
                record.update(
                    {
                        "builder_status": "ok",
                        "builder_test_unique": builder_set.unique_canonical_smiles,
                        **compare_sets(builder_set, evaluation_set),
                    }
                )
        except Exception as exc:
            record.update(
                {
                    "evaluation_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "builder_status": (
                        "ok" if endpoint in builder_by_name else "unavailable"
                    ),
                }
            )
        evaluation_records.append(record)

    absent = [name for name in EVALUATION_TDC22 if name not in builder_names]
    failures = [row for row in builder_records if row["status"] != "ok"]
    evaluation_failures = [
        row for row in evaluation_records if row.get("evaluation_status") != "ok"
    ]
    eval_missing_from_builder_union = evaluation_union - builder_union
    signature_record = signature_record or {
        "signature": "unavailable: every builder lookup failed",
        "frac_default": None,
    }

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytdc_version": pytdc_version,
            "rdkit_version": getattr(sys.modules.get("rdkit"), "__version__", None),
            "tdc_data_dir": str(args.tdc_data_dir.resolve()),
        },
        "source": {
            "builder_source": str(args.builder_source),
            "builder_source_sha256": sha256_file(args.builder_source),
            "evaluation_source": str(args.evaluation_source),
            "evaluation_source_sha256": sha256_file(args.evaluation_source),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "split_contract": {
            "method": SPLIT_METHOD,
            "seed": SPLIT_SEED,
            "observed_get_split": signature_record,
            "builder_call": "get_split(method='scaffold', seed=42)",
            "evaluation_call": (
                "get_split(method='scaffold', seed=42, frac=[0.7, 0.15, 0.15])"
            ),
        },
        "summary": {
            "builder_default_frac": list(BUILDER_DEFAULT_FRAC),
            "evaluation_frac": list(EVALUATION_FRAC),
            "fractions_match": BUILDER_DEFAULT_FRAC == EVALUATION_FRAC,
            "builder_lookup_count": len(builder_records),
            "builder_lookup_success_count": len(builder_records) - len(failures),
            "builder_lookup_failure_count": len(failures),
            "evaluation_endpoint_count": len(EVALUATION_TDC22),
            "evaluation_lookup_failure_count": len(evaluation_failures),
            "evaluation_endpoints_absent_from_builder": absent,
            "evaluation_endpoints_absent_from_builder_count": len(absent),
            "builder_union_unique_canonical_smiles": len(builder_union),
            "evaluation_union_unique_canonical_smiles": len(evaluation_union),
            "evaluation_unique_test_molecules_missing_from_builder_union": len(
                eval_missing_from_builder_union
            ),
            "evaluation_union_coverage_fraction": (
                len(evaluation_union & builder_union) / len(evaluation_union)
                if evaluation_union
                else None
            ),
            "evaluation_missing_from_builder_union_examples": sorted(
                eval_missing_from_builder_union
            )[:25],
            "evaluation_test_rows_filtered_over_96_heavy": sum(
                int(row.get("evaluation_test_over_96_heavy_rows", 0))
                for row in evaluation_records
            ),
        },
        "builder_lookups": builder_records,
        "evaluation_endpoints": evaluation_records,
        "limitations": [
            (
                "The historical final Step-2 artifact has no retained exclusion manifest "
                "or raw split-membership list, so this reproduces source behavior rather "
                "than verifying the exact materialized artifact membership."
            ),
            (
                "Exact-set comparisons use RDKit canonical SMILES and therefore do not "
                "collapse stereoisomers, salts, or tautomers beyond RDKit canonicalization."
            ),
            (
                "The effective evaluation set mirrors finetune_benchmarks.py by excluding "
                "invalid SMILES and molecules with more than 96 heavy atoms after split "
                "construction; raw split comparisons are retained per endpoint in JSON."
            ),
        ],
    }

    json_path = args.output_dir / "tdc_step2_exclusion_audit.json"
    csv_path = args.output_dir / "tdc_step2_exclusion_audit.csv"
    md_path = args.output_dir / "tdc_step2_exclusion_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat_rows: list[dict[str, Any]] = []
    for row in evaluation_records:
        flat_rows.append(
            {
                key: (json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value)
                for key, value in row.items()
            }
        )
    write_csv(csv_path, flat_rows)
    md_path.write_text(markdown_report(payload), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 1 if evaluation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
