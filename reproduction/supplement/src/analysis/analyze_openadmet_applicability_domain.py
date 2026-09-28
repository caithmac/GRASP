#!/usr/bin/env python3
"""Audit OpenADMET error against nearest same-split training-set similarity.

This is an exploratory applicability-domain analysis. It reconstructs the exact
test-row order from the retained endpoint-row table, verifies it against every
saved full-fine-tuning cell, computes ECFP4 Tanimoto similarity to the nearest
training molecule, and summarizes within-cell error/similarity associations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_ROWS = (
    REPO_ROOT
    / "results_package"
    / "results_openadmet_step2_leakage_audit_20260917_v3"
    / "endpoint_rows.csv.gz"
)
DEFAULT_RESULTS = REPO_ROOT / "results_rtd_phase4_95m_moljepa"
DEFAULT_OUTPUT = ROOT / "evidence" / "openadmet_applicability_domain"
THRESHOLDS = (0.3, 0.5, 0.7)
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260918


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smiles_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def endpoint_bootstrap(
    values: dict[str, float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    names = sorted(values)
    observed = np.asarray([values[name] for name in names], dtype=np.float64)
    require(observed.size > 0, "endpoint bootstrap received no values")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    chunk = 10_000
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        indices = rng.integers(0, observed.size, size=(stop - start, observed.size))
        draws[start:stop] = observed[indices].mean(axis=1)
    return {
        "endpoint_count": int(observed.size),
        "mean": float(observed.mean()),
        "median": float(np.median(observed)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "positive_endpoints": int(np.sum(observed > 0)),
        "negative_endpoints": int(np.sum(observed < 0)),
        "zero_endpoints": int(np.sum(observed == 0)),
        "exact_two_sided_sign_p": float(
            stats.binomtest(
                int(min(np.sum(observed > 0), np.sum(observed < 0))),
                int(np.sum(observed != 0)),
                0.5,
                alternative="two-sided",
            ).pvalue
        ) if np.sum(observed != 0) else 1.0,
        "endpoint_values": {name: float(values[name]) for name in names},
    }


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            require(row["standardization_status"] == "ok",
                    f"non-standardized endpoint row: {row['endpoint_record_id']}")
            grouped[row["endpoint_slug"]].append(row)
    require(len(grouped) == 23, f"expected 23 endpoints, found {len(grouped)}")
    return dict(grouped)


def fingerprint_cache(
    grouped: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, object], object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: dict[str, object] = {}
    for rows in grouped.values():
        for row in rows:
            smiles = row["standardized_smiles"]
            if smiles in fingerprints:
                continue
            molecule = Chem.MolFromSmiles(smiles)
            require(molecule is not None, f"RDKit could not parse standardized SMILES: {smiles}")
            fingerprints[smiles] = generator.GetFingerprint(molecule)
    return fingerprints, generator


def summarize_cell(
    endpoint: str,
    split: str,
    test_rows: list[dict[str, str]],
    y_pred: list[float],
    similarities: list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_true = np.asarray([float(row["y"]) for row in test_rows], dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    similarity = np.asarray(similarities, dtype=np.float64)
    error = np.abs(predicted - y_true)
    rho, raw_p = stats.spearmanr(similarity, error)
    detail_rows = []
    for row, truth, prediction, absolute_error, nearest in zip(
        test_rows, y_true, predicted, error, similarity
    ):
        detail_rows.append({
            "endpoint": endpoint,
            "split": split,
            "endpoint_row_index": int(row["endpoint_row_index"]),
            "query_id": int(row["query_id"]),
            "y_true": float(truth),
            "y_pred": float(prediction),
            "absolute_error": float(absolute_error),
            "nearest_train_ecfp4_tanimoto": float(nearest),
        })
    summary: dict[str, object] = {
        "endpoint": endpoint,
        "split": split,
        "test_rows": len(test_rows),
        "mae": float(error.mean()),
        "similarity_min": float(similarity.min()),
        "similarity_q25": float(np.quantile(similarity, 0.25)),
        "similarity_median": float(np.median(similarity)),
        "similarity_q75": float(np.quantile(similarity, 0.75)),
        "similarity_max": float(similarity.max()),
        "spearman_similarity_vs_absolute_error": finite_or_none(rho),
        "spearman_raw_p": finite_or_none(raw_p),
    }
    for threshold in THRESHOLDS:
        low = similarity < threshold
        high = ~low
        prefix = f"threshold_{threshold:.1f}".replace(".", "_")
        summary[f"{prefix}_low_n"] = int(low.sum())
        summary[f"{prefix}_high_n"] = int(high.sum())
        summary[f"{prefix}_low_mae"] = float(error[low].mean()) if low.any() else None
        summary[f"{prefix}_high_mae"] = float(error[high].mean()) if high.any() else None
        summary[f"{prefix}_relative_gap"] = (
            float((error[low].mean() - error[high].mean()) / error.mean())
            if low.any() and high.any() and error.mean() > 0 else None
        )
    return summary, detail_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(rows_path: Path, results_root: Path, output_dir: Path) -> dict[str, object]:
    require(rows_path.is_file(), f"endpoint-row input is missing: {rows_path}")
    require(results_root.is_dir(), f"result root is missing: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_rows(rows_path)
    fingerprints, _ = fingerprint_cache(grouped)

    cell_summaries: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    cell_files: list[Path] = []
    for endpoint in sorted(grouped):
        rows = grouped[endpoint]
        for split_index in (1, 2, 3):
            split = f"split{split_index}"
            partition = f"{split}_neural_partition"
            train_rows = [row for row in rows if row[partition] == "train"]
            validation_rows = [row for row in rows if row[partition] == "validation"]
            test_rows = [row for row in rows if row[partition] == "test"]
            require(train_rows and validation_rows and test_rows,
                    f"empty partition for {endpoint} {split}")
            cell_path = results_root / endpoint / "cells" / f"{split}_full_mix.json"
            require(cell_path.is_file(), f"missing full-fine-tuning cell: {cell_path}")
            cell_files.append(cell_path)
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            require(cell["method"] == "full_mix" and cell["task"] == endpoint
                    and cell["split"] == split,
                    f"cell identity drifted for {endpoint} {split}")
            require(cell["train_rows"] == len(train_rows)
                    and cell["validation_rows"] == len(validation_rows)
                    and cell["test_rows"] == len(test_rows),
                    f"partition size drifted for {endpoint} {split}")
            for label, partition_rows in (
                ("train", train_rows),
                ("validation", validation_rows),
                ("test", test_rows),
            ):
                observed = smiles_sha256([row["original_smiles"] for row in partition_rows])
                require(observed == cell[f"{label}_smiles_sha256"],
                        f"{label} SMILES hash drifted for {endpoint} {split}")
            truth = [float(row["y"]) for row in test_rows]
            require(len(truth) == len(cell["y_true"]) == len(cell["y_pred"]),
                    f"prediction length drifted for {endpoint} {split}")
            require(all(a == b for a, b in zip(truth, cell["y_true"])),
                    f"test target order drifted for {endpoint} {split}")
            train_fingerprints = [
                fingerprints[row["standardized_smiles"]] for row in train_rows
            ]
            similarities = [
                max(DataStructs.BulkTanimotoSimilarity(
                    fingerprints[row["standardized_smiles"]], train_fingerprints
                ))
                for row in test_rows
            ]
            summary, cell_detail = summarize_cell(
                endpoint, split, test_rows, cell["y_pred"], similarities
            )
            cell_summaries.append(summary)
            detail_rows.extend(cell_detail)

    require(len(cell_summaries) == 69, "expected exactly 69 endpoint-split summaries")
    finite_cells = [
        row for row in cell_summaries
        if row["spearman_similarity_vs_absolute_error"] is not None
    ]
    raw_p_values = [float(row["spearman_raw_p"]) for row in finite_cells]
    adjusted = holm_adjust(raw_p_values)
    for row, value in zip(finite_cells, adjusted):
        row["spearman_holm69_p"] = value
    for row in cell_summaries:
        row.setdefault("spearman_holm69_p", None)

    endpoint_rho: dict[str, float] = {}
    for endpoint in sorted(grouped):
        values = [
            float(row["spearman_similarity_vs_absolute_error"])
            for row in finite_cells if row["endpoint"] == endpoint
        ]
        require(values, f"no finite correlation for endpoint {endpoint}")
        endpoint_rho[endpoint] = float(np.mean(values))

    threshold_summaries: dict[str, object] = {}
    for threshold in THRESHOLDS:
        prefix = f"threshold_{threshold:.1f}".replace(".", "_")
        endpoint_gaps: dict[str, float] = {}
        eligible_cells = []
        for row in cell_summaries:
            gap = row[f"{prefix}_relative_gap"]
            if (gap is not None and row[f"{prefix}_low_n"] >= 5
                    and row[f"{prefix}_high_n"] >= 5):
                eligible_cells.append(row)
        for endpoint in sorted(grouped):
            values = [
                float(row[f"{prefix}_relative_gap"])
                for row in eligible_cells if row["endpoint"] == endpoint
            ]
            if values:
                endpoint_gaps[endpoint] = float(np.mean(values))
        threshold_summaries[f"{threshold:.1f}"] = {
            "definition": (
                "(MAE below threshold - MAE at/above threshold) / overall cell MAE; "
                "positive values mean lower-similarity test rows have larger error"
            ),
            "minimum_rows_per_side_per_cell": 5,
            "eligible_cell_count": len(eligible_cells),
            "endpoint_bootstrap": endpoint_bootstrap(endpoint_gaps)
            if endpoint_gaps else None,
        }

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "exploratory same-endpoint, same-split applicability-domain audit",
        "interpretation_boundary": (
            "Associations are descriptive and do not establish calibrated uncertainty, "
            "causal domain-shift effects, or pretraining-corpus applicability."
        ),
        "endpoint_count": len(grouped),
        "cell_count": len(cell_summaries),
        "test_prediction_rows_including_split_repetitions": len(detail_rows),
        "fingerprint": {
            "type": "Morgan bit vector (ECFP4)",
            "radius": 2,
            "bits": 2048,
            "similarity": "Tanimoto",
            "reference": "nearest molecule in the same endpoint and outer-split training set",
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "similarity_vs_absolute_error": endpoint_bootstrap(endpoint_rho),
        "cell_level": {
            "finite_spearman_cells": len(finite_cells),
            "negative_rho_cells": sum(
                float(row["spearman_similarity_vs_absolute_error"]) < 0
                for row in finite_cells
            ),
            "positive_rho_cells": sum(
                float(row["spearman_similarity_vs_absolute_error"]) > 0
                for row in finite_cells
            ),
            "holm69_significant_cells": sum(
                float(row["spearman_holm69_p"]) < 0.05 for row in finite_cells
            ),
        },
        "threshold_sensitivity": threshold_summaries,
        "inputs": {
            "endpoint_rows": {
                "path": rows_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
                "sha256": sha256(rows_path),
            },
            "full_mix_cells": [
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256(path),
                }
                for path in sorted(cell_files)
            ],
        },
    }

    detail_path = output_dir / "test_row_applicability.csv"
    cells_path = output_dir / "endpoint_split_summary.csv"
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "RESULTS.md"
    write_csv(detail_path, detail_rows)
    write_csv(cells_path, cell_summaries)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rho = summary["similarity_vs_absolute_error"]
    threshold = summary["threshold_sensitivity"]["0.5"]
    threshold_bootstrap = threshold["endpoint_bootstrap"]
    results_lines = [
        "# OpenADMET applicability-domain audit",
        "",
        "Status: **PASS**",
        "",
        (
            f"Across {summary['endpoint_count']} endpoints and {summary['cell_count']} "
            f"endpoint--split cells, the mean endpoint-level Spearman association between "
            f"nearest-training ECFP4 Tanimoto similarity and absolute error is "
            f"{rho['mean']:+.4f} (endpoint-bootstrap 95% interval "
            f"[{rho['bootstrap_95_interval'][0]:+.4f}, "
            f"{rho['bootstrap_95_interval'][1]:+.4f}])."
        ),
        (
            f"At the exploratory 0.5 similarity threshold, {threshold['eligible_cell_count']} "
            "cells contain at least five rows on each side. "
            + (
                f"Their endpoint-averaged relative low-minus-high-similarity MAE gap is "
                f"{threshold_bootstrap['mean']:+.4f} (95% interval "
                f"[{threshold_bootstrap['bootstrap_95_interval'][0]:+.4f}, "
                f"{threshold_bootstrap['bootstrap_95_interval'][1]:+.4f}])."
                if threshold_bootstrap else
                "No endpoint-level threshold estimate is available."
            )
        ),
        "",
        (
            "This is an exploratory same-endpoint, same-split diagnostic. It does not "
            "establish calibrated uncertainty, a causal effect of domain shift, or the "
            "model's relationship to the Step-1 pretraining domain."
        ),
        "",
        "Every saved target vector and train/validation/test SMILES hash was checked before analysis.",
    ]
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")

    manifest_entries = []
    for path in (detail_path, cells_path, summary_path, results_path):
        manifest_entries.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    (output_dir / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "files": manifest_entries},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.rows, arguments.results_root, arguments.output_dir)
    print(json.dumps({
        "status": result["status"],
        "endpoint_count": result["endpoint_count"],
        "cell_count": result["cell_count"],
        "mean_endpoint_spearman": result["similarity_vs_absolute_error"]["mean"],
        "spearman_interval": result["similarity_vs_absolute_error"]["bootstrap_95_interval"],
    }, indent=2))
