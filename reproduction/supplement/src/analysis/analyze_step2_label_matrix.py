#!/usr/bin/env python3
"""Validate and summarize a sparse binary Step-2 assay-label matrix.

The input matrix must be a two-dimensional numeric ``.npy`` array whose rows
are molecules and whose columns correspond one-to-one, in order, with the
newline-delimited assay IDs. Labels must be exactly 0 or 1; floating-point NaN
is the only accepted missing-value representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_CHUNK_ROWS = 8192
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_assay_ids(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"assay-ID file is not valid UTF-8: {path}") from exc
    if not lines:
        raise ValueError("assay-ID file is empty")
    assay_ids = [line.strip() for line in lines]
    blank_rows = [index + 1 for index, value in enumerate(assay_ids) if not value]
    if blank_rows:
        preview = ", ".join(map(str, blank_rows[:10]))
        raise ValueError(f"blank assay ID at line(s): {preview}")
    duplicates = pd.Series(assay_ids, dtype="string")[
        pd.Series(assay_ids, dtype="string").duplicated(keep=False)
    ].unique()
    if len(duplicates):
        preview = ", ".join(map(str, duplicates[:10]))
        raise ValueError(f"duplicate assay ID(s): {preview}")
    return assay_ids


def load_label_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not load numeric NPY matrix: {path}") from exc
    if matrix.ndim != 2:
        raise ValueError(f"label matrix must be 2-D, observed shape={matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"label matrix dimensions must be nonzero, shape={matrix.shape}")
    if matrix.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(
            f"label matrix dtype must be bool, integer, or floating point; "
            f"observed dtype={matrix.dtype}"
        )
    return matrix


def numeric_summary(values: np.ndarray) -> dict[str, int | float | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {
            "n": 0,
            "min": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "max": None,
            "mean": None,
            "sd_ddof0": None,
        }
    quantiles = np.quantile(data, QUANTILES)
    return {
        "n": int(data.size),
        "min": float(np.min(data)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "sd_ddof0": float(np.std(data, ddof=0)),
    }


def validate_and_count(
    matrix: np.ndarray, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_molecules, n_assays = map(int, matrix.shape)
    observed_by_assay = np.zeros(n_assays, dtype=np.int64)
    positive_by_assay = np.zeros(n_assays, dtype=np.int64)
    molecule_histogram = np.zeros(n_assays + 1, dtype=np.int64)

    for start in range(0, n_molecules, chunk_rows):
        stop = min(start + chunk_rows, n_molecules)
        block = np.asarray(matrix[start:stop])
        if matrix.dtype.kind == "f":
            observed = ~np.isnan(block)
        else:
            observed = np.ones(block.shape, dtype=bool)

        invalid = observed & (block != 0) & (block != 1)
        if np.any(invalid):
            local_row, column = np.argwhere(invalid)[0]
            row = start + int(local_row)
            value = block[int(local_row), int(column)]
            raise ValueError(
                "invalid label value; only 0, 1, and floating-point NaN are "
                f"allowed (row={row}, column={int(column)}, value={value!r})"
            )

        observed_by_assay += observed.sum(axis=0, dtype=np.int64)
        positive_by_assay += (observed & (block == 1)).sum(axis=0, dtype=np.int64)
        molecule_counts = observed.sum(axis=1, dtype=np.int64)
        molecule_histogram += np.bincount(
            molecule_counts, minlength=n_assays + 1
        ).astype(np.int64, copy=False)

    return observed_by_assay, positive_by_assay, molecule_histogram


def markdown_number(value: int | float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def summary_table_markdown(summary: dict[str, Any]) -> str:
    order = ("n", "min", "q05", "q25", "median", "q75", "q95", "max", "mean", "sd_ddof0")
    labels = {
        "n": "N",
        "min": "Min",
        "q05": "5th percentile",
        "q25": "25th percentile",
        "median": "Median",
        "q75": "75th percentile",
        "q95": "95th percentile",
        "max": "Max",
        "mean": "Mean",
        "sd_ddof0": "SD (ddof=0)",
    }
    rows = ["| Statistic | Value |", "|---|---:|"]
    rows.extend(
        f"| {labels[key]} | {markdown_number(summary[key])} |" for key in order
    )
    return "\n".join(rows)


def make_results_markdown(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    matrix = summary["matrix"]
    totals = summary["totals"]
    assays = summary["assays"]
    molecules = summary["molecules"]
    return f"""# Step-2 Label-Matrix Audit

Validation passed. The matrix contains only binary observed labels (0/1) and
floating-point NaN missing values, and the assay-ID count matches the column count.

## Inputs

| Input | Absolute path | SHA-256 |
|---|---|---|
| Labels | `{inputs['labels_npy']['path']}` | `{inputs['labels_npy']['sha256']}` |
| Assay IDs | `{inputs['assay_ids']['path']}` | `{inputs['assay_ids']['sha256']}` |

## Matrix and label totals

| Quantity | Value |
|---|---:|
| Shape | {matrix['molecules']:,} molecules x {matrix['assays']:,} assays |
| Cells | {totals['cells']:,} |
| Observed labels | {totals['observed']:,} |
| Positive labels | {totals['positive']:,} |
| Negative labels | {totals['negative']:,} |
| Missing labels | {totals['missing']:,} |
| Label density | {totals['label_density']:.8f} |
| Zero-label molecule rows | {molecules['zero_label_rows']:,} |
| Constant-label assays | {assays['constant_label_assays']:,} |
| Observed labels from constant assays | {assays['constant_label_observations']:,} ({assays['constant_label_observation_fraction']:.4%}) |
| Assays with zero observed labels | {assays['zero_observation_assays']:,} |

## Observed labels per assay

{summary_table_markdown(assays['observed_count_distribution'])}

## Positive prevalence per nonempty assay

Prevalence is `positive_count / observed_count`; assays with no observed labels
are excluded from this distribution.

{summary_table_markdown(assays['positive_prevalence_distribution'])}

## Observed labels per molecule

{summary_table_markdown(molecules['observed_label_count_distribution'])}

Full assay-level results are in `per_assay_summary.csv`. The exact molecule-level
count distribution is in `per_molecule_label_count_histogram.csv`; no molecular
identifiers or label values are copied into either output.
"""


def analyze(labels_path: Path, assay_ids_path: Path, output_dir: Path, chunk_rows: int) -> None:
    if chunk_rows <= 0:
        raise ValueError(f"chunk rows must be positive, observed {chunk_rows}")
    labels_path = labels_path.resolve(strict=True)
    assay_ids_path = assay_ids_path.resolve(strict=True)
    output_dir = output_dir.resolve()

    assay_ids = load_assay_ids(assay_ids_path)
    matrix = load_label_matrix(labels_path)
    n_molecules, n_assays = map(int, matrix.shape)
    if len(assay_ids) != n_assays:
        raise ValueError(
            "assay-ID count must equal matrix columns: "
            f"ids={len(assay_ids)}, columns={n_assays}"
        )

    observed, positive, molecule_histogram = validate_and_count(matrix, chunk_rows)
    negative = observed - positive
    missing = n_molecules - observed
    total_cells = n_molecules * n_assays
    observed_total = int(observed.sum(dtype=np.int64))
    positive_total = int(positive.sum(dtype=np.int64))
    negative_total = int(negative.sum(dtype=np.int64))
    missing_total = int(missing.sum(dtype=np.int64))

    if observed_total + missing_total != total_cells:
        raise RuntimeError("internal count invariant failed: observed + missing != cells")
    if positive_total + negative_total != observed_total:
        raise RuntimeError("internal count invariant failed: positive + negative != observed")
    if int(molecule_histogram.sum(dtype=np.int64)) != n_molecules:
        raise RuntimeError("internal count invariant failed: histogram != molecule rows")

    prevalence = np.divide(
        positive,
        observed,
        out=np.full(n_assays, np.nan, dtype=np.float64),
        where=observed > 0,
    )
    label_density_by_assay = observed.astype(np.float64) / n_molecules
    constant = (observed > 0) & ((positive == 0) | (negative == 0))
    constant_observations = int(observed[constant].sum(dtype=np.int64))
    constant_value = np.where(
        ~constant, "", np.where(positive == observed, "positive", "negative")
    )

    assay_frame = pd.DataFrame(
        {
            "assay_index": np.arange(n_assays, dtype=np.int64),
            "assay_id": assay_ids,
            "observed_count": observed,
            "positive_count": positive,
            "negative_count": negative,
            "missing_count": missing,
            "label_density": label_density_by_assay,
            "positive_prevalence": prevalence,
            "is_constant_label": constant,
            "constant_label_value": constant_value,
        }
    )

    nonzero_bins = np.flatnonzero(molecule_histogram)
    histogram_frame = pd.DataFrame(
        {
            "observed_label_count": nonzero_bins.astype(np.int64),
            "molecule_count": molecule_histogram[nonzero_bins],
            "molecule_fraction": molecule_histogram[nonzero_bins].astype(np.float64)
            / n_molecules,
        }
    )
    molecule_values = np.repeat(nonzero_bins, molecule_histogram[nonzero_bins])

    zero_observation_mask = observed == 0
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit": "Step-2 sparse binary assay-label matrix",
        "generated_at_utc": utc_now(),
        "inputs": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "labels_npy": {
                "path": str(labels_path),
                "sha256": sha256_file(labels_path),
            },
            "assay_ids": {
                "path": str(assay_ids_path),
                "sha256": sha256_file(assay_ids_path),
            },
        },
        "matrix": {
            "shape": [n_molecules, n_assays],
            "molecules": n_molecules,
            "assays": n_assays,
            "dtype": str(matrix.dtype),
        },
        "totals": {
            "cells": total_cells,
            "observed": observed_total,
            "positive": positive_total,
            "negative": negative_total,
            "missing": missing_total,
            "label_density": observed_total / total_cells,
        },
        "assays": {
            "observed_count_distribution": numeric_summary(observed),
            "positive_prevalence_distribution": numeric_summary(prevalence),
            "prevalence_excludes_zero_observation_assays": True,
            "zero_observation_assays": int(zero_observation_mask.sum()),
            "zero_observation_assay_ids": [
                assay_ids[index] for index in np.flatnonzero(zero_observation_mask)
            ],
            "constant_label_assays": int(constant.sum()),
            "constant_all_negative_assays": int((constant & (positive == 0)).sum()),
            "constant_all_positive_assays": int((constant & (negative == 0)).sum()),
            "constant_label_observations": constant_observations,
            "constant_label_observation_fraction": constant_observations / observed_total,
            "constant_all_negative_observations": int(
                observed[constant & (positive == 0)].sum(dtype=np.int64)
            ),
            "constant_all_positive_observations": int(
                observed[constant & (negative == 0)].sum(dtype=np.int64)
            ),
            "nonconstant_positive_prevalence_distribution": numeric_summary(
                prevalence[~constant & (observed > 0)]
            ),
            "constant_label_assay_ids": [
                assay_ids[index] for index in np.flatnonzero(constant)
            ],
        },
        "molecules": {
            "observed_label_count_distribution": numeric_summary(molecule_values),
            "zero_label_rows": int(molecule_histogram[0]),
            "histogram_bins_present": int(len(nonzero_bins)),
        },
        "validation": {
            "status": "passed",
            "accepted_observed_values": [0, 1],
            "accepted_missing_value": "NaN for floating-point matrices",
            "assay_ids_nonblank": True,
            "assay_ids_unique": True,
            "assay_id_count_matches_columns": True,
            "count_invariants_passed": True,
        },
        "parameters": {"chunk_rows": chunk_rows, "sd_ddof": 0},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }

    results_markdown = make_results_markdown(summary)
    atomic_csv(output_dir / "per_assay_summary.csv", assay_frame)
    atomic_csv(output_dir / "per_molecule_label_count_histogram.csv", histogram_frame)
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(output_dir / "RESULTS.md", results_markdown)

    print(
        f"Validated {n_molecules:,} x {n_assays:,} matrix; "
        f"observed={observed_total:,}, density={observed_total / total_cells:.8f}"
    )
    print(f"Wrote audit artifacts to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--labels-npy", required=True, type=Path)
    parser.add_argument("--assay-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        analyze(args.labels_npy, args.assay_ids, args.output_dir, args.chunk_rows)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
