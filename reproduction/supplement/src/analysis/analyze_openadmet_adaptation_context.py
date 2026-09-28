#!/usr/bin/env python3
"""Audit how OpenADMET adaptation behavior varies with size and diversity.

The analysis uses endpoints, rather than the three repeated split decisions, as
the inferential unit.  It verifies all 207 retained split--method cells against
the canonical endpoint-row table, summarizes training-set scaffold diversity,
and relates two predeclared predictors (size and size-adjusted scaffold richness)
to frozen/full and LoRA/full validation advantages.  Test-MAE associations are
reported only as held-out corroboration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_ROWS = (
    REPO_ROOT
    / "results_package"
    / "results_openadmet_step2_leakage_audit_20260917_v3"
    / "endpoint_rows.csv.gz"
)
DEFAULT_RESULTS = REPO_ROOT / "results_rtd_phase4_95m_moljepa"
DEFAULT_OUTPUT = ROOT / "evidence" / "openadmet_adaptation_context"
METHODS = ("frozen_mix", "lora_mix", "full_mix")
BOOTSTRAP_REPLICATES = 100_000
PERMUTATION_REPLICATES = 100_000
RANDOM_SEED = 20260918


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


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            require(
                row["standardization_status"] == "ok",
                f"non-standardized endpoint row: {row['endpoint_record_id']}",
            )
            grouped[row["endpoint_slug"]].append(row)
    require(len(grouped) == 23, f"expected 23 endpoints, found {len(grouped)}")
    return dict(grouped)


def diversity(values: list[str]) -> dict[str, float | int]:
    """Summarize categorical diversity, grouping blank scaffolds as acyclic."""
    categories = [value if value else "<acyclic>" for value in values]
    counts = np.asarray(list(Counter(categories).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return {
        "unique": int(counts.size),
        "effective_shannon": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "singleton_category_fraction": float(np.mean(counts == 1)),
        "largest_category_row_fraction": float(counts.max() / counts.sum()),
        "nonempty_row_fraction": float(np.mean([bool(value) for value in values])),
    }


def bootstrap_spearman(
    predictor: np.ndarray,
    outcome: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimate_chunks: list[np.ndarray] = []
    size = predictor.size
    chunk = 5_000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(BOOTSTRAP_REPLICATES, start + chunk)
        indices = rng.integers(0, size, size=(stop - start, size))
        x_rank = stats.rankdata(predictor[indices], axis=1)
        y_rank = stats.rankdata(outcome[indices], axis=1)
        x_centered = x_rank - x_rank.mean(axis=1, keepdims=True)
        y_centered = y_rank - y_rank.mean(axis=1, keepdims=True)
        numerator = np.sum(x_centered * y_centered, axis=1)
        denominator = np.sqrt(
            np.sum(x_centered * x_centered, axis=1)
            * np.sum(y_centered * y_centered, axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate_chunks.append(numerator / denominator)
    estimates = np.concatenate(estimate_chunks)
    estimates = estimates[np.isfinite(estimates)]
    require(
        estimates.size >= int(0.999 * BOOTSTRAP_REPLICATES),
        "too many non-finite bootstrap correlations",
    )
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def permutation_p(
    predictor: np.ndarray,
    outcome: np.ndarray,
    observed: float,
    *,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    extreme = 0
    x_rank = stats.rankdata(predictor)
    y_rank = stats.rankdata(outcome)
    x_centered = x_rank - x_rank.mean()
    x_norm = np.sqrt(np.sum(x_centered * x_centered))
    chunk = 5_000
    for start in range(0, PERMUTATION_REPLICATES, chunk):
        stop = min(PERMUTATION_REPLICATES, start + chunk)
        random_keys = rng.random((stop - start, outcome.size))
        indices = np.argsort(random_keys, axis=1)
        permuted = y_rank[indices]
        y_centered = permuted - permuted.mean(axis=1, keepdims=True)
        correlations = np.sum(y_centered * x_centered, axis=1) / (
            x_norm * np.sqrt(np.sum(y_centered * y_centered, axis=1))
        )
        extreme += int(np.sum(np.abs(correlations) >= abs(observed) - 1e-15))
    return float((extreme + 1) / (PERMUTATION_REPLICATES + 1))


def association(
    endpoint_rows: list[dict[str, object]],
    predictor: str,
    outcome: str,
    family: str,
    seed_offset: int,
) -> dict[str, object]:
    x = np.asarray([float(row[predictor]) for row in endpoint_rows], dtype=np.float64)
    y = np.asarray([float(row[outcome]) for row in endpoint_rows], dtype=np.float64)
    rho = float(stats.spearmanr(x, y).statistic)
    require(math.isfinite(rho), f"non-finite association: {predictor} vs {outcome}")
    return {
        "family": family,
        "predictor": predictor,
        "outcome": outcome,
        "endpoint_count": int(x.size),
        "spearman_rho": rho,
        "bootstrap_95_interval": bootstrap_spearman(
            x, y, seed=RANDOM_SEED + seed_offset
        ),
        "permutation_replicates": PERMUTATION_REPLICATES,
        "permutation_seed": RANDOM_SEED + 100 + seed_offset,
        "permutation_raw_p": permutation_p(
            x, y, rho, seed=RANDOM_SEED + 100 + seed_offset
        ),
    }


def run(rows_path: Path, results_root: Path, output_dir: Path) -> dict[str, object]:
    require(rows_path.is_file(), f"endpoint-row input is missing: {rows_path}")
    require(results_root.is_dir(), f"result root is missing: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_rows(rows_path)

    cell_rows: list[dict[str, object]] = []
    cell_files: list[Path] = []
    endpoint_summary_files: list[Path] = []
    selection_counts = Counter()

    for endpoint in sorted(grouped):
        rows = grouped[endpoint]
        endpoint_summary_path = results_root / f"results_{endpoint}.json"
        require(endpoint_summary_path.is_file(), f"missing endpoint summary: {endpoint}")
        endpoint_summary_files.append(endpoint_summary_path)
        endpoint_summary = json.loads(endpoint_summary_path.read_text(encoding="utf-8"))
        require(endpoint_summary["task"] == endpoint, f"endpoint summary drifted: {endpoint}")

        for split_index in (1, 2, 3):
            split = f"split{split_index}"
            partition = f"{split}_neural_partition"
            partitions = {
                label: [row for row in rows if row[partition] == label]
                for label in ("train", "validation", "test")
            }
            require(all(partitions.values()), f"empty partition for {endpoint} {split}")

            method_cells: dict[str, dict[str, object]] = {}
            for method in METHODS:
                cell_path = results_root / endpoint / "cells" / f"{split}_{method}.json"
                require(cell_path.is_file(), f"missing cell: {endpoint} {split} {method}")
                cell_files.append(cell_path)
                cell = json.loads(cell_path.read_text(encoding="utf-8"))
                require(
                    cell["method"] == method
                    and cell["task"] == endpoint
                    and cell["split"] == split,
                    f"cell identity drifted for {endpoint} {split} {method}",
                )
                require(
                    cell["train_rows"] == len(partitions["train"])
                    and cell["validation_rows"] == len(partitions["validation"])
                    and cell["test_rows"] == len(partitions["test"]),
                    f"partition size drifted for {endpoint} {split} {method}",
                )
                for label, partition_rows in partitions.items():
                    observed = smiles_sha256(
                        [row["original_smiles"] for row in partition_rows]
                    )
                    require(
                        observed == cell[f"{label}_smiles_sha256"],
                        f"{label} SMILES hash drifted for {endpoint} {split} {method}",
                    )
                truth = [float(row["y"]) for row in partitions["test"]]
                require(
                    len(truth) == len(cell["y_true"]) == len(cell["y_pred"]),
                    f"prediction length drifted for {endpoint} {split} {method}",
                )
                require(
                    all(a == b for a, b in zip(truth, cell["y_true"])),
                    f"test target order drifted for {endpoint} {split} {method}",
                )
                method_cells[method] = cell

            selected_method = min(METHODS, key=lambda name: method_cells[name]["validation_mae"])
            recorded_selected = endpoint_summary["splits"][split]["selected_method"]
            require(
                selected_method == recorded_selected,
                f"selection rule drifted for {endpoint} {split}",
            )
            selection_counts[selected_method] += 1

            train = partitions["train"]
            bm = diversity([row["bemis_murcko"] for row in train])
            generic = diversity([row["generic_scaffold"] for row in train])
            cluster_count = len({row["cluster_index"] for row in train})
            full = method_cells["full_mix"]
            frozen = method_cells["frozen_mix"]
            lora = method_cells["lora_mix"]
            cell_rows.append({
                "endpoint": endpoint,
                "split": split,
                "train_rows": len(train),
                "validation_rows": len(partitions["validation"]),
                "test_rows": len(partitions["test"]),
                "unique_bemis_murcko": bm["unique"],
                "bemis_murcko_effective_shannon": bm["effective_shannon"],
                "bemis_murcko_singleton_fraction": bm["singleton_category_fraction"],
                "bemis_murcko_largest_fraction": bm["largest_category_row_fraction"],
                "bemis_murcko_nonempty_fraction": bm["nonempty_row_fraction"],
                "unique_generic_scaffolds": generic["unique"],
                "generic_effective_shannon": generic["effective_shannon"],
                "unique_train_clusters": cluster_count,
                "selected_method": selected_method,
                "frozen_validation_mae": frozen["validation_mae"],
                "lora_validation_mae": lora["validation_mae"],
                "full_validation_mae": full["validation_mae"],
                "full_minus_frozen_validation_mae": (
                    full["validation_mae"] - frozen["validation_mae"]
                ),
                "full_minus_lora_validation_mae": (
                    full["validation_mae"] - lora["validation_mae"]
                ),
                "frozen_test_mae": frozen["test_mae"],
                "lora_test_mae": lora["test_mae"],
                "full_test_mae": full["test_mae"],
                "full_minus_frozen_test_mae": full["test_mae"] - frozen["test_mae"],
                "full_minus_lora_test_mae": full["test_mae"] - lora["test_mae"],
            })

    require(len(cell_rows) == 69, "expected exactly 69 endpoint-split rows")
    require(len(cell_files) == 207, "expected exactly 207 split-method cells")
    require(
        selection_counts == {"full_mix": 39, "lora_mix": 18, "frozen_mix": 12},
        f"selection counts drifted: {dict(selection_counts)}",
    )

    numeric_fields = [
        field for field in cell_rows[0]
        if field not in {"endpoint", "split", "selected_method"}
    ]
    endpoint_rows: list[dict[str, object]] = []
    for endpoint in sorted(grouped):
        cells = [row for row in cell_rows if row["endpoint"] == endpoint]
        require(len(cells) == 3, f"expected three split rows for {endpoint}")
        endpoint_row: dict[str, object] = {"endpoint": endpoint}
        for field in numeric_fields:
            endpoint_row[f"mean_{field}"] = float(
                np.mean([float(row[field]) for row in cells])
            )
        endpoint_rows.append(endpoint_row)

    for row in endpoint_rows:
        row["mean_log1p_train_rows"] = math.log1p(float(row["mean_train_rows"]))
        row["mean_log1p_unique_bemis_murcko"] = math.log1p(
            float(row["mean_unique_bemis_murcko"])
        )
    x = np.asarray(
        [float(row["mean_log1p_train_rows"]) for row in endpoint_rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row["mean_log1p_unique_bemis_murcko"]) for row in endpoint_rows],
        dtype=np.float64,
    )
    design = np.column_stack([np.ones(x.size), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    for row, residual in zip(endpoint_rows, residuals):
        row["size_adjusted_scaffold_richness"] = float(residual)

    predictors = ("mean_log1p_train_rows", "size_adjusted_scaffold_richness")
    validation_outcomes = (
        "mean_full_minus_frozen_validation_mae",
        "mean_full_minus_lora_validation_mae",
    )
    test_outcomes = (
        "mean_full_minus_frozen_test_mae",
        "mean_full_minus_lora_test_mae",
    )
    associations: list[dict[str, object]] = []
    offset = 0
    for family, outcomes in (
        ("validation_primary", validation_outcomes),
        ("test_corroboration", test_outcomes),
    ):
        family_rows = []
        for outcome in outcomes:
            for predictor in predictors:
                family_rows.append(
                    association(endpoint_rows, predictor, outcome, family, offset)
                )
                offset += 1
        adjusted = holm_adjust(
            [float(row["permutation_raw_p"]) for row in family_rows]
        )
        for row, value in zip(family_rows, adjusted):
            row["holm4_permutation_p"] = value
        associations.extend(family_rows)

    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "exploratory endpoint-level adaptation-context analysis",
        "interpretation_boundary": (
            "Post hoc associations across 23 endpoints do not establish a causal "
            "method-selection rule; endpoint family, label noise, size, and diversity "
            "remain confounded, and only frozen mixing, LoRA, and full fine-tuning were tested."
        ),
        "endpoint_count": 23,
        "split_count": 69,
        "verified_split_method_cells": 207,
        "selection_counts": dict(sorted(selection_counts.items())),
        "training_size": {
            "cell_min": min(int(row["train_rows"]) for row in cell_rows),
            "cell_median": float(np.median([row["train_rows"] for row in cell_rows])),
            "cell_max": max(int(row["train_rows"]) for row in cell_rows),
        },
        "bemis_murcko_richness": {
            "cell_min": min(int(row["unique_bemis_murcko"]) for row in cell_rows),
            "cell_median": float(
                np.median([row["unique_bemis_murcko"] for row in cell_rows])
            ),
            "cell_max": max(int(row["unique_bemis_murcko"]) for row in cell_rows),
            "raw_size_spearman": float(stats.spearmanr(
                [row["train_rows"] for row in cell_rows],
                [row["unique_bemis_murcko"] for row in cell_rows],
            ).statistic),
            "size_adjustment": {
                "model": "endpoint mean log1p(unique BM scaffolds) ~ intercept + endpoint mean log1p(train rows)",
                "intercept": float(coefficients[0]),
                "slope": float(coefficients[1]),
            },
        },
        "association_design": {
            "analysis_unit": "endpoint mean over three reconstructed splits",
            "primary_family": "four validation-MAE associations",
            "corroboration_family": "four held-out test-MAE associations",
            "outcome_definition": "full MAE minus parameter-efficient-method MAE; positive favors the parameter-efficient method",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "base_seed": RANDOM_SEED,
            "multiplicity": "Holm correction within each four-association family",
        },
        "associations": associations,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "endpoint_rows": {
                "path": rows_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
                "sha256": sha256(rows_path),
            },
            "endpoint_summaries": [
                {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(path)}
                for path in sorted(endpoint_summary_files)
            ],
            "split_method_cells": [
                {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(path)}
                for path in sorted(cell_files)
            ],
        },
    }

    cells_path = output_dir / "cell_summary.csv"
    endpoints_path = output_dir / "endpoint_summary.csv"
    associations_path = output_dir / "associations.csv"
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "RESULTS.md"
    write_csv(cells_path, cell_rows)
    write_csv(endpoints_path, endpoint_rows)
    write_csv(associations_path, associations)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lookup = {
        (row["family"], row["predictor"], row["outcome"]): row
        for row in associations
    }
    size_frozen = lookup[(
        "validation_primary",
        "mean_log1p_train_rows",
        "mean_full_minus_frozen_validation_mae",
    )]
    diversity_lora = lookup[(
        "validation_primary",
        "size_adjusted_scaffold_richness",
        "mean_full_minus_lora_validation_mae",
    )]
    results_lines = [
        "# OpenADMET adaptation-context audit",
        "",
        "Status: **PASS**",
        "",
        (
            "All 207 retained split--method cells were matched to the canonical "
            "23-endpoint row table and passed identity, partition-size, SMILES-hash, "
            "prediction-length, and ordered-target checks."
        ),
        (
            f"Validation selection chose full fine-tuning {selection_counts['full_mix']} times, "
            f"LoRA {selection_counts['lora_mix']} times, and frozen mixing "
            f"{selection_counts['frozen_mix']} times across 69 endpoint--split decisions."
        ),
        "",
        (
            "Using endpoint means as the inferential unit, training size was associated "
            "with full-minus-frozen validation MAE at Spearman "
            f"rho={size_frozen['spearman_rho']:+.3f} "
            f"(95% bootstrap interval [{size_frozen['bootstrap_95_interval'][0]:+.3f}, "
            f"{size_frozen['bootstrap_95_interval'][1]:+.3f}]; Holm-4 permutation "
            f"p={size_frozen['holm4_permutation_p']:.4f})."
        ),
        (
            "Size-adjusted Bemis--Murcko richness was associated with full-minus-LoRA "
            f"validation MAE at rho={diversity_lora['spearman_rho']:+.3f} "
            f"(95% interval [{diversity_lora['bootstrap_95_interval'][0]:+.3f}, "
            f"{diversity_lora['bootstrap_95_interval'][1]:+.3f}]; Holm-4 permutation "
            f"p={diversity_lora['holm4_permutation_p']:.4f})."
        ),
        "",
        (
            "These post hoc associations are exploratory. They do not establish a causal "
            "adaptation rule, and they do not generalize beyond the three tested methods."
        ),
    ]
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")

    manifest_entries = []
    for path in (cells_path, endpoints_path, associations_path, summary_path, results_path):
        manifest_entries.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(
            {"schema_version": 1, "files": manifest_entries},
            indent=2,
            sort_keys=True,
        ) + "\n",
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
        "split_count": result["split_count"],
        "verified_split_method_cells": result["verified_split_method_cells"],
        "selection_counts": result["selection_counts"],
    }, indent=2, sort_keys=True))
