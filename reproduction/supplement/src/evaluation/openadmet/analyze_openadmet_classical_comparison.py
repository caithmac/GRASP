#!/usr/bin/env python3
"""Paired endpoint-level comparison of ChemRasayan and classical OpenADMET models.

The inferential unit is an endpoint.  Each endpoint score is first averaged over
the same three reconstructed cluster splits, then the 23 paired endpoint means
are compared.  Split runs are never treated as independent replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_METHODS = (
    "dummy_median",
    "ecfp4_rf",
    "ecfp4_lgbm",
    "rdkit_desc_rf",
    "rdkit_desc_lgbm",
)
EXPECTED_SPLITS = ("split1", "split2", "split3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, using exact half-ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def exact_wilcoxon_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    """Exact sign-randomization Wilcoxon test, dropping exact zero pairs.

    A dynamic program enumerates the null distribution of the positive rank sum.
    Ranks are doubled to represent average half-ranks as integers.  The reported
    two-sided p-value is twice the smaller tail, capped at one.
    """
    raw = np.asarray(list(differences), dtype=float)
    nonzero = raw[raw != 0.0]
    if len(nonzero) == 0:
        return {
            "n_nonzero": 0,
            "zero_pairs": int(len(raw)),
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_two_sided_exact": 1.0,
            "definition": "exact sign-randomization distribution; doubled smaller tail",
        }
    doubled_ranks = np.rint(average_ranks(np.abs(nonzero)) * 2).astype(int)
    counts: dict[int, int] = {0: 1}
    for rank in doubled_ranks:
        updated = counts.copy()
        for subtotal, count in counts.items():
            updated[subtotal + int(rank)] = updated.get(subtotal + int(rank), 0) + count
        counts = updated
    observed_plus = int(doubled_ranks[nonzero > 0].sum())
    total_rank = int(doubled_ranks.sum())
    lower_boundary = min(observed_plus, total_rank - observed_plus)
    lower_count = sum(count for score, count in counts.items() if score <= lower_boundary)
    p_value = min(1.0, 2.0 * lower_count / (2 ** len(nonzero)))
    return {
        "n_nonzero": int(len(nonzero)),
        "zero_pairs": int(len(raw) - len(nonzero)),
        "w_plus": observed_plus / 2.0,
        "w_minus": (total_rank - observed_plus) / 2.0,
        "statistic": lower_boundary / 2.0,
        "p_two_sided_exact": float(p_value),
        "definition": "exact sign-randomization distribution; doubled smaller tail",
    }


def exact_sign_test_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    raw = np.asarray(list(differences), dtype=float)
    positives = int(np.sum(raw > 0.0))
    negatives = int(np.sum(raw < 0.0))
    nonzero = positives + negatives
    if nonzero == 0:
        p_value = 1.0
    else:
        smaller = min(positives, negatives)
        lower = sum(math.comb(nonzero, i) for i in range(smaller + 1)) / (2 ** nonzero)
        p_value = min(1.0, 2.0 * lower)
    return {
        "n_nonzero": nonzero,
        "positive_differences": positives,
        "negative_differences": negatives,
        "zero_pairs": int(len(raw) - nonzero),
        "p_two_sided_exact": float(p_value),
        "definition": "exact Binomial(n, 0.5); doubled smaller tail; zero pairs omitted",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def percentile_bootstrap_mean_ci(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical-dir",
        type=Path,
        default=Path("results_package/results_openadmet_same_split_baselines_20260917_r2"),
    )
    parser.add_argument(
        "--chemrasayan-dir",
        type=Path,
        default=Path("results_rtd_phase4_95m_moljepa"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_iclr/evidence/openadmet_classical_comparison"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_917)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")

    audit_path = args.classical_dir / "audited_summary" / "audit.json"
    cells_path = args.classical_dir / "audited_summary" / "cells.csv"
    endpoint_summary_path = args.classical_dir / "audited_summary" / "endpoint_summary.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError(f"Classical audit did not pass: {audit.get('status')!r}")
    if sha256_file(cells_path) != audit["cells_csv_sha256"]:
        raise ValueError("Classical cells.csv hash does not match audit.json")
    if sha256_file(endpoint_summary_path) != audit["endpoint_summary_csv_sha256"]:
        raise ValueError("Classical endpoint_summary.csv hash does not match audit.json")
    if not audit.get("neural_split_hashes_verified"):
        raise ValueError("Classical audit did not verify neural split hashes")
    if not audit.get("prediction_maes_recomputed"):
        raise ValueError("Classical audit did not recompute prediction MAEs")

    cells = read_csv(cells_path)
    audited_endpoint_summary = read_csv(endpoint_summary_path)
    methods = tuple(audit["expected_methods"])
    if set(methods) != set(DEFAULT_METHODS):
        raise ValueError(f"Unexpected classical method set: {methods}")
    splits = tuple(audit["expected_splits"])
    if splits != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split set: {splits}")
    key_counts = Counter((r["endpoint"], r["split"], r["method"]) for r in cells)
    duplicate_keys = [key for key, count in key_counts.items() if count != 1]
    if duplicate_keys:
        raise ValueError(f"Non-unique classical cells: {duplicate_keys[:5]}")
    endpoints = sorted({r["endpoint"] for r in cells})
    if len(endpoints) != audit["expected_endpoints"]:
        raise ValueError("Endpoint count differs from audit contract")
    expected_keys = {(e, s, m) for e in endpoints for s in splits for m in methods}
    if set(key_counts) != expected_keys:
        missing = sorted(expected_keys - set(key_counts))
        extra = sorted(set(key_counts) - expected_keys)
        raise ValueError(f"Incomplete classical cell grid; missing={missing[:5]}, extra={extra[:5]}")
    cell_by_key = {(r["endpoint"], r["split"], r["method"]): r for r in cells}
    endpoint_summary_by_key = {
        (r["endpoint"], r["method"]): r for r in audited_endpoint_summary
    }
    if len(endpoint_summary_by_key) != len(endpoints) * len(methods):
        raise ValueError("Audited endpoint summary is not a complete endpoint-method grid")
    for endpoint in endpoints:
        for method in methods:
            recomputed_mean = float(
                np.mean([float(cell_by_key[(endpoint, split, method)]["test_mae"]) for split in splits])
            )
            reported_mean = float(endpoint_summary_by_key[(endpoint, method)]["mean_test_mae"])
            if not math.isclose(recomputed_mean, reported_mean, abs_tol=1e-12):
                raise ValueError(f"Audited endpoint mean mismatch: {endpoint}/{method}")

    chem_files = sorted(args.chemrasayan_dir.glob("results_*.json"))
    chem_by_endpoint: dict[str, dict[str, Any]] = {}
    chem_input_hashes: dict[str, str] = {}
    for path in chem_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        endpoint = payload["task"]
        if endpoint in chem_by_endpoint:
            raise ValueError(f"Duplicate ChemRasayan endpoint: {endpoint}")
        chem_by_endpoint[endpoint] = payload
        chem_input_hashes[path.name] = sha256_file(path)
    if set(chem_by_endpoint) != set(endpoints):
        raise ValueError(
            "Endpoint sets differ: "
            f"classical_only={sorted(set(endpoints) - set(chem_by_endpoint))}, "
            f"chemrasayan_only={sorted(set(chem_by_endpoint) - set(endpoints))}"
        )

    primary_report_path = args.chemrasayan_dir / "comparison" / "moljepa_comparison_report.json"
    primary_generator_path = Path("summarize_moljepa_benchmarks.py")
    primary_report = (
        json.loads(primary_report_path.read_text(encoding="utf-8"))
        if primary_report_path.exists()
        else None
    )

    hash_checks = 0
    direct_prediction_files = 0
    prediction_files_missing = 0
    prediction_files_hash_mismatched: list[str] = []
    directly_checked_endpoint_splits: set[tuple[str, str]] = set()
    chem_split_mae: dict[tuple[str, str], float] = {}
    family_by_endpoint: dict[str, str] = {}
    for endpoint in endpoints:
        payload = chem_by_endpoint[endpoint]
        family_by_endpoint[endpoint] = payload["family"]
        classical_families = {r["family"] for r in cells if r["endpoint"] == endpoint}
        if classical_families != {payload["family"]}:
            raise ValueError(f"Endpoint family mismatch for {endpoint}: {classical_families}")
        if payload.get("primary_metric") != "mae":
            raise ValueError(f"Primary metric is not MAE for {endpoint}")
        if payload.get("data_protocol") != "reconstructed_butina_public_snapshot":
            raise ValueError(f"Unexpected data protocol for {endpoint}")
        if set(payload["splits"]) != set(splits):
            raise ValueError(f"ChemRasayan split set differs for {endpoint}")
        for split in splits:
            chem_cell = payload["splits"][split]["methods"]["full_mix"]
            truth = np.asarray(chem_cell["y_true"], dtype=float)
            prediction = np.asarray(chem_cell["y_pred"], dtype=float)
            if len(truth) != len(prediction) or len(truth) != int(chem_cell["test_rows"]):
                raise ValueError(f"ChemRasayan prediction length mismatch: {endpoint}/{split}")
            recomputed = float(np.mean(np.abs(truth - prediction)))
            if not math.isclose(recomputed, float(chem_cell["test_mae"]), abs_tol=1e-12):
                raise ValueError(f"ChemRasayan MAE mismatch: {endpoint}/{split}")
            chem_split_mae[(endpoint, split)] = recomputed

            for method in methods:
                classical_cell = cell_by_key[(endpoint, split, method)]
                comparisons = (
                    ("data_sha256", "data_sha256"),
                    ("inner_train_smiles_sha256", "train_smiles_sha256"),
                    ("validation_smiles_sha256", "validation_smiles_sha256"),
                    ("test_smiles_sha256", "test_smiles_sha256"),
                )
                for classical_name, chem_name in comparisons:
                    if classical_cell[classical_name] != chem_cell[chem_name]:
                        raise ValueError(
                            f"Split alignment mismatch ({classical_name}): "
                            f"{endpoint}/{split}/{method}"
                        )
                    hash_checks += 1
                if int(classical_cell["n_test"]) != int(chem_cell["test_rows"]):
                    raise ValueError(f"Test-size mismatch: {endpoint}/{split}/{method}")

                prediction_path = (
                    args.classical_dir / "predictions" / endpoint / f"{split}_{method}.csv"
                )
                if prediction_path.exists():
                    if sha256_file(prediction_path) != classical_cell["predictions_sha256"]:
                        # A partially retrieved prediction file is not an inferential input.  Record
                        # it explicitly and rely on the hash-locked PASS audit of cells.csv.
                        prediction_files_hash_mismatched.append(str(prediction_path))
                    else:
                        prediction_rows = read_csv(prediction_path)
                        classical_truth = np.asarray([float(r["y_true"]) for r in prediction_rows])
                        classical_prediction = np.asarray([float(r["y_pred"]) for r in prediction_rows])
                        if len(classical_truth) != len(truth) or not np.array_equal(classical_truth, truth):
                            raise ValueError(f"Direct target alignment mismatch: {endpoint}/{split}/{method}")
                        classical_mae = float(np.mean(np.abs(classical_truth - classical_prediction)))
                        if not math.isclose(
                            classical_mae, float(classical_cell["test_mae"]), abs_tol=1e-12
                        ):
                            raise ValueError(f"Classical MAE mismatch: {endpoint}/{split}/{method}")
                        direct_prediction_files += 1
                        directly_checked_endpoint_splits.add((endpoint, split))
                else:
                    prediction_files_missing += 1

    endpoint_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    comparison_json: list[dict[str, Any]] = []
    raw_wilcoxon: list[float] = []
    raw_sign: list[float] = []
    per_method_intermediate: list[dict[str, Any]] = []

    for method_index, method in enumerate(methods):
        differences = []
        chem_means = []
        baseline_means = []
        for endpoint in endpoints:
            chem_values = np.asarray(
                [chem_split_mae[(endpoint, split)] for split in splits], dtype=float
            )
            baseline_values = np.asarray(
                [float(cell_by_key[(endpoint, split, method)]["test_mae"]) for split in splits],
                dtype=float,
            )
            chem_mean = float(chem_values.mean())
            baseline_mean = float(baseline_values.mean())
            difference = chem_mean - baseline_mean
            differences.append(difference)
            chem_means.append(chem_mean)
            baseline_means.append(baseline_mean)
            winner = "ChemRasayan" if difference < 0 else method if difference > 0 else "tie"
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "family": family_by_endpoint[endpoint],
                    "classical_method": method,
                    "chemrasayan_full_mix_mean_mae": chem_mean,
                    "classical_mean_mae": baseline_mean,
                    "paired_difference_chemrasayan_minus_classical": difference,
                    "point_estimate_winner": winner,
                }
            )

        difference_array = np.asarray(differences)
        chem_array = np.asarray(chem_means)
        baseline_array = np.asarray(baseline_means)
        wilcoxon = exact_wilcoxon_two_sided(difference_array)
        sign = exact_sign_test_two_sided(difference_array)
        ci_low, ci_high = percentile_bootstrap_mean_ci(
            difference_array,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + method_index,
        )
        raw_wilcoxon.append(wilcoxon["p_two_sided_exact"])
        raw_sign.append(sign["p_two_sided_exact"])
        per_method_intermediate.append(
            {
                "method": method,
                "chemrasayan_mean_mae": float(chem_array.mean()),
                "classical_mean_mae": float(baseline_array.mean()),
                "mean_difference": float(difference_array.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "chemrasayan_point_wins": int(np.sum(difference_array < 0)),
                "classical_point_wins": int(np.sum(difference_array > 0)),
                "ties": int(np.sum(difference_array == 0)),
                "wilcoxon": wilcoxon,
                "sign_test": sign,
            }
        )

    wilcoxon_holm = holm_adjust(raw_wilcoxon)
    sign_holm = holm_adjust(raw_sign)
    overall_chemrasayan_mean = per_method_intermediate[0]["chemrasayan_mean_mae"]
    if primary_report is not None:
        report_mean = float(primary_report["fixed_method_summary_all_23"]["full_mix"]["average_mae"])
        if not math.isclose(overall_chemrasayan_mean, report_mean, abs_tol=1e-12):
            raise ValueError("Recomputed ChemRasayan aggregate differs from primary comparison report")
    for index, item in enumerate(per_method_intermediate):
        item["wilcoxon"]["p_holm_five_contrasts"] = wilcoxon_holm[index]
        item["sign_test"]["p_holm_five_contrasts"] = sign_holm[index]
        item["wilcoxon"]["reject_holm_0_05"] = wilcoxon_holm[index] < 0.05
        item["sign_test"]["reject_holm_0_05"] = sign_holm[index] < 0.05
        comparison_json.append(item)
        comparison_rows.append(
            {
                "classical_method": item["method"],
                "n_endpoints": len(endpoints),
                "chemrasayan_mean_mae": item["chemrasayan_mean_mae"],
                "classical_mean_mae": item["classical_mean_mae"],
                "mean_difference_chemrasayan_minus_classical": item["mean_difference"],
                "bootstrap_95pct_ci_low": item["bootstrap_ci_low"],
                "bootstrap_95pct_ci_high": item["bootstrap_ci_high"],
                "chemrasayan_point_wins": item["chemrasayan_point_wins"],
                "classical_point_wins": item["classical_point_wins"],
                "ties": item["ties"],
                "wilcoxon_w": item["wilcoxon"]["statistic"],
                "wilcoxon_p_exact_two_sided": item["wilcoxon"]["p_two_sided_exact"],
                "wilcoxon_p_holm": item["wilcoxon"]["p_holm_five_contrasts"],
                "sign_p_exact_two_sided": item["sign_test"]["p_two_sided_exact"],
                "sign_p_holm": item["sign_test"]["p_holm_five_contrasts"],
            }
        )

    validation = {
        "status": "PASS",
        "endpoint_count": len(endpoints),
        "split_count_per_endpoint": len(splits),
        "classical_method_count": len(methods),
        "classical_cells": len(cells),
        "membership_hash_comparisons_passed": hash_checks,
        "aligned_endpoint_split_pairs": len(endpoints) * len(splits),
        "direct_prediction_files_recomputed_and_target_matched": direct_prediction_files,
        "retrieved_prediction_files_missing": prediction_files_missing,
        "retrieved_prediction_files_hash_mismatched": prediction_files_hash_mismatched,
        "endpoint_split_pairs_with_direct_target_match": len(directly_checked_endpoint_splits),
        "note": (
            "All endpoint/split memberships were aligned cryptographically. Direct y_true "
            "array equality was additionally checked only for retrieved prediction CSVs whose "
            "content matched the hash locked in audited cells.csv; missing/partial raw files are "
            "reported and are not used by this comparison."
        ),
    }
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "classical_audit": str(audit_path.resolve()),
        "classical_audit_sha256": sha256_file(audit_path),
        "classical_cells": str(cells_path.resolve()),
        "classical_cells_sha256": sha256_file(cells_path),
        "classical_endpoint_summary_sha256": sha256_file(endpoint_summary_path),
        "chemrasayan_result_sha256": chem_input_hashes,
        "chemrasayan_primary_report": (
            str(primary_report_path.resolve()) if primary_report_path.exists() else None
        ),
        "chemrasayan_primary_report_sha256": (
            sha256_file(primary_report_path) if primary_report_path.exists() else None
        ),
        "chemrasayan_primary_generator": (
            str(primary_generator_path.resolve()) if primary_generator_path.exists() else None
        ),
        "chemrasayan_primary_generator_sha256": (
            sha256_file(primary_generator_path) if primary_generator_path.exists() else None
        ),
        "bootstrap": {
            "unit": "endpoint",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "interval": "two-sided percentile, 2.5% and 97.5%",
        },
        "inferential_unit": (
            "23 endpoint-level paired means; each method's endpoint mean averages the same "
            "three reconstructed cluster splits"
        ),
        "multiplicity": (
            "Holm adjustment across the five classical contrasts, separately for the exact "
            "Wilcoxon and exact sign-test p-value families"
        ),
    }
    summary = {
        "schema_version": 1,
        "validation": validation,
        "provenance": provenance,
        "contrast_direction": "ChemRasayan full_mix MAE minus classical method MAE; negative favors ChemRasayan",
        "comparisons": comparison_json,
    }

    endpoint_fields = [
        "endpoint",
        "family",
        "classical_method",
        "chemrasayan_full_mix_mean_mae",
        "classical_mean_mae",
        "paired_difference_chemrasayan_minus_classical",
        "point_estimate_winner",
    ]
    comparison_fields = [
        "classical_method",
        "n_endpoints",
        "chemrasayan_mean_mae",
        "classical_mean_mae",
        "mean_difference_chemrasayan_minus_classical",
        "bootstrap_95pct_ci_low",
        "bootstrap_95pct_ci_high",
        "chemrasayan_point_wins",
        "classical_point_wins",
        "ties",
        "wilcoxon_w",
        "wilcoxon_p_exact_two_sided",
        "wilcoxon_p_holm",
        "sign_p_exact_two_sided",
        "sign_p_holm",
    ]
    atomic_csv(args.output_dir / "endpoint_paired_differences.csv", endpoint_rows, endpoint_fields)
    atomic_csv(args.output_dir / "comparisons.csv", comparison_rows, comparison_fields)
    atomic_json(args.output_dir / "summary.json", summary)

    md = [
        "# Same-split OpenADMET classical comparison",
        "",
        "**Validation: PASS.** ChemRasayan `full_mix` and every classical method use the same "
        "23 endpoints and the same three reconstructed cluster splits. Data and train/validation/test "
        "membership hashes and test sizes agree for every paired cell.",
        "",
        "The analysis unit is the endpoint: each score below first averages the three split MAEs, "
        "then compares the 23 paired endpoint means. The three splits are **not** treated as "
        "independent observations. Difference = ChemRasayan MAE - classical MAE, so negative is better.",
        "",
        "| Classical method | ChemRasayan | Classical | Mean difference (95% endpoint bootstrap CI) | Point wins | Exact Wilcoxon p / Holm | Exact sign p / Holm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        md.append(
            f"| `{row['classical_method']}` | {row['chemrasayan_mean_mae']:.4f} | "
            f"{row['classical_mean_mae']:.4f} | "
            f"{row['mean_difference_chemrasayan_minus_classical']:+.4f} "
            f"[{row['bootstrap_95pct_ci_low']:+.4f}, {row['bootstrap_95pct_ci_high']:+.4f}] | "
            f"{row['chemrasayan_point_wins']}/{row['n_endpoints']} | "
            f"{row['wilcoxon_p_exact_two_sided']:.6g} / {row['wilcoxon_p_holm']:.6g} | "
            f"{row['sign_p_exact_two_sided']:.6g} / {row['sign_p_holm']:.6g} |"
        )
    md.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Point-estimate wins only count the direction of the 23 endpoint means; they are not themselves inferential claims.",
            "- The Wilcoxon test evaluates paired signed ranks; the sign test evaluates directions only. Both are exact two-sided tests. Holm adjustment is applied across the five classical contrasts separately for each test family.",
            f"- Bootstrap intervals are deterministic percentile intervals from {args.bootstrap_replicates:,} paired endpoint resamples (base seed {args.bootstrap_seed}). They quantify endpoint heterogeneity, not training-seed uncertainty.",
            "- This is a same-reconstructed-split comparison. It does not make cross-paper claims about unavailable author-defined splits.",
            "",
            "## Alignment audit",
            "",
            f"- {validation['classical_cells']} audited classical cells; audit status PASS.",
            f"- {validation['aligned_endpoint_split_pairs']} endpoint-split pairs aligned by cryptographic data and membership hashes.",
            f"- {validation['direct_prediction_files_recomputed_and_target_matched']} retrieved classical prediction files additionally had their MAE recomputed and target arrays matched directly ({validation['endpoint_split_pairs_with_direct_target_match']}/69 unique endpoint-split pairs covered).",
            f"- {validation['retrieved_prediction_files_missing']} raw prediction files were absent and {len(validation['retrieved_prediction_files_hash_mismatched'])} present file(s) failed their recorded hash; these partial-retrieval files were excluded. The comparison uses the complete, hash-verified, PASS-audited `cells.csv`.",
            "",
            "Machine-readable outputs: `summary.json`, `comparisons.csv`, and `endpoint_paired_differences.csv`.",
        ]
    )
    atomic_text(args.output_dir / "RESULTS.md", "\n".join(md) + "\n")
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir), "comparisons": comparison_rows}, indent=2))


if __name__ == "__main__":
    main()
