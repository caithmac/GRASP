#!/usr/bin/env python3
"""Recompute OpenADMET MAE after excluding test structures exposed in Step 2.

This is a post-hoc sensitivity analysis, not a retraining experiment.  It joins
the fail-closed Step-2 structure audit to the retained per-example predictions,
verifies row order from the stored targets, and reports both exact standardized
structure and connectivity-InChIKey exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STRUCTURAL_KEY_TYPES = ("standardized_exact", "connectivity_inchikey")
NEIGHBOR_KEY_COLUMNS = {
    "tanimoto_ge_0_70": "hit_ge_0_70",
    "tanimoto_ge_0_80": "hit_ge_0_80",
    "tanimoto_ge_0_90": "hit_ge_0_90",
    "tanimoto_ge_0_95": "hit_ge_0_95",
}
KEY_TYPES = STRUCTURAL_KEY_TYPES + tuple(NEIGHBOR_KEY_COLUMNS)
METHODS = ("full_mix", "lora_mix", "frozen_mix", "validation_selected")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260917) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(100_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-rows", type=Path, required=True)
    parser.add_argument("--query-counts", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--classical-predictions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_rows = pd.read_csv(args.endpoint_rows)
    query_counts = pd.read_csv(args.query_counts)
    required_endpoint_columns = {
        "endpoint_slug", "endpoint_row_index", "query_id", "y", "split1", "split2", "split3"
    }
    if not required_endpoint_columns.issubset(endpoint_rows.columns):
        raise RuntimeError("endpoint-row audit is missing required columns")
    if set(query_counts.columns) != {"query_id", "key_type", "step2_match_count"}:
        raise RuntimeError("unexpected query-count schema")
    if not set(query_counts["key_type"]).issubset(set(STRUCTURAL_KEY_TYPES) | {"bemis_murcko", "generic_scaffold"}):
        raise RuntimeError("unexpected overlap key type")

    counts = query_counts.pivot_table(
        index="query_id", columns="key_type", values="step2_match_count", aggfunc="sum", fill_value=0
    )
    for key_type in STRUCTURAL_KEY_TYPES:
        if key_type not in counts:
            counts[key_type] = 0
    count_columns = {
        key_type: f"{key_type}_step2_match_count" for key_type in STRUCTURAL_KEY_TYPES
    }
    counts = counts[list(STRUCTURAL_KEY_TYPES)].rename(columns=count_columns)
    endpoint_rows = endpoint_rows.join(counts, on="query_id")
    endpoint_rows[list(count_columns.values())] = (
        endpoint_rows[list(count_columns.values())].fillna(0).astype(int)
    )
    missing_neighbor_columns = set(NEIGHBOR_KEY_COLUMNS.values()) - set(endpoint_rows.columns)
    if missing_neighbor_columns:
        raise RuntimeError(
            "endpoint-row audit lacks final nearest-neighbour fields: "
            f"{sorted(missing_neighbor_columns)}"
        )
    exposure_columns = {**count_columns, **NEIGHBOR_KEY_COLUMNS}

    records: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    result_paths = sorted(args.results_dir.glob("results_*.json"))
    if len(result_paths) != 23:
        raise RuntimeError(f"expected 23 result JSONs, found {len(result_paths)}")

    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        endpoint = payload["task"]
        if result_path.stem != f"results_{endpoint}":
            raise RuntimeError(f"task/file mismatch in {result_path}")
        source_hashes[result_path.name] = sha256(result_path)
        endpoint_frame = endpoint_rows[endpoint_rows["endpoint_slug"] == endpoint].copy()
        if len(endpoint_frame) == 0:
            raise RuntimeError(f"audit contains no rows for {endpoint}")

        for split_name in ("split1", "split2", "split3"):
            test_frame = endpoint_frame[endpoint_frame[split_name] == "test"].copy()
            split_payload = payload["splits"][split_name]
            selected_method = split_payload["selected_method"]
            for method in METHODS:
                stored_method = selected_method if method == "validation_selected" else method
                cell = split_payload["methods"][stored_method]
                y_true = np.asarray(cell["y_true"], dtype=float)
                y_pred = np.asarray(cell["y_pred"], dtype=float)
                expected = test_frame["y"].to_numpy(dtype=float)
                if len(y_true) != len(test_frame) or len(y_pred) != len(test_frame):
                    raise RuntimeError(f"{endpoint}/{split_name}/{method}: row-count mismatch")
                max_delta = float(np.max(np.abs(y_true - expected))) if len(y_true) else 0.0
                if max_delta > 1e-12:
                    raise RuntimeError(
                        f"{endpoint}/{split_name}/{method}: stored target/order mismatch ({max_delta})"
                    )
                original = mae(y_true, y_pred)
                if abs(original - float(cell["test_mae"])) > 1e-12:
                    raise RuntimeError(f"{endpoint}/{split_name}/{method}: stored MAE mismatch")

                for key_type in KEY_TYPES:
                    exposed = test_frame[exposure_columns[key_type]].to_numpy(dtype=int) > 0
                    retained = ~exposed
                    if not retained.any():
                        raise RuntimeError(f"{endpoint}/{split_name}/{key_type}: no unexposed rows remain")
                    records.append(
                        {
                            "endpoint": endpoint,
                            "family": payload["family"],
                            "split": split_name,
                            "model_source": "ChemRasayan",
                            "method": method,
                            "stored_method": stored_method,
                            "key_type": key_type,
                            "n_test": int(len(test_frame)),
                            "n_exposed": int(exposed.sum()),
                            "fraction_exposed": float(exposed.mean()),
                            "n_unexposed": int(retained.sum()),
                            "all_test_mae": original,
                            "exposed_only_mae": mae(y_true[exposed], y_pred[exposed]) if exposed.any() else np.nan,
                            "unexposed_only_mae": mae(y_true[retained], y_pred[retained]),
                            "unexposed_minus_all_mae": mae(y_true[retained], y_pred[retained]) - original,
                        }
                    )

    if args.classical_predictions_dir is not None:
        classical_paths = sorted(args.classical_predictions_dir.glob("*/*.csv"))
        if len(classical_paths) != 23 * 3 * 5:
            raise RuntimeError(f"expected 345 classical prediction CSVs, found {len(classical_paths)}")
        for prediction_path in classical_paths:
            prediction = pd.read_csv(prediction_path)
            required = {"endpoint", "split", "method", "row_index", "y_true", "y_pred"}
            if not required.issubset(prediction.columns):
                raise RuntimeError(f"malformed classical prediction file: {prediction_path}")
            endpoint_values = prediction["endpoint"].unique()
            split_values = prediction["split"].unique()
            method_values = prediction["method"].unique()
            if len(endpoint_values) != 1 or len(split_values) != 1 or len(method_values) != 1:
                raise RuntimeError(f"mixed classical prediction file: {prediction_path}")
            endpoint = str(endpoint_values[0])
            split_name = str(split_values[0])
            method = str(method_values[0])
            endpoint_frame = endpoint_rows[endpoint_rows["endpoint_slug"] == endpoint].copy()
            test_frame = endpoint_frame[endpoint_frame[split_name] == "test"].copy()
            if prediction["row_index"].duplicated().any():
                raise RuntimeError(f"duplicate row indices in {prediction_path}")
            joined = test_frame.merge(
                prediction[["row_index", "y_true", "y_pred"]],
                left_on="endpoint_row_index",
                right_on="row_index",
                how="left",
                validate="one_to_one",
            )
            if joined["y_pred"].isna().any() or len(joined) != len(test_frame):
                raise RuntimeError(f"incomplete row-index join for {prediction_path}")
            y_true = joined["y_true"].to_numpy(dtype=float)
            y_pred = joined["y_pred"].to_numpy(dtype=float)
            expected = joined["y"].to_numpy(dtype=float)
            max_delta = float(np.max(np.abs(y_true - expected))) if len(y_true) else 0.0
            if max_delta > 1e-12:
                raise RuntimeError(f"target mismatch in {prediction_path}: {max_delta}")
            original = mae(y_true, y_pred)
            for key_type in KEY_TYPES:
                exposed = joined[exposure_columns[key_type]].to_numpy(dtype=int) > 0
                retained = ~exposed
                if not retained.any():
                    raise RuntimeError(f"{endpoint}/{split_name}/{key_type}: no unexposed rows remain")
                records.append(
                    {
                        "endpoint": endpoint,
                        "family": endpoint_frame["endpoint_slug"].map(
                            lambda value: "ExpansionRx" if value.startswith("expansion_") else
                            ("Biogen" if value.startswith("biogen_") else
                             ("ASAP" if value.startswith("asap_") else "PXR"))
                        ).iloc[0],
                        "split": split_name,
                        "model_source": "classical",
                        "method": method,
                        "stored_method": method,
                        "key_type": key_type,
                        "n_test": int(len(joined)),
                        "n_exposed": int(exposed.sum()),
                        "fraction_exposed": float(exposed.mean()),
                        "n_unexposed": int(retained.sum()),
                        "all_test_mae": original,
                        "exposed_only_mae": mae(y_true[exposed], y_pred[exposed]) if exposed.any() else np.nan,
                        "unexposed_only_mae": mae(y_true[retained], y_pred[retained]),
                        "unexposed_minus_all_mae": mae(y_true[retained], y_pred[retained]) - original,
                    }
                )

    detail = pd.DataFrame.from_records(records)
    detail.to_csv(args.output_dir / "overlap_filtered_split_metrics.csv", index=False)

    endpoint_summary = (
        detail.groupby(["endpoint", "family", "model_source", "method", "key_type"], as_index=False)
        .agg(
            mean_all_test_mae=("all_test_mae", "mean"),
            mean_unexposed_only_mae=("unexposed_only_mae", "mean"),
            mean_unexposed_minus_all_mae=("unexposed_minus_all_mae", "mean"),
            mean_fraction_exposed=("fraction_exposed", "mean"),
            min_unexposed_rows=("n_unexposed", "min"),
        )
    )
    endpoint_summary.to_csv(args.output_dir / "overlap_filtered_endpoint_metrics.csv", index=False)

    aggregate: dict[str, dict[str, object]] = {}
    for method in sorted(detail["method"].unique()):
        aggregate[method] = {}
        for key_type in KEY_TYPES:
            subset = endpoint_summary[
                (endpoint_summary["method"] == method) & (endpoint_summary["key_type"] == key_type)
            ]
            if len(subset) != 23:
                raise RuntimeError(f"{method}/{key_type}: expected 23 endpoint summaries")
            deltas = subset["mean_unexposed_minus_all_mae"].to_numpy(dtype=float)
            aggregate[method][key_type] = {
                "endpoints": 23,
                "mean_endpoint_all_test_mae": float(subset["mean_all_test_mae"].mean()),
                "mean_endpoint_unexposed_only_mae": float(subset["mean_unexposed_only_mae"].mean()),
                "mean_endpoint_delta_unexposed_minus_all": float(deltas.mean()),
                "bootstrap_95pct_ci_for_mean_endpoint_delta": bootstrap_mean_ci(deltas),
                "endpoints_with_any_exposed_test_row": int((subset["mean_fraction_exposed"] > 0).sum()),
                "endpoints_unexposed_mae_lower": int((deltas < 0).sum()),
                "endpoints_unexposed_mae_higher": int((deltas > 0).sum()),
                "endpoints_unchanged": int((deltas == 0).sum()),
                "minimum_retained_test_rows_in_any_split": int(subset["min_unexposed_rows"].min()),
            }

    comparisons: dict[str, dict[str, object]] = {}
    if "ecfp4_lgbm" in aggregate:
        for key_type in KEY_TYPES:
            chem = endpoint_summary[
                (endpoint_summary["method"] == "full_mix") &
                (endpoint_summary["key_type"] == key_type)
            ][["endpoint", "mean_all_test_mae", "mean_unexposed_only_mae"]]
            base = endpoint_summary[
                (endpoint_summary["method"] == "ecfp4_lgbm") &
                (endpoint_summary["key_type"] == key_type)
            ][["endpoint", "mean_all_test_mae", "mean_unexposed_only_mae"]]
            paired = chem.merge(base, on="endpoint", suffixes=("_chemrasayan", "_baseline"), validate="one_to_one")
            original_delta = (
                paired["mean_all_test_mae_chemrasayan"] - paired["mean_all_test_mae_baseline"]
            ).to_numpy(dtype=float)
            filtered_delta = (
                paired["mean_unexposed_only_mae_chemrasayan"] -
                paired["mean_unexposed_only_mae_baseline"]
            ).to_numpy(dtype=float)
            comparisons[key_type] = {
                "baseline": "ecfp4_lgbm",
                "mean_endpoint_original_chemrasayan_minus_baseline": float(original_delta.mean()),
                "mean_endpoint_unexposed_chemrasayan_minus_baseline": float(filtered_delta.mean()),
                "bootstrap_95pct_ci_for_unexposed_difference": bootstrap_mean_ci(filtered_delta, seed=20260918),
                "chemrasayan_unexposed_wins": int((filtered_delta < 0).sum()),
                "baseline_unexposed_wins": int((filtered_delta > 0).sum()),
            }

    manifest = {
        "schema_version": 1,
        "analysis": "post-hoc OpenADMET test sensitivity after Step-2 structural-overlap exclusion",
        "boundary": (
            "No model is retrained. The analysis removes exposed test rows only; it does not remove "
            "scaffold, assay-semantic, or Step-1 ZINC exposure."
        ),
        "inputs": {
            "endpoint_rows": {"path": str(args.endpoint_rows), "sha256": sha256(args.endpoint_rows)},
            "query_counts": {"path": str(args.query_counts), "sha256": sha256(args.query_counts)},
            "result_json_sha256": source_hashes,
        },
        "aggregate": aggregate,
        "paired_comparison": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    exact = aggregate["full_mix"]["standardized_exact"]
    connectivity = aggregate["full_mix"]["connectivity_inchikey"]
    lines = [
        "# Step-2 overlap-filtered OpenADMET sensitivity",
        "",
        "This is a post-hoc test-row exclusion analysis; models were not retrained.",
        "",
        "| Exclusion | Original mean endpoint MAE | Unexposed-only mean endpoint MAE | Delta | Endpoints with exposed test rows | 95% bootstrap CI for delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display_keys = (
        ("Standardized exact", "standardized_exact"),
        ("Connectivity InChIKey", "connectivity_inchikey"),
        ("ECFP4 Tanimoto >= 0.90", "tanimoto_ge_0_90"),
        ("ECFP4 Tanimoto >= 0.70", "tanimoto_ge_0_70"),
    )
    for label, key_type in display_keys:
        item = aggregate["full_mix"][key_type]
        ci = item["bootstrap_95pct_ci_for_mean_endpoint_delta"]
        lines.append(
            f"| {label} | {item['mean_endpoint_all_test_mae']:.6f} | "
            f"{item['mean_endpoint_unexposed_only_mae']:.6f} | "
            f"{item['mean_endpoint_delta_unexposed_minus_all']:+.6f} | "
            f"{item['endpoints_with_any_exposed_test_row']}/23 | [{ci[0]:+.6f}, {ci[1]:+.6f}] |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                "| Exclusion | Unexposed ChemRasayan - ECFP4-LightGBM MAE | 95% bootstrap CI | ChemRasayan wins |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, key_type in display_keys:
            item = comparisons[key_type]
            ci = item["bootstrap_95pct_ci_for_unexposed_difference"]
            lines.append(
                f"| {label} | {item['mean_endpoint_unexposed_chemrasayan_minus_baseline']:+.6f} | "
                f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {item['chemrasayan_unexposed_wins']}/23 |"
            )
    lines.extend(
        [
            "",
            "Boundary: this removes test rows by the stated Step-2 structure or ECFP4-neighbour rule. "
            "It does not address scaffold exposure, endpoint-assay semantic equivalence, or Step-1 ZINC exposure.",
            "",
        ]
    )
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate["full_mix"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
