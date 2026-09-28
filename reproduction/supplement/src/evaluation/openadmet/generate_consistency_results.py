#!/usr/bin/env python3
"""Regenerate numerical claims and table rows used by the manuscript.

Experiment artifacts are read-only. The script validates checkpoint
provenance, derives the validation-selected OpenADMET summary, builds the
matched RTD-15%/MLM-15% TDC comparison, computes the post hoc endpoint-level
tests reported in the paper, and writes LaTeX fragments plus a machine-readable
audit report beside main.tex.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import fmean, median


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

OPENADMET = [
    ("ExpansionRx", "Caco-2 permeability", "expansion_caco2_pappa", 0.340, 0.390),
    ("ExpansionRx", "Caco-2 efflux", "expansion_caco2_efflux", 0.250, 0.240),
    ("ExpansionRx", "LogD", "expansion_logd", 0.330, 0.740),
    ("ExpansionRx", "KSOL", "expansion_ksol", 0.450, 0.600),
    ("ExpansionRx", "HLM CLint", "expansion_hlm", 0.390, 0.460),
    ("ExpansionRx", "MLM CLint", "expansion_mlm", 0.400, 0.420),
    ("ExpansionRx", "MBPB", "expansion_mbpb", 0.300, 0.450),
    ("ExpansionRx", "MGMB", "expansion_mgmb", 0.300, 0.410),
    ("ExpansionRx", "MPPB", "expansion_mppb", 0.260, 0.410),
    ("ASAP", "MERS-CoV potency", "asap_mers", 0.590, 0.520),
    ("ASAP", "SARS-CoV-2 potency", "asap_sars", 0.620, 0.730),
    ("ASAP", "LogD", "asap_logd", 0.680, 0.640),
    ("ASAP", "KSOL", "asap_ksol", 0.420, 0.670),
    ("ASAP", "HLM", "asap_hlm", 0.390, 0.360),
    ("ASAP", "MLM", "asap_mlm", 0.530, 0.640),
    ("ASAP", "MDR1 efflux", "asap_mdr1", 0.420, 0.450),
    ("PXR", "PXR activity", "pxr", 0.550, 0.790),
    ("Biogen", "Solubility", "biogen_solubility", 0.300, 0.420),
    ("Biogen", "HLM CLint", "biogen_hlm", 0.330, 0.510),
    ("Biogen", "RLM CLint", "biogen_rlm", 0.370, 0.570),
    ("Biogen", "HPPB", "biogen_hppb", 0.330, 0.560),
    ("Biogen", "RPPB", "biogen_rppb", 0.430, 0.570),
    ("Biogen", "MDR1 efflux", "biogen_mdr1", 0.270, 0.390),
]

TDC_NAMES = {
    "ames": "Ames",
    "bbb_martins": "BBB Martins",
    "bioavailability_ma": "Bioavailability",
    "caco2_wang": "Caco-2",
    "clearance_hepatocyte_az": "Hepatocyte clearance",
    "clearance_microsome_az": "Microsome clearance",
    "clintox": "ClinTox",
    "cyp2c9_substrate_carbonmangels": "CYP2C9 substrate",
    "cyp2c9_veith": "CYP2C9 inhibition",
    "cyp2d6_substrate_carbonmangels": "CYP2D6 substrate",
    "cyp2d6_veith": "CYP2D6 inhibition",
    "cyp3a4_substrate_carbonmangels": "CYP3A4 substrate",
    "cyp3a4_veith": "CYP3A4 inhibition",
    "dili": "DILI",
    "half_life_obach": "Half-life",
    "herg": "hERG",
    "hia_hou": "HIA",
    "ld50_zhu": "LD50",
    "lipophilicity_astrazeneca": "Lipophilicity",
    "pgp_broccatelli": "P-glycoprotein",
    "ppbr_az": "PPBR",
    "solubility_aqsoldb": "Solubility",
    "vdss_lombardo": "VDss",
}

METRIC_FIELDS = {
    "auroc": ("test_auroc_mean", "test_auroc_std", False, r"AUROC $\uparrow$"),
    "auprc": ("test_auprc_mean", "test_auprc_std", False, r"AUPRC $\uparrow$"),
    "mae": ("test_mae_mean", "test_mae_std", True, r"MAE $\downarrow$"),
    "spearman": (
        "test_spearman_mean",
        "test_spearman_std",
        False,
        r"Spearman $\uparrow$",
    ),
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def tex_value(mean: float, sd: float, bold: bool) -> str:
    value = "$" + rf"{mean:.3f}\pm{sd:.3f}$"
    return rf"\best{{{value}}}" if bold else value


def tex_scalar(value: float, bold: bool) -> str:
    rendered = f"{value:.3f}"
    return rf"\best{{{rendered}}}" if bold else rendered


def average_ranks(values: list[float]) -> list[float]:
    """Return one-based average ranks, with smaller values ranked first."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def exact_sign_test(wins: int, total: int) -> float:
    """Two-sided exact binomial sign test under P(win)=0.5."""
    smaller_tail = min(wins, total - wins)
    tail = sum(math.comb(total, k) for k in range(smaller_tail + 1)) / (2**total)
    return min(1.0, 2 * tail)


def exact_wilcoxon(x: list[float], y: list[float]) -> dict:
    """Two-sided paired Wilcoxon signed-rank test by exact sign enumeration."""
    differences = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-15]
    ranks = average_ranks([abs(value) for value in differences])
    # Average ranks are integers or half-integers. Doubling makes the dynamic
    # program exact without floating-point state keys.
    rank_units = [round(2 * rank) for rank in ranks]
    observed_positive = sum(
        units for units, difference in zip(rank_units, differences) if difference > 0
    )
    total_units = sum(rank_units)
    counts = [0] * (total_units + 1)
    counts[0] = 1
    for units in rank_units:
        for score in range(total_units, units - 1, -1):
            counts[score] += counts[score - units]
    lower_score = min(observed_positive, total_units - observed_positive)
    tail_count = sum(counts[: lower_score + 1])
    p_value = min(1.0, 2 * tail_count / (2 ** len(differences)))
    w_plus = observed_positive / 2
    w_minus = (total_units - observed_positive) / 2
    return {
        "n": len(differences),
        "w": min(w_plus, w_minus),
        "p_two_sided_exact": p_value,
        "median_paired_difference": median(differences),
        "rank_biserial": (w_plus - w_minus) / (w_plus + w_minus),
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    """Linearly interpolated percentile of an already sorted sample."""
    if not 0 <= probability <= 1 or not sorted_values:
        raise ValueError("invalid percentile request")
    location = probability * (len(sorted_values) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return sorted_values[lower]
    weight = location - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def endpoint_bootstrap_difference(
    left: list[float],
    right: list[float],
    *,
    seed: int,
    replicates: int = 100_000,
) -> dict:
    """Bootstrap the mean paired difference with the endpoint as the unit."""
    if len(left) != len(right) or not left:
        raise ValueError("paired endpoint bootstrap requires equal nonempty columns")
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    samples = []
    n = len(differences)
    for _ in range(replicates):
        samples.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return {
        "analysis_unit": "endpoint mean across three reconstructed splits",
        "n_endpoints": n,
        "replicates": replicates,
        "seed": seed,
        "mean_paired_difference": fmean(differences),
        "percentile_95_ci": [
            percentile(samples, 0.025),
            percentile(samples, 0.975),
        ],
        "direction": "negative favors the left-named method (lower MAE)",
    }


def holm_adjust(named_p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(named_p_values, key=named_p_values.get)
    adjusted = {}
    running_max = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        candidate = min(1.0, (total - index) * named_p_values[name])
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def friedman_three_way(columns: list[list[float]]) -> dict:
    """Friedman rank test for three repeated methods; chi-square df=2 tail."""
    if len(columns) != 3 or len({len(column) for column in columns}) != 1:
        raise ValueError("Friedman analysis expects three equal-length columns")
    n_blocks = len(columns[0])
    rank_sums = [0.0, 0.0, 0.0]
    tie_term = 0
    for values in zip(*columns):
        ranks = average_ranks(list(values))
        rank_sums = [total + rank for total, rank in zip(rank_sums, ranks)]
        counts = {value: values.count(value) for value in set(values)}
        tie_term += sum(count**3 - count for count in counts.values() if count > 1)
    statistic = 12 * sum(value**2 for value in rank_sums) / (n_blocks * 3 * 4)
    statistic -= 3 * n_blocks * 4
    correction = 1 - tie_term / (n_blocks * (3**3 - 3))
    statistic /= correction
    return {
        "n_endpoints": n_blocks,
        "chi_square": statistic,
        "df": 2,
        # For two degrees of freedom, the chi-square survival function is exp(-x/2).
        "p_asymptotic": math.exp(-statistic / 2),
        "mean_ranks": [value / n_blocks for value in rank_sums],
    }


def build_adaptation_statistics(rows: list[dict]) -> dict:
    methods = ["full_mean", "lora_mean", "frozen_mean"]
    labels = ["full", "lora", "frozen"]
    columns = [[row[method] for row in rows] for method in methods]
    pairwise = {}
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        name = f"{labels[left_index]}_vs_{labels[right_index]}"
        pairwise[name] = exact_wilcoxon(columns[left_index], columns[right_index])
    adjusted = holm_adjust(
        {name: result["p_two_sided_exact"] for name, result in pairwise.items()}
    )
    for name, value in adjusted.items():
        pairwise[name]["p_holm"] = value
    bootstrap = {}
    for comparison_index, (left_index, right_index) in enumerate(
        ((0, 1), (0, 2), (1, 2))
    ):
        name = f"{labels[left_index]}_vs_{labels[right_index]}"
        bootstrap[name] = endpoint_bootstrap_difference(
            columns[left_index],
            columns[right_index],
            seed=20260917 + comparison_index,
        )
    return {
        "analysis_unit": "endpoint mean across three reconstructed splits",
        "method_order": labels,
        "method_mean_mae": [fmean(column) for column in columns],
        "friedman": friedman_three_way(columns),
        "pairwise_wilcoxon": pairwise,
        "paired_endpoint_bootstrap": bootstrap,
    }


def summarize_adaptation_selection(records: list[dict]) -> dict:
    """Describe validation-selected adaptation choices without treating splits as independent."""
    labels = {
        "full_mix": "full",
        "lora_mix": "lora",
        "frozen_mix": "frozen",
    }
    counts = Counter(labels[row["selected_method"]] for row in records)
    medians = {
        label: median(
            row["train_rows"]
            for row in records
            if labels[row["selected_method"]] == label
        )
        for label in ("full", "lora", "frozen")
    }
    ordered = sorted(records, key=lambda row: row["train_rows"])
    smallest_quartile = ordered[: len(ordered) // 4]
    remainder = ordered[len(ordered) // 4 :]
    smallest_counts = Counter(
        labels[row["selected_method"]] for row in smallest_quartile
    )
    remainder_counts = Counter(labels[row["selected_method"]] for row in remainder)

    expected = {"full": 39, "lora": 18, "frozen": 12}
    if dict(counts) != expected:
        raise ValueError(f"unexpected adaptation-selection counts: {dict(counts)}")
    observed_smallest = {label: smallest_counts[label] for label in expected}
    if observed_smallest != {"full": 8, "lora": 1, "frozen": 8}:
        raise ValueError(
            "unexpected adaptation selections in the smallest training-size quartile"
        )

    return {
        "analysis_unit": "split-level validation selection (descriptive; not independent)",
        "total_split_decisions": len(records),
        "selection_counts": {label: counts[label] for label in expected},
        "median_training_rows_when_selected": medians,
        "smallest_training_size_quartile": {
            "n": len(smallest_quartile),
            "training_rows_min": smallest_quartile[0]["train_rows"],
            "training_rows_max": smallest_quartile[-1]["train_rows"],
            "selection_counts": observed_smallest,
        },
        "remaining_split_decisions": {
            "n": len(remainder),
            "selection_counts": {
                label: remainder_counts[label] for label in expected
            },
        },
    }


def build_openadmet() -> tuple[list[dict], dict]:
    rows = []
    selection_records = []
    trainable_parameters = {
        "full_mix": set(),
        "lora_mix": set(),
        "frozen_mix": set(),
    }
    total_parameters = {method: set() for method in trainable_parameters}
    for source, endpoint, task, moljepa, chemprop in OPENADMET:
        data = load_json(ROOT / "results_rtd_phase4_95m_moljepa" / f"results_{task}.json")
        if data["task"] != task:
            raise ValueError(f"{task}: task identifier mismatch")
        expected_rule = "method selected separately per split using validation MAE only"
        if data["selection_rule"] != expected_rule:
            raise ValueError(f"{task}: unexpected selection rule")
        if len(data["selected_split_test_mae"]) != 3:
            raise ValueError(f"{task}: expected three selected test splits")
        for split_name, split in sorted(data["splits"].items()):
            selected_method = split["selected_method"]
            selected_record = split["methods"][selected_method]
            selection_records.append(
                {
                    "task": task,
                    "split": split_name,
                    "selected_method": selected_method,
                    "train_rows": int(selected_record["train_rows"]),
                }
            )
            for method_name, method_record in split["methods"].items():
                trainable_parameters[method_name].add(
                    int(method_record["trainable_parameters"])
                )
                total_parameters[method_name].add(int(method_record["total_parameters"]))

        selected = float(data["selected_mean_test_mae"])
        selected_sd = float(data["selected_std_test_mae"])
        full = float(data["method_summary"]["full_mix"]["mean_test_mae"])
        full_sd = float(data["method_summary"]["full_mix"]["std_test_mae"])
        lora = float(data["method_summary"]["lora_mix"]["mean_test_mae"])
        frozen = float(data["method_summary"]["frozen_mix"]["mean_test_mae"])
        best_value = min(selected, full, moljepa, chemprop)
        rows.append(
            {
                "source": source,
                "endpoint": endpoint,
                "task": task,
                "selected_mean": selected,
                "selected_sd": selected_sd,
                "full_mean": full,
                "full_sd": full_sd,
                "lora_mean": lora,
                "frozen_mean": frozen,
                "moljepa": moljepa,
                "chemprop": chemprop,
                "selected_lower_than_moljepa": selected < moljepa,
                "latex": (
                    f"{source} & {endpoint} & "
                    f"{tex_value(selected, selected_sd, selected == best_value)} & "
                    f"{tex_value(full, full_sd, full == best_value)} & "
                    f"{tex_scalar(moljepa, moljepa == best_value)} & "
                    f"{tex_scalar(chemprop, chemprop == best_value)} \\\\"
                ),
            }
        )

    selected_mean = fmean(row["selected_mean"] for row in rows)
    wins = sum(row["selected_lower_than_moljepa"] for row in rows)
    expansion_wins = sum(
        row["selected_lower_than_moljepa"] for row in rows if row["source"] == "ExpansionRx"
    )
    if len(rows) != 23 or wins != 18 or expansion_wins != 9:
        raise ValueError(
            f"OpenADMET consistency failure: rows={len(rows)}, wins={wins}, "
            f"ExpansionRx wins={expansion_wins}"
        )
    if abs(selected_mean - 0.375834) > 5e-7:
        raise ValueError(f"unexpected validation-selected aggregate: {selected_mean}")
    if any(len(values) != 1 for values in trainable_parameters.values()):
        raise ValueError(f"inconsistent trainable parameter counts: {trainable_parameters}")
    if any(len(values) != 1 for values in total_parameters.values()):
        raise ValueError(f"inconsistent total parameter counts: {total_parameters}")
    parameter_summary = {}
    for method in trainable_parameters:
        trainable = next(iter(trainable_parameters[method]))
        total = next(iter(total_parameters[method]))
        parameter_summary[method] = {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        }
    return rows, {
        "endpoints": len(rows),
        "selected_mean_mae": selected_mean,
        "lower_point_estimates_than_moljepa": wins,
        "expansionrx_lower_point_estimates": expansion_wins,
        "fixed_adaptation_statistics": build_adaptation_statistics(rows),
        "adaptation_selection_by_training_size": summarize_adaptation_selection(
            selection_records
        ),
        "adaptation_parameter_counts": parameter_summary,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_openadmet_release_manifest() -> dict:
    """Inventory the result JSONs that carry held-out targets and predictions."""
    result_dir = ROOT / "results_rtd_phase4_95m_moljepa"
    entries = []
    for _, _, task, _, _ in OPENADMET:
        path = result_dir / f"results_{task}.json"
        data = load_json(path)
        split_entries = []
        for split_name, split in sorted(data["splits"].items()):
            methods = split["methods"]
            reference_targets = None
            for method_name, method in sorted(methods.items()):
                targets = method["y_true"]
                predictions = method["y_pred"]
                if len(targets) != len(predictions) or len(targets) != method["test_rows"]:
                    raise ValueError(
                        f"{task}/{split_name}/{method_name}: prediction length mismatch"
                    )
                if reference_targets is None:
                    reference_targets = targets
                elif targets != reference_targets:
                    raise ValueError(
                        f"{task}/{split_name}: held-out targets differ across methods"
                    )
            selected_method = split["selected_method"]
            selected = methods[selected_method]
            split_entries.append(
                {
                    "split": split_name,
                    "selected_method": selected_method,
                    "train_rows": selected["train_rows"],
                    "validation_rows": selected["validation_rows"],
                    "test_rows": selected["test_rows"],
                    "train_smiles_sha256": selected["train_smiles_sha256"],
                    "validation_smiles_sha256": selected["validation_smiles_sha256"],
                    "test_smiles_sha256": selected["test_smiles_sha256"],
                    "methods_with_y_true_y_pred": sorted(methods),
                }
            )
        entries.append(
            {
                "task": task,
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_file_sha256": file_sha256(path),
                "checkpoint_sha256": data["checkpoint_sha256"],
                "data_sha256": next(iter(data["splits"].values()))["methods"][
                    "full_mix"
                ]["data_sha256"],
                "splits": split_entries,
            }
        )
    return {
        "schema_version": 1,
        "description": (
            "Inventory of the 23 OpenADMET result JSONs. Each source JSON contains "
            "held-out y_true/y_pred arrays for full fine-tuning, LoRA, and frozen "
            "layer mixing on three reconstructed splits."
        ),
        "endpoint_files": len(entries),
        "entries": entries,
    }


def build_tdc() -> tuple[list[dict], dict]:
    metric_rows = {}
    with (ROOT / "benchmark_results" / "ablation_rtd25.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            metric_rows[row["task"]] = row

    rows = []
    excluded = []
    for task, display in TDC_NAMES.items():
        rtd = load_json(
            ROOT / "paperlike_results" / f"results_paperlike_{task}.json"
        )
        mlm_path = ROOT / "results_small" / "mnt" / f"results_mlm_s2_{task}.json"
        try:
            mlm = load_json(mlm_path)
        except (json.JSONDecodeError, FileNotFoundError):
            excluded.append(
                {
                    "task": task,
                    "reason": "MLM artifact is missing or not valid JSON",
                }
            )
            continue
        if "sup_pretrain_paperlike" not in rtd["ckpt"]:
            raise ValueError(f"{task}: RTD-15 checkpoint provenance failed")
        if "sup_pretrain_mlm" not in mlm["ckpt"]:
            excluded.append(
                {
                    "task": task,
                    "reason": f"MLM-labelled artifact points to {mlm['ckpt']}",
                }
            )
            continue

        metric = metric_rows[task]["metric"].lower()
        mean_field, sd_field, lower, metric_tex = METRIC_FIELDS[metric]
        rtd_mean = float(rtd["pretrained"][mean_field])
        rtd_sd = float(rtd["pretrained"][sd_field])
        mlm_mean = float(mlm["pretrained"][mean_field])
        mlm_sd = float(mlm["pretrained"][sd_field])
        rtd_seeds = [item["seed"] for item in rtd["pretrained"]["per_seed"]]
        mlm_seeds = [item["seed"] for item in mlm["pretrained"]["per_seed"]]
        if rtd_seeds != [0, 1, 2] or mlm_seeds != [0, 1, 2]:
            raise ValueError(f"{task}: expected matched seeds 0, 1, 2")
        for key in ("epochs", "batch_size", "lr"):
            if rtd[key] != mlm[key]:
                raise ValueError(f"{task}: downstream setting {key} does not match")

        rtd_wins = rtd_mean < mlm_mean if lower else rtd_mean > mlm_mean
        rows.append(
            {
                "task": task,
                "display": display,
                "metric": metric,
                "rtd15_mean": rtd_mean,
                "rtd15_sd": rtd_sd,
                "mlm15_mean": mlm_mean,
                "mlm15_sd": mlm_sd,
                "winner": "RTD-15%-S2" if rtd_wins else "MLM-15%-S2",
                "latex": (
                    f"{display} & {metric_tex} & "
                    f"{tex_value(rtd_mean, rtd_sd, rtd_wins)} & "
                    f"{tex_value(mlm_mean, mlm_sd, not rtd_wins)} \\\\"
                ),
            }
        )

    rtd_wins = sum(row["winner"] == "RTD-15%-S2" for row in rows)
    mlm_wins = len(rows) - rtd_wins
    if len(rows) != 22 or rtd_wins != 17 or mlm_wins != 5:
        raise ValueError(
            f"TDC consistency failure: rows={len(rows)}, RTD={rtd_wins}, MLM={mlm_wins}"
        )
    if len(excluded) != 1 or excluded[0]["task"] != "bbb_martins":
        raise ValueError(f"unexpected excluded TDC artifacts: {excluded}")
    return rows, {
        "valid_endpoints": len(rows),
        "rtd15_wins": rtd_wins,
        "mlm15_wins": mlm_wins,
        "exact_sign_test_p_two_sided": exact_sign_test(rtd_wins, len(rows)),
        "excluded": excluded,
    }


def build_mask_rate_sensitivity() -> dict:
    rtd15_wins = 0
    rtd25_wins = 0
    with (ROOT / "benchmark_results" / "ablation_rtd25.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        lower = row["metric"].lower() == "mae"
        rtd15 = float(row["s2_15pct"])
        rtd25 = float(row["s2_25pct"])
        rtd15_won = rtd15 < rtd25 if lower else rtd15 > rtd25
        if rtd15_won:
            rtd15_wins += 1
        else:
            rtd25_wins += 1
    if len(rows) != 23 or (rtd15_wins, rtd25_wins) != (9, 14):
        raise ValueError(
            f"mask-rate consistency failure: rows={len(rows)}, "
            f"RTD15={rtd15_wins}, RTD25={rtd25_wins}"
        )
    return {
        "endpoints": len(rows),
        "rtd15_wins": rtd15_wins,
        "rtd25_wins": rtd25_wins,
        "exact_sign_test_p_two_sided": exact_sign_test(rtd25_wins, len(rows)),
    }


def build_step2_sensitivity() -> dict:
    wins = 0
    total = 0
    with (ROOT / "benchmark_results" / "ablation_rtd25.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        lower = row["metric"].lower() == "mae"
        step1 = float(row["s1_25pct"])
        step2 = float(row["s2_25pct"])
        wins += step2 < step1 if lower else step2 > step1
        total += 1
    if (wins, total) != (18, 23):
        raise ValueError(f"Step-2 consistency failure: wins={wins}, total={total}")
    return {
        "endpoints": total,
        "step2_wins": wins,
        "step1_wins": total - wins,
        "exact_sign_test_p_two_sided": exact_sign_test(wins, total),
    }


def audit_openadmet_objective_package() -> dict:
    package = ROOT / "results_package" / "mlm_rtd_moljepa23_retrieved_20260903"
    provenance = {}
    for label in ("mlm_s2", "rtd25_s2"):
        files = sorted((package / label).glob("results_*.json"))
        if len(files) != 23:
            raise ValueError(f"{label}: expected 23 files, found {len(files)}")
        records = [load_json(path) for path in files]
        provenance[label] = {
            "checkpoint_paths": sorted({row["checkpoint"] for row in records}),
            "checkpoint_hashes": sorted({row["checkpoint_sha256"] for row in records}),
        }
    distinct = (
        set(provenance["mlm_s2"]["checkpoint_paths"]).isdisjoint(
            provenance["rtd25_s2"]["checkpoint_paths"]
        )
        and set(provenance["mlm_s2"]["checkpoint_hashes"]).isdisjoint(
            provenance["rtd25_s2"]["checkpoint_hashes"]
        )
    )
    return {
        "publishable_as_rtd_vs_mlm": distinct,
        "provenance": provenance,
        "decision": (
            "retain paired objective comparison"
            if distinct
            else "exclude: both result directories contain the same MLM checkpoint provenance"
        ),
    }


def challenge_summary() -> dict:
    data = load_json(
        ROOT / "paper_rtd" / "evidence" / "asap_polaris_potency_rtd25_s2.json"
    )
    wanted = {}
    for row in data["test_metrics"]:
        if row["Metric"] == "mean_absolute_error":
            wanted[row["Target Label"]] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
    expected = {
        "aggregated",
        "pIC50 (MERS-CoV Mpro)",
        "pIC50 (SARS-CoV-2 Mpro)",
    }
    if set(wanted) != expected:
        raise ValueError("ASAP challenge MAE rows are incomplete")
    return {
        "n_bootstrap": data["n_bootstrap"],
        "seed": data["seed"],
        "validation_fraction": data["val_frac"],
        "mae": wanted,
    }


def write_outputs() -> None:
    openadmet_rows, openadmet_summary = build_openadmet()
    release_manifest = build_openadmet_release_manifest()
    tdc_rows, tdc_summary = build_tdc()
    mask_rate = build_mask_rate_sensitivity()
    step2 = build_step2_sensitivity()
    tdc_direction_adjusted = holm_adjust(
        {
            "rtd15_vs_mlm15": tdc_summary["exact_sign_test_p_two_sided"],
            "rtd25_vs_rtd15": mask_rate["exact_sign_test_p_two_sided"],
            "step2_vs_step1": step2["exact_sign_test_p_two_sided"],
        }
    )
    tdc_summary["exact_sign_test_p_holm"] = tdc_direction_adjusted["rtd15_vs_mlm15"]
    mask_rate["exact_sign_test_p_holm"] = tdc_direction_adjusted["rtd25_vs_rtd15"]
    step2["exact_sign_test_p_holm"] = tdc_direction_adjusted["step2_vs_step1"]
    objective_audit = audit_openadmet_objective_package()
    challenge = challenge_summary()

    if objective_audit["publishable_as_rtd_vs_mlm"]:
        raise ValueError(
            "The OpenADMET objective package is now provenance-distinct; "
            "review the manuscript exclusion before regenerating."
        )

    report = {
        "openadmet_validation_selected": openadmet_summary,
        "asap_challenge_test": challenge,
        "tdc_rtd15_vs_mlm15": tdc_summary,
        "rtd15_vs_rtd25_mask_rate": mask_rate,
        "rtd25_step1_vs_step2": step2,
        "openadmet_objective_package_audit": objective_audit,
        "openadmet_release_bundle": {
            "manifest": "generated_openadmet_release_manifest.json",
            "endpoint_files": release_manifest["endpoint_files"],
            "contains_held_out_targets_and_predictions": True,
        },
        "openadmet_rows": [
            {key: value for key, value in row.items() if key != "latex"}
            for row in openadmet_rows
        ],
        "tdc_rows": [
            {key: value for key, value in row.items() if key != "latex"}
            for row in tdc_rows
        ],
    }
    (OUT / "generated_consistency_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "generated_openadmet_rows.tex").write_text(
        "\n".join(row["latex"] for row in openadmet_rows) + "\n\\bottomrule\n",
        encoding="utf-8",
    )
    (OUT / "generated_rtd_mlm_rows.tex").write_text(
        "\n".join(row["latex"] for row in tdc_rows) + "\n\\bottomrule\n",
        encoding="utf-8",
    )
    (OUT / "generated_openadmet_release_manifest.json").write_text(
        json.dumps(release_manifest, indent=2) + "\n", encoding="utf-8"
    )

    concise = {key: value for key, value in report.items() if not key.endswith("_rows")}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    write_outputs()
