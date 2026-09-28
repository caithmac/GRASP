#!/usr/bin/env python
"""Audit RTD calibration and frozen-checkpoint corruption-mode sensitivity.

This evaluation-only diagnostic reuses the exact endpoint-balanced sample,
masking contract, vocabulary, model configuration, and checkpoint loader from
``audit_rtd_objective.py``.  It compares deterministic argmax corruption with
fixed-seed categorical sampling (temperature 1) and top-k sampling (k=5), then
exports token-level discriminator probabilities and labels, equal-width
calibration bins, and a descriptive threshold sweep.

Threshold selection in this script is deliberately descriptive: thresholds
are evaluated and summarized on the same diagnostic sample.  They must not be
reported as independently validated operating points.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_geometric
from rdkit import Chem, RDLogger, __version__ as rdkit_version
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch

import audit_rtd_objective as base
from mole.training.data.pretrain_datasets import MolPreTrainDataset


REPO_ROOT = base.REPO_ROOT
DEFAULT_STEM = REPO_ROOT / "results_package" / "rtd_calibration_sampling_phase4_tdc"
DEFAULT_JSON = DEFAULT_STEM.with_suffix(".json")
DEFAULT_SUMMARY_CSV = DEFAULT_STEM.with_name(DEFAULT_STEM.name + "_summary.csv")
DEFAULT_TOKENS_CSV = DEFAULT_STEM.with_name(DEFAULT_STEM.name + "_tokens.csv")
DEFAULT_BINS_CSV = DEFAULT_STEM.with_name(DEFAULT_STEM.name + "_calibration_bins.csv")
DEFAULT_THRESHOLDS_CSV = DEFAULT_STEM.with_name(
    DEFAULT_STEM.name + "_threshold_sweep.csv"
)
SUPPORTED_MODES = ("argmax", "categorical", "topk5")
MISMATCH_FIELDS = (
    "atomic_number",
    "degree",
    "formal_charge",
    "aromaticity",
    "total_valence",
    "total_hydrogens",
    "ring_membership",
    "hybridization",
)


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def stable_mode_seed(mode: str, sampling_seed: int) -> int | None:
    if mode == "argmax":
        return None
    if mode == "categorical":
        return sampling_seed
    if mode == "topk5":
        return sampling_seed + 1
    raise ValueError(f"Unsupported corruption mode: {mode}")


def parse_modes(text: str) -> list[str]:
    modes = list(dict.fromkeys(part.strip().lower() for part in text.split(",")))
    invalid = [mode for mode in modes if mode not in SUPPORTED_MODES]
    if not modes or invalid:
        raise argparse.ArgumentTypeError(
            f"Modes must be a comma-separated subset of {SUPPORTED_MODES}; "
            f"received {text!r}"
        )
    return modes


def sample_predictions(
    generator_logits: torch.Tensor,
    selected: torch.Tensor,
    mode: str,
    rng: torch.Generator | None,
) -> torch.Tensor:
    """Return one token prediction per position without changing mask RNG state."""
    predictions = generator_logits.argmax(dim=-1)
    if mode == "argmax" or not bool(selected.any()):
        return predictions

    selected_logits = generator_logits[selected]
    if mode == "categorical":
        probabilities = torch.softmax(selected_logits, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=rng).squeeze(1)
    elif mode == "topk5":
        top_values, top_indices = selected_logits.topk(k=5, dim=-1)
        top_probabilities = torch.softmax(top_values, dim=-1)
        sampled_offsets = torch.multinomial(
            top_probabilities, 1, generator=rng
        ).squeeze(1)
        sampled = top_indices.gather(1, sampled_offsets[:, None]).squeeze(1)
    else:  # guarded by argument validation
        raise ValueError(f"Unsupported corruption mode: {mode}")

    predictions = predictions.clone()
    predictions[selected] = sampled
    return predictions


def counter_diversity(
    counter: collections.Counter[int],
) -> dict[str, float | int | None]:
    total = sum(counter.values())
    if not total:
        return {
            "observations": 0,
            "unique_tokens": 0,
            "shannon_entropy_nats": None,
            "normalized_shannon_entropy": None,
            "effective_number_of_tokens": None,
            "top1_token_share": None,
            "top5_token_share": None,
        }
    probabilities = np.asarray(list(counter.values()), dtype=np.float64) / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    unique = len(counter)
    normalized = entropy / math.log(unique) if unique > 1 else 0.0
    return {
        "observations": total,
        "unique_tokens": unique,
        "shannon_entropy_nats": entropy,
        "normalized_shannon_entropy": normalized,
        "effective_number_of_tokens": math.exp(entropy),
        "top1_token_share": base.safe_divide(counter.most_common(1)[0][1], total),
        "top5_token_share": base.safe_divide(
            sum(value for _, value in counter.most_common(5)), total
        ),
    }


def confusion_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    positive = labels.astype(bool)
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    precision = base.safe_divide(tp, tp + fp)
    recall = base.safe_divide(tp, tp + fn)
    specificity = base.safe_divide(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": base.safe_divide(2 * tp, 2 * tp + fp + fn),
        "accuracy": base.safe_divide(tp + tn, len(labels)),
    }


def calibration_bins(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[list[dict[str, float | int | None]], float]:
    bin_ids = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    rows: list[dict[str, float | int | None]] = []
    ece = 0.0
    for bin_index in range(n_bins):
        member = bin_ids == bin_index
        count = int(np.count_nonzero(member))
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if count:
            mean_probability = float(probabilities[member].mean())
            positive_fraction = float(labels[member].mean())
            absolute_gap = abs(mean_probability - positive_fraction)
            contribution = count / len(labels) * absolute_gap
            ece += contribution
        else:
            mean_probability = None
            positive_fraction = None
            absolute_gap = None
            contribution = 0.0
        rows.append(
            {
                "bin_index": bin_index,
                "lower_bound_inclusive": lower,
                "upper_bound_inclusive_only_for_last_bin": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "empirical_positive_fraction": positive_fraction,
                "absolute_calibration_gap": absolute_gap,
                "ece_contribution": contribution,
            }
        )
    return rows, ece


def best_descriptive_row(
    rows: list[dict[str, float | int]], metric: str
) -> dict[str, float | int]:
    finite_rows = [row for row in rows if math.isfinite(float(row[metric]))]
    if not finite_rows:
        return {"threshold": float("nan"), metric: float("nan")}
    # Prefer the lowest threshold when values tie; this rule is recorded solely
    # to make the descriptive summary deterministic.
    return max(finite_rows, key=lambda row: (float(row[metric]), -float(row["threshold"])))


def run_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.mask_seed)
    np.random.seed(args.mask_seed)
    random.seed(args.mask_seed)
    torch.set_num_threads(args.num_threads)

    data_paths = [Path(path).resolve() for path in sorted(base.glob.glob(args.data_glob))]
    if not data_paths:
        raise FileNotFoundError(f"No data files matched {args.data_glob!r}")

    import pickle

    with args.vocab.open("rb") as handle:
        vocabulary = pickle.load(handle)  # trusted repository artifact
    vocabulary.update({"PAD": 0, "MASK": 208, "UNK": 209, "CLS": 210})

    smiles_sample, sample_provenance = base.build_evaluation_sample(
        data_paths, args.per_task, args.sample_seed
    )
    signature_map, signature_provenance = base.build_signature_map(
        data_paths, vocabulary, args.signature_per_task
    )
    model, checkpoint_metadata = base.load_model(args.checkpoint, args.config)

    mode_rngs: dict[str, torch.Generator | None] = {}
    for mode in args.modes:
        seed = stable_mode_seed(mode, args.sampling_seed)
        if seed is None:
            mode_rngs[mode] = None
        else:
            mode_rngs[mode] = torch.Generator(device="cpu").manual_seed(seed)

    mode_counts = {mode: collections.Counter() for mode in args.modes}
    sampled_tokens = {mode: collections.Counter() for mode in args.modes}
    replacement_tokens = {mode: collections.Counter() for mode in args.modes}
    mismatch_counts = {mode: collections.Counter() for mode in args.modes}
    probabilities_by_mode: dict[str, list[torch.Tensor]] = {
        mode: [] for mode in args.modes
    }
    labels_by_mode: dict[str, list[torch.Tensor]] = {mode: [] for mode in args.modes}
    token_rows: list[dict[str, Any]] = []

    common_counts: collections.Counter = collections.Counter()
    generator_nll_sum = 0.0
    started_at = time.time()
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    for start in range(0, len(smiles_sample), args.batch_size):
        batch_smiles = smiles_sample[start : start + args.batch_size]
        dataset = MolPreTrainDataset(
            pd.Series(batch_smiles),
            vocabulary,
            mask_prob=args.mask_probability,
            radius_inp=0,
            useFeatures_inp=False,
        )
        items = [dataset[index] for index in range(len(dataset))]
        batch = Batch.from_data_list(items)
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        original_ids, _ = to_dense_batch(batch.original_ids, batch.batch, fill_value=0)
        mlm_labels, _ = to_dense_batch(batch.mlm_labels, batch.batch, fill_value=-100)
        relative_positions = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length).expand(batch_size, sequence_length)
        selected = mlm_labels.ne(-100)
        valid = input_mask.bool()

        with torch.inference_mode():
            generator_hidden = model.generator(
                input_ids,
                input_mask,
                attention_mask=input_mask,
                relative_pos=relative_positions,
            )["hidden_states"][-1]
            generator_logits = model.gen_head(generator_hidden, position_ids=position_ids)

        common_counts["valid_tokens"] += int(valid.sum())
        common_counts["atom_tokens"] += int(valid.sum()) - batch_size
        common_counts["selected_tokens"] += int(selected.sum())
        generator_nll_sum += float(
            torch.nn.functional.cross_entropy(
                generator_logits[selected], original_ids[selected], reduction="sum"
            )
        )

        for mode in args.modes:
            predictions = sample_predictions(
                generator_logits, selected, mode, mode_rngs[mode]
            )
            discriminator_ids = original_ids.clone()
            discriminator_ids[selected] = predictions[selected]
            discriminator_labels = (discriminator_ids != original_ids) & valid

            with torch.inference_mode():
                discriminator_hidden = model.discriminator(
                    discriminator_ids,
                    input_mask,
                    attention_mask=input_mask,
                    relative_pos=relative_positions,
                )["hidden_states"][-1]
                discriminator_logits = model.disc_head(
                    discriminator_hidden, position_ids=position_ids
                )
                discriminator_probabilities = torch.sigmoid(discriminator_logits)

            probabilities_by_mode[mode].append(
                discriminator_probabilities[valid].detach().cpu()
            )
            labels_by_mode[mode].append(discriminator_labels[valid].detach().cpu())
            mode_counts[mode]["replaced_tokens"] += int(discriminator_labels.sum())
            mode_counts[mode]["sampled_original_tokens"] += int(
                (predictions[selected] == original_ids[selected]).sum()
            )
            mode_counts[mode]["discriminator_bce_sum"] += float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    discriminator_logits[valid],
                    discriminator_labels[valid].float(),
                    reduction="sum",
                )
            )
            sampled_tokens[mode].update(int(token) for token in predictions[selected].tolist())
            replacement_mask = selected & (predictions != original_ids)
            replacement_tokens[mode].update(
                int(token) for token in predictions[replacement_mask].tolist()
            )

            valid_positions = torch.nonzero(valid, as_tuple=False)
            for batch_index, position in valid_positions.tolist():
                token_rows.append(
                    {
                        "mode": mode,
                        "sample_index": start + batch_index,
                        "position": position,
                        "atom_index": position - 1,
                        "is_cls": int(position == 0),
                        "selected_for_masking": int(selected[batch_index, position]),
                        "original_token_id": int(original_ids[batch_index, position]),
                        "corrupted_token_id": int(discriminator_ids[batch_index, position]),
                        "discriminator_label": int(
                            discriminator_labels[batch_index, position]
                        ),
                        "discriminator_probability": float(
                            discriminator_probabilities[batch_index, position]
                        ),
                    }
                )

            # A position after CLS corresponds to the preceding RDKit atom index.
            for batch_index, smiles in enumerate(batch_smiles):
                molecule = Chem.MolFromSmiles(smiles)
                assert molecule is not None
                positions = torch.nonzero(
                    discriminator_labels[batch_index], as_tuple=False
                ).flatten()
                for position in positions.tolist():
                    if position == 0 or position - 1 >= molecule.GetNumAtoms():
                        continue
                    predicted_token = int(predictions[batch_index, position])
                    original_signature = base.atom_signature(
                        molecule.GetAtomWithIdx(position - 1)
                    )
                    predicted_signature = signature_map.get(predicted_token)
                    if predicted_token in (0, 208, 209, 210):
                        mode_counts[mode]["special_token_replacements"] += 1
                        continue
                    if predicted_signature is None:
                        mode_counts[mode]["unknown_signature_replacements"] += 1
                        continue
                    mode_counts[mode]["known_signature_replacements"] += 1
                    field_mismatches = [
                        left != right
                        for left, right in zip(original_signature, predicted_signature)
                    ]
                    for field, differs in zip(MISMATCH_FIELDS, field_mismatches):
                        mismatch_counts[mode][field] += int(differs)
                    mismatch_counts[mode]["any"] += int(any(field_mismatches))

    threshold_grid = np.linspace(
        args.threshold_min, args.threshold_max, args.threshold_count
    )
    calibration_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    mode_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for mode in args.modes:
        probabilities = torch.cat(probabilities_by_mode[mode]).numpy().astype(np.float64)
        labels = torch.cat(labels_by_mode[mode]).numpy().astype(np.int64)
        bins, ece = calibration_bins(probabilities, labels, args.ece_bins)
        for row in bins:
            calibration_rows.append({"mode": mode, **row})

        sweep = [
            confusion_metrics(probabilities, labels, float(threshold))
            for threshold in threshold_grid
        ]
        for row in sweep:
            threshold_rows.append(
                {
                    "mode": mode,
                    "selection_scope": "descriptive_same_diagnostic_sample",
                    **row,
                }
            )
        at_half = confusion_metrics(probabilities, labels, 0.5)
        best_balanced = best_descriptive_row(sweep, "balanced_accuracy")
        best_f1 = best_descriptive_row(sweep, "f1")

        counts = mode_counts[mode]
        selected_tokens_count = common_counts["selected_tokens"]
        replaced_tokens_count = counts["replaced_tokens"]
        known = counts["known_signature_replacements"]
        brier = float(np.mean((probabilities - labels) ** 2))
        mode_metrics = {
            "sampled_original_token_rate_at_selected_tokens": base.safe_divide(
                counts["sampled_original_tokens"], selected_tokens_count
            ),
            "realized_replacement_rate_among_selected_tokens": base.safe_divide(
                replaced_tokens_count, selected_tokens_count
            ),
            "realized_replacement_rate_among_atom_tokens": base.safe_divide(
                replaced_tokens_count, common_counts["atom_tokens"]
            ),
            "realized_replacement_rate_among_all_valid_tokens": base.safe_divide(
                replaced_tokens_count, common_counts["valid_tokens"]
            ),
            "discriminator_binary_cross_entropy": base.safe_divide(
                counts["discriminator_bce_sum"], common_counts["valid_tokens"]
            ),
            "discriminator_brier_score": brier,
            "equal_width_ece": ece,
        }
        chemistry = {
            "known_signature_replacements": known,
            "special_token_replacements": counts["special_token_replacements"],
            "unknown_signature_replacements": counts[
                "unknown_signature_replacements"
            ],
            "rates_among_known_signature_replacements": {
                field: base.safe_divide(mismatch_counts[mode][field], known)
                for field in ("any", *MISMATCH_FIELDS)
            },
        }
        mode_results[mode] = {
            "sampling_seed": stable_mode_seed(mode, args.sampling_seed),
            "temperature": 1.0 if mode == "categorical" else None,
            "top_k": 5 if mode == "topk5" else None,
            "counts": {
                "valid_tokens": common_counts["valid_tokens"],
                "atom_tokens": common_counts["atom_tokens"],
                "selected_tokens": selected_tokens_count,
                "replaced_tokens": replaced_tokens_count,
                **{key: int(value) for key, value in at_half.items() if key.endswith("positive") or key.endswith("negative")},
            },
            "metrics": {**mode_metrics, **{f"threshold_0_5_{key}": value for key, value in at_half.items() if key not in {"threshold", "true_positive", "false_positive", "true_negative", "false_negative"}}},
            "selected_sample_token_diversity": counter_diversity(sampled_tokens[mode]),
            "replacement_token_diversity": counter_diversity(replacement_tokens[mode]),
            "replacement_token_top15": [
                {"token_id": token, "count": count}
                for token, count in replacement_tokens[mode].most_common(15)
            ],
            "chemistry_mismatch": chemistry,
            "calibration": {
                "binning": "equal_width",
                "n_bins": args.ece_bins,
                "ece": ece,
                "bins": bins,
            },
            "threshold_sweep": {
                "selection_scope": "descriptive_same_diagnostic_sample",
                "selection_warning": (
                    "Best thresholds are selected and evaluated on the same diagnostic "
                    "sample; they are not independently validated operating points."
                ),
                "grid_min": args.threshold_min,
                "grid_max": args.threshold_max,
                "grid_count": args.threshold_count,
                "threshold_0_5": at_half,
                "best_balanced_accuracy_descriptive": best_balanced,
                "best_f1_descriptive": best_f1,
            },
        }
        summary_rows.append(
            {
                "mode": mode,
                "sampling_seed": stable_mode_seed(mode, args.sampling_seed),
                "valid_tokens": common_counts["valid_tokens"],
                "selected_tokens": selected_tokens_count,
                "replaced_tokens": replaced_tokens_count,
                **mode_metrics,
                **{f"threshold_0_5_{key}": value for key, value in at_half.items() if key != "threshold"},
                "unique_sampled_tokens": len(sampled_tokens[mode]),
                "sampled_token_normalized_shannon_entropy": counter_diversity(
                    sampled_tokens[mode]
                )["normalized_shannon_entropy"],
                "unique_replacement_tokens": len(replacement_tokens[mode]),
                "chemistry_mismatch_any": chemistry[
                    "rates_among_known_signature_replacements"
                ]["any"],
                "best_balanced_accuracy_threshold_descriptive": best_balanced[
                    "threshold"
                ],
                "best_f1_threshold_descriptive": best_f1["threshold"],
            }
        )

    audited_sources = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        args.config,
        args.vocab,
        REPO_ROOT / "mole_public" / "mole" / "training" / "models" / "pretrain.py",
        REPO_ROOT
        / "mole_public"
        / "mole"
        / "training"
        / "data"
        / "pretrain_datasets.py",
        REPO_ROOT / "mole_public" / "mole" / "training" / "data" / "datasets.py",
    ]
    source_hashes = {display_path(path): base.sha256_file(path) for path in audited_sources}
    input_hashes = {
        display_path(path): base.sha256_file(path)
        for path in [args.checkpoint, args.config, args.vocab, *data_paths]
    }
    checkpoint_metadata.update(
        {
            "path": display_path(args.checkpoint),
            "sha256": input_hashes[display_path(args.checkpoint)],
        }
    )
    result = {
        "audit": "ChemRasayan Phase-4 RTD calibration and corruption sampling diagnostic",
        "scope_notes": [
            (
                "Frozen-checkpoint inference on the same endpoint-balanced local TDC "
                "molecule sample and masking contract as audit_rtd_objective.py; this is "
                "a downstream-distribution diagnostic, not a held-out ZINC estimate."
            ),
            (
                "Sampling-mode comparisons test the frozen discriminator under alternate "
                "corruptions; they do not estimate the effect of training with those modes."
            ),
            (
                "Threshold selection is descriptive on this same diagnostic sample and "
                "must not be presented as independently validated."
            ),
        ],
        "created_at_utc": started_at_utc,
        "git_revision": base.git_revision(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "rdkit": rdkit_version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "num_threads": args.num_threads,
            "elapsed_seconds": time.time() - started_at,
        },
        "parameters": {
            "sample_seed": args.sample_seed,
            "mask_seed": args.mask_seed,
            "sampling_seed": args.sampling_seed,
            "mask_probability": args.mask_probability,
            "batch_size": args.batch_size,
            "data_glob": args.data_glob,
            "modes": args.modes,
            "ece_bins": args.ece_bins,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_count": args.threshold_count,
            "special_token_ids": {"PAD": 0, "MASK": 208, "UNK": 209, "CLS": 210},
        },
        "checkpoint": checkpoint_metadata,
        "source_hashes_sha256": source_hashes,
        "input_hashes_sha256": input_hashes,
        "sample_provenance": sample_provenance,
        "signature_map_provenance": signature_provenance,
        "common_counts": dict(common_counts),
        "generator_cross_entropy": base.safe_divide(
            generator_nll_sum, common_counts["selected_tokens"]
        ),
        "modes": mode_results,
        "output_schema": {
            "token_rows": len(token_rows),
            "calibration_bin_rows": len(calibration_rows),
            "threshold_sweep_rows": len(threshold_rows),
        },
    }
    tables = {
        "summary": summary_rows,
        "tokens": token_rows,
        "bins": calibration_rows,
        "thresholds": threshold_rows,
    }
    return result, tables


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs_atomic(
    result: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    destinations: dict[str, Path],
) -> None:
    """Write all artifacts to sibling temp files before atomically replacing targets."""
    temporary: dict[str, Path] = {}
    try:
        for key, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(descriptor)
            temporary[key] = Path(temporary_name)

        with temporary["json"].open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        write_csv(temporary["summary"], tables["summary"])
        write_csv(temporary["tokens"], tables["tokens"])
        write_csv(temporary["bins"], tables["bins"])
        write_csv(temporary["thresholds"], tables["thresholds"])

        for key, destination in destinations.items():
            os.replace(temporary[key], destination)
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG)
    parser.add_argument("--vocab", type=Path, default=base.DEFAULT_VOCAB)
    parser.add_argument("--data-glob", default=str(REPO_ROOT / "data" / "*.tab"))
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--signature-per-task", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260917)
    parser.add_argument("--mask-seed", type=int, default=20260917)
    parser.add_argument("--sampling-seed", type=int, default=20260917)
    parser.add_argument("--mask-probability", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=list(SUPPORTED_MODES),
        help="Comma-separated subset of argmax,categorical,topk5.",
    )
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-count", type=int, default=101)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-tokens-csv", type=Path, default=DEFAULT_TOKENS_CSV)
    parser.add_argument("--output-bins-csv", type=Path, default=DEFAULT_BINS_CSV)
    parser.add_argument(
        "--output-thresholds-csv", type=Path, default=DEFAULT_THRESHOLDS_CSV
    )
    args = parser.parse_args()
    if args.ece_bins <= 0:
        parser.error("--ece-bins must be positive")
    if args.threshold_count < 2:
        parser.error("--threshold-count must be at least 2")
    if not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        parser.error("threshold range must satisfy 0 <= min < max <= 1")
    if not 0.0 <= args.mask_probability <= 1.0:
        parser.error("--mask-probability must be in [0, 1]")
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    args.vocab = args.vocab.resolve()
    return args


def main() -> None:
    args = parse_args()
    result, tables = run_audit(args)
    destinations = {
        "json": args.output_json,
        "summary": args.output_summary_csv,
        "tokens": args.output_tokens_csv,
        "bins": args.output_bins_csv,
        "thresholds": args.output_thresholds_csv,
    }
    write_outputs_atomic(result, tables, destinations)
    for destination in destinations.values():
        print(f"Wrote {destination}")
    print(json.dumps({mode: value["metrics"] for mode, value in result["modes"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
