#!/usr/bin/env python
"""Audit the realized ChemRasayan RTD objective on local molecular data.

This script is intentionally evaluation-only.  It loads a frozen RTD checkpoint,
applies deterministic masking to an endpoint-balanced sample, and reports the
generator recovery rate, the discriminator's realized class balance/confusion
matrix, losses, replacement diversity, and token/host-atom chemistry mismatch.

Example (from the repository root):

    python scripts/audit_rtd_objective.py

The default checkpoint is the final Phase-4 Step-1 checkpoint used to construct
the paper's 93.5M-parameter encoder.  No cluster access or training is required.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for dependency_root in (REPO_ROOT, REPO_ROOT / "mole_public", REPO_ROOT / "DeBERTa"):
    sys.path.insert(0, str(dependency_root))

import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from rdkit import Chem, RDLogger, __version__ as rdkit_version
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from mole.training.data.datasets import getAtomEnvironments
from mole.training.data.pretrain_datasets import MolPreTrainDataset
from mole.training.models.pretrain import MolERTDModel


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "hf_mole_rtd_zinc1_5b"
    / "step1"
    / "phase4"
    / "phase4_final_weights.ckpt"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "mole_public"
    / "mole"
    / "training"
    / "configs"
    / "model"
    / "pretrain_rtd_25pct_phase4.yaml"
)
DEFAULT_VOCAB = (
    REPO_ROOT
    / "mole_public"
    / "mole"
    / "training"
    / "data"
    / "vocabularies"
    / "vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
)
DEFAULT_JSON = REPO_ROOT / "results_package" / "rtd_objective_audit_phase4_tdc.json"
DEFAULT_CSV = REPO_ROOT / "results_package" / "rtd_objective_audit_phase4_tdc.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atom_signature(atom: Chem.Atom) -> tuple[Any, ...]:
    return (
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalValence(),
        atom.GetTotalNumHs(),
        int(atom.IsInRing()),
        str(atom.GetHybridization()),
    )


def load_smiles_column(path: Path) -> pd.Series:
    frame = pd.read_csv(path, sep="\t")
    column = "Drug" if "Drug" in frame.columns else frame.columns[1]
    return frame[column].dropna().astype(str).drop_duplicates()


def build_evaluation_sample(
    data_paths: list[Path], per_task: int, sample_seed: int
) -> tuple[list[str], dict[str, Any]]:
    pooled: list[str] = []
    per_file: list[dict[str, Any]] = []
    for path in data_paths:
        values = load_smiles_column(path)
        n_selected = min(per_task, len(values))
        selected = values.sample(n=n_selected, random_state=sample_seed).tolist()
        pooled.extend(selected)
        per_file.append(
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "unique_smiles_available": int(len(values)),
                "selected_before_cross_file_deduplication": n_selected,
            }
        )

    # Preserve sorted-file/sample order while removing molecules shared by tasks.
    deduplicated = list(dict.fromkeys(pooled))
    manifest_text = "\n".join(item["sha256"] for item in per_file).encode("ascii")
    return deduplicated, {
        "files": per_file,
        "aggregate_file_hash": hashlib.sha256(manifest_text).hexdigest(),
        "per_task_requested": per_task,
        "selected_before_cross_file_deduplication": len(pooled),
        "selected_after_cross_file_deduplication": len(deduplicated),
        "sample_seed": sample_seed,
    }


def build_signature_map(
    data_paths: list[Path], vocabulary: dict[Any, int], per_task: int
) -> tuple[dict[int, tuple[Any, ...]], dict[str, int]]:
    signature_counts: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    molecules_seen = 0
    for path in data_paths:
        for smiles in load_smiles_column(path).head(per_task):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            tokens = getAtomEnvironments(molecule, vocabulary, 0, False)
            if len(tokens) != molecule.GetNumAtoms():
                continue
            for token, atom in zip(tokens, molecule.GetAtoms()):
                if token not in (0, 208, 209, 210):
                    signature_counts[int(token)][atom_signature(atom)] += 1
            molecules_seen += 1
    signature_map = {
        token: counts.most_common(1)[0][0]
        for token, counts in signature_counts.items()
    }
    return signature_map, {
        "molecules_scanned": molecules_seen,
        "tokens_covered": len(signature_map),
        "vocabulary_atom_tokens": 207,
        "molecules_per_task_limit": per_task,
    }


def load_model(checkpoint: Path, config: Path) -> tuple[MolERTDModel, dict[str, Any]]:
    with config.open("r", encoding="utf-8") as handle:
        config_document = yaml.safe_load(handle)
    model_config = dict(
        config_document["model"]["hyperparameters"]["pl_module"]["model"]
    )
    model_config.pop("_target_", None)
    model_config["vocab_size_inp"] = 211
    model_config["mask_token_id"] = 208

    checkpoint_object = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint_object["state_dict"].items()
    }
    model = MolERTDModel(**model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    metadata = {
        "global_step": int(checkpoint_object.get("global_step", -1)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "model_config": model_config,
    }
    del checkpoint_object
    return model, metadata


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.mask_seed)
    np.random.seed(args.mask_seed)
    random.seed(args.mask_seed)
    torch.set_num_threads(args.num_threads)

    data_paths = [Path(path).resolve() for path in sorted(glob.glob(args.data_glob))]
    if not data_paths:
        raise FileNotFoundError(f"No data files matched {args.data_glob!r}")

    import pickle

    with args.vocab.open("rb") as handle:
        vocabulary = pickle.load(handle)  # trusted repository artifact
    vocabulary.update({"PAD": 0, "MASK": 208, "UNK": 209, "CLS": 210})

    smiles_sample, sample_provenance = build_evaluation_sample(
        data_paths, args.per_task, args.sample_seed
    )
    signature_map, signature_provenance = build_signature_map(
        data_paths, vocabulary, args.signature_per_task
    )
    model, checkpoint_metadata = load_model(args.checkpoint, args.config)

    counts: collections.Counter = collections.Counter()
    wrong_predictions: collections.Counter = collections.Counter()
    correct_confidences: list[float] = []
    wrong_confidences: list[float] = []
    mismatch: collections.Counter = collections.Counter()
    mismatch_fields = [
        "atomic_number",
        "degree",
        "formal_charge",
        "aromaticity",
        "total_valence",
        "total_hydrogens",
        "ring_membership",
        "hybridization",
    ]
    known_replacements = 0
    special_replacements = 0
    unknown_replacements = 0

    started_at = time.time()
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
        original_ids, _ = to_dense_batch(
            batch.original_ids, batch.batch, fill_value=0
        )
        mlm_labels, _ = to_dense_batch(
            batch.mlm_labels, batch.batch, fill_value=-100
        )
        relative_positions = to_dense_adj(
            batch.edge_index, batch.batch, batch.edge_attr
        )
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length).expand(batch_size, sequence_length)
        selected = mlm_labels.ne(-100)

        with torch.inference_mode():
            generator_hidden = model.generator(
                input_ids,
                input_mask,
                attention_mask=input_mask,
                relative_pos=relative_positions,
            )["hidden_states"][-1]
            generator_logits = model.gen_head(
                generator_hidden, position_ids=position_ids
            )
            generator_predictions = generator_logits.argmax(dim=-1)
            discriminator_ids = original_ids.clone()
            discriminator_ids[selected] = generator_predictions[selected]
            discriminator_labels = (
                (discriminator_ids != original_ids) & input_mask.bool()
            )
            discriminator_hidden = model.discriminator(
                discriminator_ids,
                input_mask,
                attention_mask=input_mask,
                relative_pos=relative_positions,
            )["hidden_states"][-1]
            discriminator_logits = model.disc_head(
                discriminator_hidden, position_ids=position_ids
            )

        valid = input_mask.bool()
        predicted_positive = discriminator_logits.gt(0) & valid
        counts["valid_tokens"] += int(valid.sum())
        counts["atom_tokens"] += int(valid.sum()) - batch_size
        counts["selected_tokens"] += int(selected.sum())
        counts["generator_correct"] += int(
            (generator_predictions[selected] == original_ids[selected]).sum()
        )
        counts["replaced_tokens"] += int(discriminator_labels.sum())
        counts["true_positive"] += int(
            (predicted_positive & discriminator_labels).sum()
        )
        counts["false_positive"] += int(
            (predicted_positive & ~discriminator_labels & valid).sum()
        )
        counts["true_negative"] += int(
            (~predicted_positive & ~discriminator_labels & valid).sum()
        )
        counts["false_negative"] += int(
            (~predicted_positive & discriminator_labels).sum()
        )
        counts["generator_nll_sum"] += float(
            torch.nn.functional.cross_entropy(
                generator_logits[selected], original_ids[selected], reduction="sum"
            )
        )
        counts["discriminator_bce_sum"] += float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                discriminator_logits[valid],
                discriminator_labels[valid].float(),
                reduction="sum",
            )
        )

        selected_probabilities = torch.softmax(generator_logits[selected], dim=-1)
        top_confidence = selected_probabilities.max(dim=-1).values
        generator_correct = generator_predictions[selected].eq(original_ids[selected])
        correct_confidences.extend(top_confidence[generator_correct].tolist())
        wrong_confidences.extend(top_confidence[~generator_correct].tolist())
        for token in generator_predictions[
            selected & (generator_predictions != original_ids)
        ].tolist():
            wrong_predictions[int(token)] += 1

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
                predicted_token = int(generator_predictions[batch_index, position])
                original_signature = atom_signature(
                    molecule.GetAtomWithIdx(position - 1)
                )
                predicted_signature = signature_map.get(predicted_token)
                if predicted_token in (0, 208, 209, 210):
                    special_replacements += 1
                    continue
                if predicted_signature is None:
                    unknown_replacements += 1
                    continue
                known_replacements += 1
                field_mismatches = [
                    left != right
                    for left, right in zip(original_signature, predicted_signature)
                ]
                for field, differs in zip(mismatch_fields, field_mismatches):
                    mismatch[field] += int(differs)
                mismatch["any"] += int(any(field_mismatches))

    valid_tokens = counts["valid_tokens"]
    atom_tokens = counts["atom_tokens"]
    selected_tokens = counts["selected_tokens"]
    replaced_tokens = counts["replaced_tokens"]
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    positive_rate = safe_divide(replaced_tokens, valid_tokens)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    metrics = {
        "mask_rate_among_atom_tokens": safe_divide(selected_tokens, atom_tokens),
        "generator_top1_accuracy_at_selected_tokens": safe_divide(
            counts["generator_correct"], selected_tokens
        ),
        "generator_original_token_recovery_rate": safe_divide(
            counts["generator_correct"], selected_tokens
        ),
        "realized_replacement_rate_among_selected_tokens": safe_divide(
            replaced_tokens, selected_tokens
        ),
        "realized_replacement_rate_among_atom_tokens": safe_divide(
            replaced_tokens, atom_tokens
        ),
        "realized_replacement_rate_among_all_valid_tokens": positive_rate,
        "discriminator_original_class_fraction": 1.0 - positive_rate,
        "discriminator_accuracy": safe_divide(tp + tn, valid_tokens),
        "discriminator_all_original_baseline_accuracy": 1.0 - positive_rate,
        "discriminator_precision": safe_divide(tp, tp + fp),
        "discriminator_recall": recall,
        "discriminator_specificity": specificity,
        "discriminator_balanced_accuracy": 0.5 * (recall + specificity),
        "generator_cross_entropy": safe_divide(
            counts["generator_nll_sum"], selected_tokens
        ),
        "discriminator_binary_cross_entropy": safe_divide(
            counts["discriminator_bce_sum"], valid_tokens
        ),
        "mean_generator_top1_confidence_when_correct": safe_divide(
            sum(correct_confidences), len(correct_confidences)
        ),
        "mean_generator_top1_confidence_when_wrong": safe_divide(
            sum(wrong_confidences), len(wrong_confidences)
        ),
        "unique_wrong_prediction_tokens": len(wrong_predictions),
        "wrong_prediction_top1_token_share": safe_divide(
            wrong_predictions.most_common(1)[0][1], replaced_tokens
        ),
        "wrong_prediction_top5_token_share": safe_divide(
            sum(value for _, value in wrong_predictions.most_common(5)),
            replaced_tokens,
        ),
    }

    chemistry_mismatch = {
        "known_signature_replacements": known_replacements,
        "special_token_replacements": special_replacements,
        "unknown_signature_replacements": unknown_replacements,
        "rates_among_known_signature_replacements": {
            field: safe_divide(mismatch[field], known_replacements)
            for field in ["any", *mismatch_fields]
        },
    }

    audited_sources = [
        Path(__file__).resolve(),
        args.checkpoint,
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
    source_hashes = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in audited_sources
    }
    checkpoint_key = str(args.checkpoint.relative_to(REPO_ROOT)).replace("\\", "/")
    checkpoint_metadata.update(
        {
            "path": checkpoint_key,
            "sha256": source_hashes[checkpoint_key],
        }
    )

    return {
        "audit": "ChemRasayan Phase-4 RTD realized-objective diagnostic",
        "scope_note": (
            "Frozen-checkpoint inference on an endpoint-balanced local TDC molecule "
            "sample; this is a downstream-distribution diagnostic, not a held-out "
            "ZINC validation estimate."
        ),
        "git_revision": git_revision(),
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
            "mask_probability": args.mask_probability,
            "batch_size": args.batch_size,
            "data_glob": args.data_glob,
            "special_token_ids": {"PAD": 0, "MASK": 208, "UNK": 209, "CLS": 210},
        },
        "checkpoint": checkpoint_metadata,
        "source_hashes_sha256": source_hashes,
        "sample_provenance": sample_provenance,
        "signature_map_provenance": signature_provenance,
        "counts": dict(counts),
        "metrics": metrics,
        "wrong_prediction_top15": [
            {"token_id": token, "count": count}
            for token, count in wrong_predictions.most_common(15)
        ],
        "chemistry_mismatch": chemistry_mismatch,
    }


def write_outputs(result: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    flattened: dict[str, Any] = {
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "global_step": result["checkpoint"]["global_step"],
        "sample_seed": result["parameters"]["sample_seed"],
        "mask_seed": result["parameters"]["mask_seed"],
        "molecules": result["sample_provenance"][
            "selected_after_cross_file_deduplication"
        ],
    }
    flattened.update(result["counts"])
    flattened.update(result["metrics"])
    for field, value in result["chemistry_mismatch"][
        "rates_among_known_signature_replacements"
    ].items():
        flattened[f"chemistry_mismatch_{field}"] = value
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--data-glob", default=str(REPO_ROOT / "data" / "*.tab")
    )
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--signature-per-task", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260917)
    parser.add_argument("--mask-seed", type=int, default=20260917)
    parser.add_argument("--mask-probability", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    write_outputs(result, args.output_json, args.output_csv)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
