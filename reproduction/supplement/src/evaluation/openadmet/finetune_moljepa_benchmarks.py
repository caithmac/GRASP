#!/usr/bin/env python3
"""Architecture-aware MolE-RTD evaluation on one Mol-JEPA regression endpoint.

Each paper train/test split gets a cluster-held-out validation subset.  Frozen,
LoRA, and full encoder adaptation are all evaluated, but the reported primary
method is selected by validation MAE only.  Test labels never influence method
selection or early stopping.  Completed method/split cells are atomic and
restart-safe.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch_geometric
from rdkit import Chem
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from DeBERTa.deberta.config import ModelConfig
from encoder_arch import load_encoder_state, resolve_encoder_config
from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary
from mole.training.models.mole import AtomEnvEmbeddings
from moljepa_benchmark_config import ENDPOINT_BY_SLUG, PAPER_PROTOCOL


TASK_NAME = os.environ.get("TASK_NAME", "expansion_logd")
DATA_DIR = Path(os.environ.get("MOLJEPA_DATA_DIR", "/mnt/data/moljepa_benchmarks/prepared"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/mnt/results_rtd_phase4_95m_moljepa"))
RTD_CKPT = Path(os.environ.get(
    "RTD_CKPT",
    "/mnt/checkpoints/sup_pretrain_rtd25_1.5b_phase4/encoder_step_050000.pt",
))
METHODS = tuple(x.strip() for x in os.environ.get(
    "METHODS", "frozen_mix,lora_mix,full_mix"
).split(",") if x.strip())
MIX_LAYERS = tuple(int(x) for x in os.environ.get("MIX_LAYERS", "4,8,10,12").split(","))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
PATIENCE = int(os.environ.get("PATIENCE", "12"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
VALID_FRAC = float(os.environ.get("VALID_FRAC", "0.15"))
HEAD_HIDDEN = int(os.environ.get("HEAD_HIDDEN", "512"))
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
USE_BF16 = os.environ.get("USE_BF16", "1") == "1"
ATOM_ORDER_PERMUTATIONS = int(os.environ.get("ATOM_ORDER_PERMUTATIONS", "0"))
LEARNING_RATES = {
    "frozen_mix": float(os.environ.get("FROZEN_LR", "1e-3")),
    "lora_mix": float(os.environ.get("LORA_LR", "5e-5")),
    "full_mix": float(os.environ.get("FULL_LR", "1e-5")),
}
METHOD_SEED_OFFSETS = {
    "frozen_mix": 0,
    "lora_mix": 1,
    "full_mix": 2,
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        update = (self.dropout(inputs) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(inputs) + self.scale * update


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, mask):
        logits = self.score(hidden).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * hidden).sum(dim=1)


class MolERTDRegressor(nn.Module):
    def __init__(self, encoder_state: OrderedDict, adaptation: str):
        super().__init__()
        config = resolve_encoder_config(encoder_state, "auto")
        cfg = ModelConfig.from_dict(config)
        if min(MIX_LAYERS) < 1 or max(MIX_LAYERS) > cfg.num_hidden_layers:
            raise ValueError(
                f"MIX_LAYERS={MIX_LAYERS} invalid for {cfg.num_hidden_layers}-layer encoder"
            )
        self.adaptation = adaptation
        self.encoder = AtomEnvEmbeddings(cfg)
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"strict encoder load failed: missing={missing} unexpected={unexpected}"
            )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if adaptation == "lora_mix":
            for layer in self.encoder.encoder.layer:
                attention = layer.attention.self
                attention.query_proj = LoRALinear(
                    attention.query_proj, LORA_RANK, LORA_ALPHA, LORA_DROPOUT
                )
                attention.value_proj = LoRALinear(
                    attention.value_proj, LORA_RANK, LORA_ALPHA, LORA_DROPOUT
                )
        elif adaptation == "full_mix":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        elif adaptation != "frozen_mix":
            raise ValueError(f"unsupported method {adaptation}")

        self.layer_logits = nn.Parameter(torch.zeros(len(MIX_LAYERS)))
        self.pooler = AttentionPool(cfg.hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HEAD_HIDDEN, 1),
        )
        self.architecture = {
            "num_hidden_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "mix_layers": list(MIX_LAYERS),
        }

    def forward(self, batch):
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        if self.adaptation == "frozen_mix":
            self.encoder.eval()
            with torch.no_grad():
                output = self.encoder(
                    input_ids, input_mask, attention_mask=input_mask, relative_pos=relative_pos
                )
        else:
            output = self.encoder(
                input_ids, input_mask, attention_mask=input_mask, relative_pos=relative_pos
            )
        weights = torch.softmax(self.layer_logits, dim=0)
        hidden = sum(
            weights[index] * output["hidden_states"][layer - 1]
            for index, layer in enumerate(MIX_LAYERS)
        )
        atom_mask = input_mask.bool().clone()
        atom_mask[:, 0] = False  # RTD never trained the synthetic CLS token.
        pooled = self.pooler(hidden, atom_mask)
        return self.head(pooled).squeeze(-1)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def valid_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def run_config(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "learning_rate": LEARNING_RATES[method],
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "validation_fraction": VALID_FRAC,
        "head_hidden": HEAD_HIDDEN,
        "mix_layers": list(MIX_LAYERS),
        "weight_decay": WEIGHT_DECAY,
        "use_bf16": USE_BF16,
        "lora_rank": LORA_RANK if method == "lora_mix" else None,
        "lora_alpha": LORA_ALPHA if method == "lora_mix" else None,
        "lora_dropout": LORA_DROPOUT if method == "lora_mix" else None,
        "atom_order_permutations": ATOM_ORDER_PERMUTATIONS,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smiles_sha256(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def resolve_vocab_path() -> str:
    override = os.environ.get("VOCAB_PATH")
    if override:
        return override
    import mole
    return str(
        Path(mole.__path__[0])
        / "training/data/vocabularies/vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cluster_validation_indices(train_frame: pd.DataFrame, seed: int):
    """Hold out entire Butina clusters until approximately VALID_FRAC rows."""
    groups = train_frame.groupby("cluster_index").indices
    cluster_ids = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)
    target = max(1, round(len(train_frame) * VALID_FRAC))
    selected: list[int] = []
    count = 0
    for cluster_id in cluster_ids:
        size = len(groups[cluster_id])
        if count < target or not selected:
            selected.append(cluster_id)
            count += size
        if count >= target:
            break
    is_valid = train_frame["cluster_index"].isin(selected).to_numpy()
    if is_valid.all() or not is_valid.any():
        raise ValueError("cluster validation split produced an empty train or validation set")
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)


def make_loader(frame: pd.DataFrame, labels: np.ndarray, dictionary, shuffle: bool):
    dataset = MolDataset(
        smiles=frame["smiles"].reset_index(drop=True),
        dictionary_inp=dictionary,
        radius_inp=0,
        useFeatures_inp=False,
        cls_token=True,
        labels=np.asarray(labels, dtype=np.float32),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, device, scaler: StandardScaler):
    model.eval()
    predictions = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda" and USE_BF16,
        ):
            output = model(batch)
        predictions.append(output.float().cpu().numpy())
    scaled = np.concatenate(predictions).reshape(-1, 1)
    return scaler.inverse_transform(scaled).ravel()


def randomized_atom_order_smiles(values, seed: int) -> list[str]:
    """Create deterministic atom-renumbered SMILES without changing chemistry."""
    rng = np.random.default_rng(seed)
    randomized = []
    for row, value in enumerate(map(str, values)):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"invalid SMILES in atom-order audit at row {row}: {value}")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        order = rng.permutation(mol.GetNumAtoms()).astype(int).tolist()
        reordered = Chem.RenumberAtoms(mol, order)
        candidate = Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True)
        check = Chem.MolFromSmiles(candidate)
        if check is None or Chem.MolToSmiles(
            check, canonical=True, isomericSmiles=True
        ) != canonical:
            raise RuntimeError(f"atom renumbering changed chemistry at row {row}")
        randomized.append(candidate)
    return randomized


def train_cell(
    encoder_state, dictionary, train_frame, valid_frame, test_frame,
    method: str, seed: int,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(train_frame[["y"]].to_numpy(dtype=np.float32))
    train_y = scaler.transform(train_frame[["y"]]).ravel()
    valid_y = scaler.transform(valid_frame[["y"]]).ravel()
    test_y = scaler.transform(test_frame[["y"]]).ravel()
    train_loader = make_loader(train_frame, train_y, dictionary, True)
    valid_loader = make_loader(valid_frame, valid_y, dictionary, False)
    test_loader = make_loader(test_frame, test_y, dictionary, False)

    model = MolERTDRegressor(encoder_state, method).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATES[method], weight_decay=WEIGHT_DECAY
    )
    total_steps = max(1, EPOCHS * len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    step = 0
    best = None
    best_state = None
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.target_labels.float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda" and USE_BF16,
            ):
                output = model(batch)
                loss = nn.functional.smooth_l1_loss(output, target.reshape_as(output))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            step += 1
            warmup_scale = min(1.0, step / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATES[method] * warmup_scale
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_pred = predict(model, valid_loader, device, scaler)
        valid_mae = float(mean_absolute_error(valid_frame["y"], valid_pred))
        print(
            f"task={TASK_NAME} method={method} seed={seed} epoch={epoch:02d}/{EPOCHS} "
            f"loss={np.mean(losses):.5f} val_mae={valid_mae:.5f}",
            flush=True,
        )
        if best is None or valid_mae < best["mae"] - 1e-8:
            best = {"epoch": epoch, "mae": valid_mae}
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("training completed without a validation state")
    model.load_state_dict(best_state)
    test_pred = predict(model, test_loader, device, scaler)
    test_true = test_frame["y"].to_numpy(dtype=float)
    test_mae = float(mean_absolute_error(test_true, test_pred))
    atom_order = None
    if ATOM_ORDER_PERMUTATIONS > 0:
        permutation_predictions = []
        permutation_records = []
        for permutation_index in range(ATOM_ORDER_PERMUTATIONS):
            permutation_seed = seed * 1000 + permutation_index
            permuted_smiles = randomized_atom_order_smiles(
                test_frame["smiles"], seed=permutation_seed
            )
            permuted_frame = test_frame.copy()
            permuted_frame["smiles"] = permuted_smiles
            permuted_loader = make_loader(permuted_frame, test_y, dictionary, False)
            permuted_pred = predict(model, permuted_loader, device, scaler)
            permutation_predictions.append(permuted_pred)
            permutation_records.append({
                "permutation_index": permutation_index,
                "seed": permutation_seed,
                "smiles_sha256": smiles_sha256(permuted_smiles),
                "test_mae": float(mean_absolute_error(test_true, permuted_pred)),
                "mean_absolute_prediction_delta": float(
                    np.mean(np.abs(permuted_pred - test_pred))
                ),
                "maximum_absolute_prediction_delta": float(
                    np.max(np.abs(permuted_pred - test_pred))
                ),
                "changed_smiles": int(sum(
                    str(original) != permuted
                    for original, permuted in zip(test_frame["smiles"], permuted_smiles)
                )),
                "fraction_changed_smiles": float(np.mean([
                    str(original) != permuted
                    for original, permuted in zip(test_frame["smiles"], permuted_smiles)
                ])),
                "y_pred": permuted_pred.astype(float).tolist(),
            })
        stacked = np.stack(permutation_predictions, axis=0)
        atom_order = {
            "protocol": (
                "RDKit RenumberAtoms followed by non-canonical isomeric SMILES; "
                "canonical isomeric identity asserted"
            ),
            "permutations": ATOM_ORDER_PERMUTATIONS,
            "canonical_test_mae": test_mae,
            "mean_permuted_test_mae": float(np.mean([
                record["test_mae"] for record in permutation_records
            ])),
            "mean_mae_change": float(np.mean([
                record["test_mae"] - test_mae for record in permutation_records
            ])),
            "per_molecule_prediction_sd": np.std(stacked, axis=0).astype(float).tolist(),
            "per_molecule_maximum_absolute_delta": np.max(
                np.abs(stacked - test_pred[None, :]), axis=0
            ).astype(float).tolist(),
            "records": permutation_records,
        }
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "method": method,
        "learning_rate": LEARNING_RATES[method],
        "seed": seed,
        "best_epoch": best["epoch"],
        "validation_mae": best["mae"],
        "test_mae": test_mae,
        "y_true": test_true.tolist(),
        "y_pred": test_pred.astype(float).tolist(),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "architecture": model.architecture,
        "atom_order_robustness": atom_order,
    }


def main() -> None:
    if TASK_NAME not in ENDPOINT_BY_SLUG:
        raise ValueError(f"unknown TASK_NAME={TASK_NAME}; choose from {sorted(ENDPOINT_BY_SLUG)}")
    unknown_methods = set(METHODS).difference(LEARNING_RATES)
    if unknown_methods:
        raise ValueError(f"unsupported METHODS={sorted(unknown_methods)}")
    if not RTD_CKPT.is_file():
        raise FileNotFoundError(RTD_CKPT)
    manifest_path = DATA_DIR / "manifest.json"
    data_path = DATA_DIR / f"{TASK_NAME}.csv"
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"prepared benchmark missing: {manifest_path} or {data_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    endpoint_manifest = manifest["endpoints"].get(TASK_NAME)
    if endpoint_manifest is None:
        raise ValueError(f"prepared manifest has no endpoint {TASK_NAME}")
    if file_sha256(data_path) != endpoint_manifest["prepared_sha256"]:
        raise ValueError(f"prepared CSV hash mismatch: {data_path}")

    frame = pd.read_csv(data_path)
    required = {"smiles", "y", "cluster_index", *PAPER_PROTOCOL["splits"]}
    if required.difference(frame.columns):
        raise ValueError(f"prepared data lacks {sorted(required.difference(frame.columns))}")
    encoder_state = load_encoder_state(str(RTD_CKPT))
    if encoder_state is None:
        raise RuntimeError(f"could not load encoder state from {RTD_CKPT}")
    dictionary = open_dictionary(resolve_vocab_path())
    checkpoint_hash = file_sha256(RTD_CKPT)
    runner_hash = file_sha256(Path(__file__).resolve())
    endpoint = ENDPOINT_BY_SLUG[TASK_NAME]
    task_dir = OUTPUT_DIR / TASK_NAME
    task_dir.mkdir(parents=True, exist_ok=True)

    split_records = {}
    for split_index, split_name in enumerate(PAPER_PROTOCOL["splits"], start=1):
        paper_train = frame[frame[split_name].eq("train")].reset_index(drop=True)
        test_frame = frame[frame[split_name].eq("test")].reset_index(drop=True)
        inner_train_idx, valid_idx = cluster_validation_indices(
            paper_train, seed=7300 + split_index
        )
        train_frame = paper_train.iloc[inner_train_idx].reset_index(drop=True)
        valid_frame = paper_train.iloc[valid_idx].reset_index(drop=True)
        cells = {}
        for method in METHODS:
            # Keep a method's seed stable when wrappers evaluate a subset of METHODS.
            # Positional seeding made full_mix use +2 in the main benchmark but +0 in
            # one-method ablations, confounding cross-table comparisons.
            seed = 9100 + split_index * 100 + METHOD_SEED_OFFSETS[method]
            cell_path = task_dir / "cells" / f"{split_name}_{method}.json"
            cell = valid_json(cell_path)
            if cell is not None and not (
                cell.get("checkpoint_sha256") == checkpoint_hash
                and cell.get("data_sha256") == endpoint_manifest["prepared_sha256"]
                and cell.get("runner_sha256") == runner_hash
                and cell.get("seed") == seed
                and cell.get("run_config") == run_config(method)
            ):
                print(f"STALE cell will be recomputed: {cell_path}", flush=True)
                cell = None
            if cell is None:
                cell = train_cell(
                    encoder_state, dictionary, train_frame, valid_frame, test_frame,
                    method, seed=seed,
                )
                cell.update({
                    "task": TASK_NAME,
                    "split": split_name,
                    "checkpoint": str(RTD_CKPT),
                    "checkpoint_sha256": checkpoint_hash,
                    "data_sha256": endpoint_manifest["prepared_sha256"],
                    "runner_sha256": runner_hash,
                    "run_config": run_config(method),
                    "train_rows": len(train_frame),
                    "validation_rows": len(valid_frame),
                    "test_rows": len(test_frame),
                    "train_smiles_sha256": smiles_sha256(train_frame["smiles"]),
                    "validation_smiles_sha256": smiles_sha256(valid_frame["smiles"]),
                    "test_smiles_sha256": smiles_sha256(test_frame["smiles"]),
                })
                atomic_json(cell_path, cell)
            else:
                print(f"SKIP completed {TASK_NAME} {split_name} {method}", flush=True)
            cells[method] = cell

        selected_method = min(METHODS, key=lambda method: cells[method]["validation_mae"])
        split_records[split_name] = {
            "selection_rule": "minimum cluster-held-out validation MAE",
            "selected_method": selected_method,
            "selected_validation_mae": cells[selected_method]["validation_mae"],
            "selected_test_mae": cells[selected_method]["test_mae"],
            "methods": cells,
        }

    selected_scores = [record["selected_test_mae"] for record in split_records.values()]
    method_summary = {}
    for method in METHODS:
        scores = [record["methods"][method]["test_mae"] for record in split_records.values()]
        method_summary[method] = {
            "mean_test_mae": float(np.mean(scores)),
            "std_test_mae": float(np.std(scores)),
            "split_test_mae": scores,
        }
    result = {
        "schema_version": 1,
        "task": TASK_NAME,
        "family": endpoint.family,
        "display_name": endpoint.display_name,
        "primary_metric": "mae",
        "checkpoint": str(RTD_CKPT),
        "checkpoint_sha256": checkpoint_hash,
        "runner_sha256": runner_hash,
        "run_configs": {method: run_config(method) for method in METHODS},
        "data_protocol": manifest["protocol"],
        "exact_table3_split_reproduction": manifest["exact_table3_split_reproduction"],
        "source_count_matches_paper": endpoint_manifest["source_count_matches_paper"],
        "software_versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "selection_rule": "method selected separately per split using validation MAE only",
        "selected_mean_test_mae": float(np.mean(selected_scores)),
        "selected_std_test_mae": float(np.std(selected_scores)),
        "selected_split_test_mae": selected_scores,
        "method_summary": method_summary,
        "splits": split_records,
    }
    atomic_json(OUTPUT_DIR / f"results_{TASK_NAME}.json", result)
    print(json.dumps({
        "task": TASK_NAME,
        "protocol": result["data_protocol"],
        "selected_mae": result["selected_mean_test_mae"],
        "selected_std": result["selected_std_test_mae"],
        "selected_methods": {
            split: record["selected_method"] for split, record in split_records.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
