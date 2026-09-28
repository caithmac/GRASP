"""Protocol-matched TDC evaluation for the Phase-4/50K MolE-RTD encoder.

This runner deliberately differs from finetune_benchmarks.py:
* fixed TDC ADMET_Group test sets;
* five validation folds for LR/dropout selection;
* final-layer CLS pooling and MolE-style GELU head;
* Adam, zero weight decay, 10% warmup then constant LR;
* batch 32, 100 epochs, MSE regression loss;
* no test evaluation until a task configuration is locked.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tdc
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from DeBERTa.deberta.config import ModelConfig
from encoder_arch import load_encoder_state, resolve_encoder_config
from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary
from mole.training.models.mole import AtomEnvEmbeddings


TASK_NAME = os.environ.get("TASK_NAME", "bbb_martins").lower()
RTD_CKPT = os.environ.get(
    "RTD_CKPT",
    "/mnt/checkpoints/sup_pretrain_rtd25_1.5b_phase4/encoder_step_050000.pt",
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/mnt/results_rtd_phase4_95m_mole_protocol"))
TDC_DATA_DIR = os.environ.get("TDC_DATA_DIR", "/mnt/tdc_data")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
FOLDS = tuple(int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4").split(","))
TEST_SEEDS = tuple(int(x) for x in os.environ.get("TEST_SEEDS", "100,101,102").split(","))
LEARNING_RATES = tuple(float(x) for x in os.environ.get("LEARNING_RATES", "1e-5,3e-6,1e-6").split(","))
DROPOUTS = tuple(float(x) for x in os.environ.get("DROPOUTS", "0,0.1,0.15").split(","))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5"))

TASK_CONFIG = {
    "ames": ("classification", "auroc"),
    "bbb_martins": ("classification", "auroc"),
    "bioavailability_ma": ("classification", "auroc"),
    "cyp3a4_substrate_carbonmangels": ("classification", "auroc"),
    "dili": ("classification", "auroc"),
    "hia_hou": ("classification", "auroc"),
    "pgp_broccatelli": ("classification", "auroc"),
    "herg": ("classification", "auroc"),
    "cyp2c9_veith": ("classification", "auprc"),
    "cyp2c9_substrate_carbonmangels": ("classification", "auprc"),
    "caco2_wang": ("regression", "mae"),
    "cyp2d6_veith": ("classification", "auprc"),
    "cyp2d6_substrate_carbonmangels": ("classification", "auprc"),
    "cyp3a4_veith": ("classification", "auprc"),
    "half_life_obach": ("regression", "spearman"),
    "clearance_hepatocyte_az": ("regression", "spearman"),
    "clearance_microsome_az": ("regression", "spearman"),
    "vdss_lombardo": ("regression", "spearman"),
    "ld50_zhu": ("regression", "mae"),
    "lipophilicity_astrazeneca": ("regression", "mae"),
    "ppbr_az": ("regression", "mae"),
    "solubility_aqsoldb": ("regression", "mae"),
}


class MoleProtocolModel(nn.Module):
    """Final-layer CLS plus MolE's dropout-linear-GELU-dropout head."""

    def __init__(self, encoder_state: OrderedDict, dropout: float):
        super().__init__()
        cfg = ModelConfig.from_dict(resolve_encoder_config(encoder_state, "auto"))
        self.encoder = AtomEnvEmbeddings(cfg)
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"strict encoder load failed: missing={missing} unexpected={unexpected}")
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.classifier = nn.Linear(cfg.hidden_size, 1)

    def forward(self, batch):
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        output = self.encoder(input_ids, input_mask, attention_mask=input_mask, relative_pos=relative_pos)
        cls = output["hidden_states"][-1][:, 0]
        hidden = self.dropout(cls)
        hidden = F.gelu(self.dense(hidden))
        return self.classifier(self.dropout(hidden)).squeeze(-1)


def atomic_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def load_valid_json(path: Path):
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def resolve_vocab_path():
    import mole
    return str(
        Path(mole.__path__[0])
        / "training/data/vocabularies/vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
    )


def frame_xy(frame: pd.DataFrame):
    if not {"Drug", "Y"}.issubset(frame.columns):
        raise ValueError(f"unexpected TDC columns: {list(frame.columns)}")
    if frame.Y.isna().any():
        frame = frame.dropna(subset=["Y"]).reset_index(drop=True)
    return frame.Drug.reset_index(drop=True), frame.Y.to_numpy(dtype=np.float32)


def make_loader(frame, dictionary, shuffle):
    smiles, labels = frame_xy(frame)
    dataset = MolDataset(
        smiles=smiles,
        dictionary_inp=dictionary,
        radius_inp=0,
        useFeatures_inp=False,
        cls_token=True,
        labels=labels,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=shuffle,
    )


def metric_utility(task_type, metric, labels, logits):
    labels = np.asarray(labels).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    if task_type == "classification":
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        score = average_precision_score(labels, probs) if metric == "auprc" else roc_auc_score(labels, probs)
        return float(score), float(score)
    if metric == "spearman":
        score = float(spearmanr(labels, logits).statistic)
        return score, score
    score = float(mean_absolute_error(labels, logits))
    return score, -score


@torch.no_grad()
def evaluate(model, loader, device, task_type, metric):
    model.eval()
    labels, logits = [], []
    losses = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.target_labels.float().reshape_as(pred)
        loss = F.binary_cross_entropy_with_logits(pred, target) if task_type == "classification" else F.mse_loss(pred, target)
        losses.append(float(loss.item()))
        labels.append(target.cpu().numpy())
        logits.append(pred.float().cpu().numpy())
    score, score_utility = metric_utility(task_type, metric, np.concatenate(labels), np.concatenate(logits))
    return {"loss": float(np.mean(losses)), metric: score, "utility": score_utility}


def train_run(encoder_state, dictionary, train, valid, test, task_type, metric, lr, dropout, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoleProtocolModel(encoder_state, dropout).to(device)
    train_loader = make_loader(train, dictionary, True)
    valid_loader = make_loader(valid, dictionary, False)
    test_loader = make_loader(test, dictionary, False) if test is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    total_steps = EPOCHS * max(1, len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best = None
    best_state = None
    step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.target_labels.float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(batch)
                target = target.reshape_as(pred)
                loss = (F.binary_cross_entropy_with_logits(pred, target)
                        if task_type == "classification" else F.mse_loss(pred, target))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = min(1.0, (step + 1) / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr * scale
            scaler.step(optimizer)
            scaler.update()
            step += 1

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            val = evaluate(model, valid_loader, device, task_type, metric)
            print(
                f"task={TASK_NAME} seed={seed} lr={lr:g} dropout={dropout:g} "
                f"epoch={epoch}/{EPOCHS} val_{metric}={val[metric]:.6f} val_loss={val['loss']:.6f}",
                flush=True,
            )
            if best is None or val["utility"] > best["utility"]:
                best = {**val, "epoch": epoch}
                if test_loader is not None:
                    best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}

    result = {"best_validation": best}
    if test_loader is not None:
        model.load_state_dict(best_state)
        result["test"] = evaluate(model, test_loader, device, task_type, metric)
    return result


def canonical_hashes(frame):
    canonical = []
    invalid = 0
    over_96 = 0
    for smiles in frame.Drug:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        over_96 += int(mol.GetNumHeavyAtoms() > 96)
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    digest = hashlib.sha256("\n".join(sorted(canonical)).encode()).hexdigest()
    return {"rows": len(frame), "valid": len(canonical), "invalid": invalid, "over_96_heavy_atoms": over_96,
            "canonical_smiles_sha256": digest}


def main():
    if TASK_NAME not in TASK_CONFIG:
        raise ValueError(f"unsupported pilot task {TASK_NAME}; choose from {sorted(TASK_CONFIG)}")
    if not Path(RTD_CKPT).is_file():
        raise FileNotFoundError(RTD_CKPT)
    task_type, metric = TASK_CONFIG[TASK_NAME]
    output = OUTPUT_DIR / TASK_NAME
    output.mkdir(parents=True, exist_ok=True)

    group = tdc.BenchmarkGroup(name="ADMET_Group", path=TDC_DATA_DIR)
    benchmark = group.get(TASK_NAME)
    test = benchmark["test"].reset_index(drop=True)
    dictionary = open_dictionary(resolve_vocab_path())
    encoder_state = load_encoder_state(RTD_CKPT)
    if encoder_state is None:
        raise RuntimeError(f"could not load encoder from {RTD_CKPT}")

    folds = {}
    split_manifest = {"task": TASK_NAME, "tdc_group": "ADMET_Group", "test": canonical_hashes(test), "folds": {}}
    for fold in FOLDS:
        train, valid = group.get_train_valid_split(
            benchmark=TASK_NAME, split_type="default", seed=fold
        )
        train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        folds[fold] = (train, valid)
        split_manifest["folds"][str(fold)] = {
            "train": canonical_hashes(train), "valid": canonical_hashes(valid)
        }
    atomic_json(output / "split_manifest.json", split_manifest)

    grid = []
    for lr in LEARNING_RATES:
        for dropout in DROPOUTS:
            fold_scores = []
            for fold in FOLDS:
                path = output / "hpo" / f"lr{lr:g}_dropout{dropout:g}_fold{fold}.json"
                record = load_valid_json(path)
                if record is None:
                    train, valid = folds[fold]
                    run = train_run(
                        encoder_state, dictionary, train, valid, None,
                        task_type, metric, lr, dropout, fold,
                    )
                    record = {
                        "task": TASK_NAME, "fold": fold, "lr": lr, "dropout": dropout,
                        "task_type": task_type, "primary_metric": metric, **run,
                    }
                    atomic_json(path, record)
                fold_scores.append(record["best_validation"]["utility"])
            grid.append({
                "lr": lr, "dropout": dropout,
                "mean_validation_utility": float(np.mean(fold_scores)),
                "std_validation_utility": float(np.std(fold_scores)),
                "fold_utilities": fold_scores,
            })

    selected = max(grid, key=lambda item: item["mean_validation_utility"])
    selection = {
        "task": TASK_NAME, "selection_rule": "maximum mean validation primary-metric utility over five folds",
        "selected_lr": selected["lr"], "selected_dropout": selected["dropout"],
        "grid": grid,
    }
    atomic_json(output / "selected_config.json", selection)

    test_runs = []
    # Independent locked-test seeds use the corresponding official train/valid split seed modulo five.
    for index, seed in enumerate(TEST_SEEDS):
        fold = FOLDS[index % len(FOLDS)]
        path = output / "test" / f"seed{seed}.json"
        record = load_valid_json(path)
        if record is None:
            train, valid = folds[fold]
            run = train_run(
                encoder_state, dictionary, train, valid, test,
                task_type, metric, selected["lr"], selected["dropout"], seed,
            )
            record = {
                "task": TASK_NAME, "seed": seed, "split_fold": fold,
                "lr": selected["lr"], "dropout": selected["dropout"],
                "task_type": task_type, "primary_metric": metric, **run,
            }
            atomic_json(path, record)
        test_runs.append(record)

    scores = [record["test"][metric] for record in test_runs]
    summary = {
        "task": TASK_NAME, "task_type": task_type, "primary_metric": metric,
        "protocol": "fixed TDC ADMET_Group test; validation-only 5-fold HPO; CLS/GELU head; Adam; no WD",
        "checkpoint": RTD_CKPT,
        "selected_lr": selected["lr"], "selected_dropout": selected["dropout"],
        "test_mean": float(np.mean(scores)), "test_std": float(np.std(scores)),
        "test_scores": scores, "test_seeds": list(TEST_SEEDS),
    }
    atomic_json(OUTPUT_DIR / f"results_{TASK_NAME}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
