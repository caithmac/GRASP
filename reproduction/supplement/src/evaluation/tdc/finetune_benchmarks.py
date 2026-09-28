"""Generalized fine-tuning script for MolE-RTD on various TDC benchmarks.

Supported tasks (examples):
  - Classification: BBBP, BACE, ClinTox, HIA_Hou, SIDER, HIV
  - Regression: Lipophilicity_AstraZeneca, Solubility_AqSolDB, FreeSolv

Env vars:
  TASK_NAME         TDC task name (e.g., "BBBP", "BACE", "Lipophilicity_AstraZeneca")
  TASK_TYPE         "classification" or "regression" (default: auto-detect)
  RTD_CKPT          Path to pretrained checkpoint
  RESULTS_PATH      Path to save JSON results
  TDC_DATA_DIR      Path to TDC data cache
  N_SEEDS           Number of seeds to run (default: 3)
  EPOCHS            Number of epochs (default: 40)
  BATCH_SIZE        Batch size (default: 32)
  LR                Learning rate (default: 1e-4)
"""
from __future__ import annotations

import json
import math
import os
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from DeBERTa.deberta.config import ModelConfig
from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary
from mole.training.models.mole import AtomEnvEmbeddings
from encoder_arch import (
    architecture_summary,
    load_encoder_state,
    resolve_encoder_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENCODER_ARCH = os.environ.get("ENCODER_ARCH", "auto").lower()
STRICT_ENCODER_LOAD = os.environ.get("STRICT_ENCODER_LOAD", "0") == "1"
DISC_CFG = resolve_encoder_config(requested=ENCODER_ARCH)

# Pooling over encoder outputs. RTD never trains the CLS token, so "cls" pools
# from an untrained position. "attn" (default) and "mean" pool over real atom
# tokens instead. A/B via POOL_MODE env var.
POOL_MODE = os.environ.get("POOL_MODE", "attn").lower()

# Configuration from Environment
RTD_CKPT = os.environ.get("RTD_CKPT", "/mnt/checkpoints/last.ckpt")
TASK_NAME = os.environ.get("TASK_NAME", "bbb_martins")
TASK_TYPE = os.environ.get("TASK_TYPE", None)  # classification or regression
RESULTS_PATH = os.environ.get("RESULTS_PATH", f"/mnt/results_{TASK_NAME.lower()}.json")
TDC_DATA_DIR = os.environ.get("TDC_DATA_DIR", "/mnt/tdc_data")
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
SKIP_RANDOM = os.environ.get("SKIP_RANDOM", "0") == "1"

# Per-task metric config.
# primary_metric: auroc | auprc | mae | spearman
# normalize_targets: z-score targets on train, apply to val/test (fixes scale issues like PPBR)
TASK_CONFIG = {
    # Regression — Spearman primary (matches MolE Table 1)
    "vdss_lombardo":                  {"primary_metric": "spearman", "normalize_targets": True},
    "clearance_microsome_az":         {"primary_metric": "spearman", "normalize_targets": True},
    "clearance_hepatocyte_az":        {"primary_metric": "spearman", "normalize_targets": True},
    "half_life_obach":                {"primary_metric": "spearman", "normalize_targets": True},
    # Classification — AUPRC primary (matches MolE Table 1, imbalanced CYP inhibition)
    "cyp2d6_veith":                   {"primary_metric": "auprc"},
    "cyp3a4_veith":                   {"primary_metric": "auprc"},
    "cyp2c9_veith":                   {"primary_metric": "auprc"},
    "cyp2d6_substrate_carbonmangels": {"primary_metric": "auprc"},
    "cyp2c9_substrate_carbonmangels": {"primary_metric": "auprc"},
    # Regression — MAE primary, normalize to fix gradient scale
    "ppbr_az":                        {"primary_metric": "mae", "normalize_targets": True},
    "solubility_aqsoldb":             {"primary_metric": "mae", "normalize_targets": True},
    "lipophilicity_astrazeneca":      {"primary_metric": "mae", "normalize_targets": True},
    "caco2_wang":                     {"primary_metric": "mae", "normalize_targets": True},
    "ld50_zhu":                       {"primary_metric": "mae", "normalize_targets": True},
}
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mole-rtd-benchmarks")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
USE_WANDB = os.environ.get("WANDB_API_KEY", None) is not None

def resolve_vocab_path() -> str:
    p = os.environ.get("VOCAB_PATH")
    if p and Path(p).exists():
        return p
    import mole
    candidate = Path(mole.__path__[0]) / "training" / "data" / "vocabularies" / "vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
    return str(candidate)

def load_rtd_as_encoder_state(ckpt_path: str) -> OrderedDict | None:
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint {ckpt_path} not found! Training from scratch.")
        return None
    return load_encoder_state(ckpt_path)

class ClsHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense(cls_hidden))
        x = self.dropout(x)
        return self.classifier(x)

class AttnPool(nn.Module):
    """Additive attention pooling over atom tokens (single learned query)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden [B,T,H], mask [B,T] bool (True = real token)
        w = self.score(hidden).squeeze(-1)                 # [B,T]
        w = w.masked_fill(~mask, float("-inf"))
        w = torch.softmax(w, dim=1).unsqueeze(-1)          # [B,T,1]
        return (w * hidden).sum(dim=1)                     # [B,H]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


class MolEModel(nn.Module):
    def __init__(self, encoder_state: OrderedDict | None, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        encoder_config = (
            resolve_encoder_config(encoder_state, ENCODER_ARCH)
            if encoder_state is not None
            else DISC_CFG
        )
        cfg = ModelConfig.from_dict(encoder_config)
        self.encoder = AtomEnvEmbeddings(cfg)
        if encoder_state is not None:
            miss, unexp = self.encoder.load_state_dict(encoder_state, strict=False)
            logger.info(f"  encoder load: missing={len(miss)} unexpected={len(unexp)}")
            if miss:
                logger.error(f"  missing encoder keys: {list(miss)}")
            if unexp:
                logger.error(f"  unexpected encoder keys: {list(unexp)}")
            if STRICT_ENCODER_LOAD and (miss or unexp):
                raise RuntimeError(
                    f"Strict encoder load failed: missing={len(miss)} unexpected={len(unexp)}"
                )
        self.pool_mode = POOL_MODE
        self.pooler = AttnPool(cfg.hidden_size) if self.pool_mode == "attn" else None
        self.head = ClsHead(cfg.hidden_size, num_classes=num_classes, dropout=dropout)

    def forward(self, batch) -> torch.Tensor:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        out = self.encoder(
            input_ids,
            input_mask,
            attention_mask=input_mask,
            relative_pos=relative_pos,
        )
        hidden = out["hidden_states"][-1]                  # [B,T,H]
        if self.pool_mode == "cls":
            pooled = hidden[:, 0, :]
        else:
            # atom tokens only: drop CLS at position 0 (untrained under RTD)
            atom_mask = input_mask.clone()
            atom_mask[:, 0] = False
            pooled = self.pooler(hidden, atom_mask) if self.pool_mode == "attn" else masked_mean(hidden, atom_mask)
        return self.head(pooled).squeeze(-1)

def linear_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

def make_loader(smiles, labels, dictionary, batch_size: int, shuffle: bool) -> DataLoader:
    ds = MolDataset(
        smiles=smiles,
        dictionary_inp=dictionary,
        radius_inp=0,
        useFeatures_inp=False,
        cls_token=True,
        labels=labels,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)

def evaluate(
    model: MolEModel,
    loader: DataLoader,
    device: str,
    task_type: str,
    primary_metric: str = "auroc",
    scaler: StandardScaler | None = None,
) -> float:
    """Return primary metric score (higher always better — negate MAE internally)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if task_type == "classification":
                ps.append(torch.sigmoid(logits).float().cpu().numpy())
            else:
                ps.append(logits.float().cpu().numpy())
            ys.append(batch.target_labels.float().cpu().numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    # Denormalize regression predictions back to original units for metrics
    if scaler is not None:
        ps = scaler.inverse_transform(ps.reshape(-1, 1)).ravel()
        ys = scaler.inverse_transform(ys.reshape(-1, 1)).ravel()

    if task_type == "classification":
        ys_1d = ys.ravel()
        ps_1d = ps.ravel()
        if primary_metric == "auprc":
            return float(average_precision_score(ys_1d, ps_1d))
        # auroc — handle multi-task
        if len(ys.shape) > 1 and ys.shape[1] > 1:
            aucs = []
            for i in range(ys.shape[1]):
                try:
                    aucs.append(roc_auc_score(ys[:, i], ps[:, i]))
                except ValueError:
                    continue
            return float(np.mean(aucs))
        return float(roc_auc_score(ys_1d, ps_1d))
    else:
        if primary_metric == "spearman":
            return float(spearmanr(ys.ravel(), ps.ravel()).statistic)
        # mae — return negative so "higher is better" for best_val tracking
        return -float(mean_absolute_error(ys.ravel(), ps.ravel()))

def train_one(
    encoder_state: OrderedDict | None,
    splits: Dict[str, Tuple[list, list]],
    dictionary,
    device: str,
    seed: int,
    label: str,
    task_type: str,
    num_classes: int,
    primary_metric: str = "auroc",
    scaler: StandardScaler | None = None,
    run=None,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MolEModel(encoder_state, num_classes=num_classes).to(device)
    train_loader = make_loader(splits["train"][0], splits["train"][1], dictionary, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(splits["valid"][0], splits["valid"][1], dictionary, BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(splits["test"][0], splits["test"][1], dictionary, BATCH_SIZE, shuffle=False)

    total_steps = EPOCHS * max(1, len(train_loader))
    warmup_steps = int(WARMUP_FRAC * total_steps)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_val = -float("inf")
    best_epoch = -1
    best_test_metrics: Dict[str, float] = {}
    step = 0

    criterion = F.binary_cross_entropy_with_logits if task_type == "classification" else F.l1_loss

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)

            y = batch.target_labels.float()
            if task_type == "classification":
                mask = ~torch.isnan(y)
                loss = criterion(logits[mask], y[mask])
            else:
                loss = criterion(logits, y)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in optim.param_groups:
                g["lr"] = LR * linear_warmup(step, total_steps, warmup_steps)
            optim.step()
            step += 1
            epoch_loss += loss.item()
            n_batches += 1

        val_score = evaluate(model, val_loader, device, task_type, primary_metric, scaler)
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_test_metrics = _collect_all_metrics(model, test_loader, device, task_type, scaler)

        logger.info(
            f"    [{label} seed={seed}] epoch {epoch+1:02d}/{EPOCHS} "
            f"loss={epoch_loss/max(1,n_batches):.4f} val_{primary_metric}={val_score:.4f} "
            f"best={best_val:.4f}"
        )
        if run is not None:
            run.log({
                f"{label}/seed{seed}/train_loss": epoch_loss / max(1, n_batches),
                f"{label}/seed{seed}/val_{primary_metric}": val_score,
                "epoch": epoch + 1,
            })

    result = {"seed": seed, "best_epoch": best_epoch, f"best_val_{primary_metric}": best_val}
    result.update({f"{k}_at_best_val": v for k, v in best_test_metrics.items()})
    return result


def _collect_all_metrics(
    model: MolEModel,
    loader: DataLoader,
    device: str,
    task_type: str,
    scaler: StandardScaler | None,
) -> Dict[str, float]:
    """Compute all relevant metrics in original (denormalized) units."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            ps.append(logits.float().cpu().numpy())
            ys.append(batch.target_labels.float().cpu().numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    if scaler is not None:
        ps = scaler.inverse_transform(ps.reshape(-1, 1)).ravel()
        ys = scaler.inverse_transform(ys.reshape(-1, 1)).ravel()

    if task_type == "classification":
        ys_1d, ps_1d = ys.ravel(), ps.ravel()
        # sigmoid already applied in evaluate but NOT here (raw logits from model)
        ps_prob = 1 / (1 + np.exp(-ps_1d))
        return {
            "auroc": float(roc_auc_score(ys_1d, ps_prob)),
            "auprc": float(average_precision_score(ys_1d, ps_prob)),
        }
    else:
        ys_1d, ps_1d = ys.ravel(), ps.ravel()
        return {
            "mae":      float(mean_absolute_error(ys_1d, ps_1d)),
            "rmse":     float(np.sqrt(mean_squared_error(ys_1d, ps_1d))),
            "r2":       float(r2_score(ys_1d, ps_1d)),
            "spearman": float(spearmanr(ys_1d, ps_1d).statistic),
        }

def is_valid_smiles(s: str, max_heavy: int = 96) -> bool:
    if not isinstance(s, str): return False
    m = Chem.MolFromSmiles(s)
    return m is not None and 0 < m.GetNumHeavyAtoms() <= max_heavy

def get_tdc_data(task_name: str) -> Tuple[Dict[str, Tuple[list, list]], str, int]:
    from tdc.single_pred import ADME, Tox

    # Try ADME first, then Tox — avoids brittle retrieve_dataset_names API
    data = None
    for cls in (ADME, Tox):
        try:
            data = cls(name=task_name, path=TDC_DATA_DIR)
            break
        except Exception:
            continue
    if data is None:
        raise ValueError(f"Task '{task_name}' not found in TDC ADME or Tox.")

    logger.info(f"Loading TDC task: {task_name}")
    
    # Auto-detect task type if not provided
    global TASK_TYPE
    if TASK_TYPE is None:
        y_vals = data.get_data()["Y"].dropna().unique()
        if set(y_vals).issubset({0, 1, 0.0, 1.0}):
            TASK_TYPE = "classification"
        else:
            TASK_TYPE = "regression"
        logger.info(f"Auto-detected TASK_TYPE: {TASK_TYPE}")

    split = data.get_split(method="scaffold", seed=42, frac=[0.7, 0.15, 0.15])
    
    out = {}
    num_classes = 1 # default
    
    for k in ("train", "valid", "test"):
        df = split[k]
        df = df[df["Drug"].apply(is_valid_smiles)].reset_index(drop=True)
        # Handle multi-column labels if they exist
        labels = df["Y"].values
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        num_classes = labels.shape[1]
        
        out[k] = (df["Drug"].reset_index(drop=True), labels)
        logger.info(f"  {k}: n={len(df)}, num_labels={num_classes}")
        
    return out, TASK_TYPE, num_classes

def summarize(runs: List[Dict[str, float]], task_type: str, primary_metric: str) -> Dict[str, float]:
    out: Dict[str, Any] = {"n_seeds": len(runs)}

    def _agg(key: str) -> Tuple[float, float]:
        vals = np.array([r[key] for r in runs if key in r])
        return float(vals.mean()), float(vals.std())

    if task_type == "classification":
        for m in ("auroc", "auprc"):
            key = f"{m}_at_best_val"
            if any(key in r for r in runs):
                mean, std = _agg(key)
                out[f"test_{m}_mean"] = mean
                out[f"test_{m}_std"] = std
        # val primary metric
        val_key = f"best_val_{primary_metric}"
        if any(val_key in r for r in runs):
            vm, vs = _agg(val_key)
            out[f"val_{primary_metric}_mean"] = vm
            out[f"val_{primary_metric}_std"] = vs
    else:
        for m in ("mae", "rmse", "r2", "spearman"):
            key = f"{m}_at_best_val"
            if any(key in r for r in runs):
                mean, std = _agg(key)
                out[f"test_{m}_mean"] = mean
                out[f"test_{m}_std"] = std
        val_key = f"best_val_{primary_metric}"
        if any(val_key in r for r in runs):
            vm, vs = _agg(val_key)
            out[f"val_{primary_metric}_mean"] = vm
            out[f"val_{primary_metric}_std"] = vs

    out["per_seed"] = runs
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Task: {TASK_NAME}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {RTD_CKPT}")

    run = None
    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=TASK_NAME,
            config=dict(task=TASK_NAME, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        lr=LR, n_seeds=N_SEEDS, ckpt=RTD_CKPT),
            reinit=True,
        )

    dictionary = open_dictionary(resolve_vocab_path())
    splits, task_type, num_classes = get_tdc_data(TASK_NAME)

    # Task-specific config (metric + normalization)
    cfg = TASK_CONFIG.get(TASK_NAME.lower(), {})
    if "primary_metric" not in cfg:
        cfg["primary_metric"] = "auroc" if task_type == "classification" else "mae"
    primary_metric = cfg["primary_metric"]
    normalize_targets = cfg.get("normalize_targets", False)

    # Fit target scaler on train labels (regression only)
    scaler: StandardScaler | None = None
    if normalize_targets and task_type == "regression":
        train_smiles, train_labels = splits["train"]
        scaler = StandardScaler()
        train_labels_norm = scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()
        splits["train"] = (train_smiles, train_labels_norm)
        for split_key in ("valid", "test"):
            sm, lb = splits[split_key]
            splits[split_key] = (sm, scaler.transform(lb.reshape(-1, 1)).ravel())
        logger.info(f"  Target normalization: mean={scaler.mean_[0]:.3f} std={scaler.scale_[0]:.3f}")

    logger.info(f"  Primary metric: {primary_metric}")

    pretrained_state = load_rtd_as_encoder_state(RTD_CKPT)
    if pretrained_state is not None:
        DISC_CFG.clear()
        DISC_CFG.update(resolve_encoder_config(pretrained_state, ENCODER_ARCH))
    logger.info(f"  Encoder architecture: {architecture_summary(DISC_CFG)}")

    logger.info(f"\nFine-tuning condition A: pretrained encoder")
    runs_pre = []
    for seed in range(N_SEEDS):
        runs_pre.append(train_one(
            pretrained_state, splits, dictionary, device, seed, "pretrained",
            task_type, num_classes, primary_metric=primary_metric, scaler=scaler, run=run,
        ))
    summary_pre = summarize(runs_pre, task_type, primary_metric)

    if SKIP_RANDOM:
        logger.info("SKIP_RANDOM=1 — skipping random-init condition")
        summary_rnd = None
    else:
        logger.info(f"\nFine-tuning condition B: random-init encoder")
        runs_rnd = []
        for seed in range(N_SEEDS):
            runs_rnd.append(train_one(
                None, splits, dictionary, device, seed, "random    ",
                task_type, num_classes, primary_metric=primary_metric, scaler=scaler, run=run,
            ))
        summary_rnd = summarize(runs_rnd, task_type, primary_metric)

    logger.info("\n" + "=" * 70)
    logger.info(f"Task: {TASK_NAME} ({task_type}, primary={primary_metric})")
    pre_val = summary_pre.get(f"test_{primary_metric}_mean", "N/A")
    pre_std = summary_pre.get(f"test_{primary_metric}_std", 0)
    logger.info(f"  pretrained test {primary_metric}: {pre_val:.4f} +/- {pre_std:.4f}")
    if summary_rnd:
        rnd_val = summary_rnd.get(f"test_{primary_metric}_mean", float("nan"))
        rnd_std = summary_rnd.get(f"test_{primary_metric}_std", 0)
        logger.info(f"  random     test {primary_metric}: {rnd_val:.4f} +/- {rnd_std:.4f}")
        if isinstance(pre_val, float):
            logger.info(f"  Delta: {pre_val - rnd_val:+.4f}")

    results = {
        "task": TASK_NAME,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "normalize_targets": normalize_targets,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "ckpt": RTD_CKPT,
        "pretrained": summary_pre,
    }
    if summary_rnd is not None:
        results["random_init"] = summary_rnd

    os.makedirs(os.path.dirname(os.path.abspath(RESULTS_PATH)), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_PATH}")

    if run is not None:
        pre_key = f"test_{primary_metric}_mean"
        d = {f"pretrained/{primary_metric}": summary_pre.get(pre_key, float("nan"))}
        if summary_rnd:
            rnd_key = f"test_{primary_metric}_mean"
            d[f"random/{primary_metric}"] = summary_rnd.get(rnd_key, float("nan"))
            d[f"delta_{primary_metric}"] = (
                summary_pre.get(pre_key, 0) - summary_rnd.get(rnd_key, 0)
            )
        run.summary.update(d)
        run.finish()

if __name__ == "__main__":
    main()
