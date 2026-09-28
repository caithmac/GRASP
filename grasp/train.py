"""Fine-tune a GRASP encoder on a user's property CSV files."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from rdkit import Chem

from mole.training.data.datasets import MolDataset

from .encoder import GRASPEncoder, validate_smiles
from .predictor import GRASPPredictor


def read_labeled_csv(path: str | Path, smiles_column: str, target_column: str,
                     task: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    missing = [name for name in (smiles_column, target_column) if name not in frame]
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")
    smiles = validate_smiles(frame[smiles_column].tolist())
    targets = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all():
        raise ValueError(f"{path} contains a missing or non-finite target")
    if task == "binary" and not np.isin(targets, [0.0, 1.0]).all():
        raise ValueError("Binary targets must be 0 or 1")
    return smiles, targets


def fit(*, model_source: str | Path, train_csv: str | Path, valid_csv: str | Path,
        output_dir: str | Path, task: str, method: str = "full",
        smiles_column: str = "smiles", target_column: str = "target",
        epochs: int = 60, patience: int = 12, batch_size: int = 64,
        learning_rate: float | None = None, seed: int = 0,
        device: str | None = None) -> dict:
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if method not in {"full", "lora", "frozen"}:
        raise ValueError("method must be full, lora, or frozen")
    if task not in {"regression", "binary"}:
        raise ValueError("task must be regression or binary")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_smiles, train_y = read_labeled_csv(train_csv, smiles_column, target_column, task)
    valid_smiles, valid_y = read_labeled_csv(valid_csv, smiles_column, target_column, task)
    overlap = ({Chem.MolToSmiles(Chem.MolFromSmiles(value)) for value in train_smiles}
               & {Chem.MolToSmiles(Chem.MolFromSmiles(value)) for value in valid_smiles})
    if overlap:
        raise ValueError(f"Training and validation CSVs overlap on {len(overlap)} molecular structures")
    if task == "binary" and len(np.unique(train_y)) != 2:
        raise ValueError("Binary training data must contain both classes")
    encoder = GRASPEncoder.from_pretrained(model_source, device=device)
    predictor = GRASPPredictor.from_encoder(encoder, task=task, method=method)
    use_bf16 = predictor.device.type == "cuda" and torch.cuda.is_bf16_supported()
    if task == "regression":
        mean = float(np.mean(train_y))
        std = float(np.std(train_y)) or 1.0
        predictor.config.update(target_mean=mean, target_std=std)
        fitted_y = (train_y - mean) / std
    else:
        fitted_y = train_y

    def loader(smiles, targets, shuffle):
        dataset = MolDataset(pd.Series(smiles), predictor.vocabulary, radius_inp=0,
                             useFeatures_inp=False, cls_token=True,
                             labels=np.asarray(targets, dtype=np.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = loader(train_smiles, fitted_y, True)
    valid_loader = loader(valid_smiles, valid_y, False)
    rate = learning_rate or {"full": 1e-5, "lora": 5e-5, "frozen": 1e-3}[method]
    if rate <= 0:
        raise ValueError("learning_rate must be positive")
    trainable = [param for param in predictor.model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=rate, weight_decay=1e-4)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    step = 0
    for epoch in range(1, epochs + 1):
        predictor.model.train()
        for batch in train_loader:
            batch = batch.to(predictor.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                output = predictor.model(batch)
                target = batch.target_labels.float().reshape_as(output)
                loss = (nn.functional.binary_cross_entropy_with_logits(output, target)
                        if task == "binary" else nn.functional.smooth_l1_loss(output, target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = rate * min(1.0, step / warmup_steps)
            optimizer.step()
        predictor.model.eval()
        validation_outputs = []
        with torch.no_grad():
            for batch in valid_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    validation_outputs.append(predictor.model(batch.to(predictor.device)).float().cpu())
        raw = torch.cat(validation_outputs).numpy()
        if task == "regression":
            score = float(np.mean(np.abs(raw * predictor.config["target_std"]
                                         + predictor.config["target_mean"] - valid_y)))
        else:
            score = float(np.mean(np.logaddexp(0.0, raw) - valid_y * raw))
        print(f"epoch={epoch} validation_{'mae' if task == 'regression' else 'bce'}={score:.6f}", flush=True)
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in predictor.model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    predictor.model.load_state_dict(best_state, strict=True)
    predictor.model.eval()
    predictor.config.update(best_epoch=best_epoch, validation_score=best_score,
                            validation_metric="mae" if task == "regression" else "bce",
                            precision="bf16" if use_bf16 else "fp32",
                            smiles_column=smiles_column, target_column=target_column,
                            seed=seed)
    predictor.save_pretrained(output_dir)
    return {"best_epoch": best_epoch, "validation_score": best_score,
            "validation_metric": predictor.config["validation_metric"],
            "output_dir": str(output_dir)}
