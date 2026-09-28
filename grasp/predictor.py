"""Paper-aligned downstream readout with full, LoRA, and frozen adaptation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from DeBERTa.deberta.config import ModelConfig
from mole.training.models.mole import AtomEnvEmbeddings

from .encoder import GRASPEncoder, encoder_inputs, make_loader, read_vocab, resolve_files


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, inputs):
        update = (self.dropout(inputs) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(inputs) + self.scale * update


class PropertyNetwork(nn.Module):
    def __init__(self, architecture: dict, method: str, *, attach_lora: bool = True):
        super().__init__()
        if method not in {"full", "lora", "frozen"}:
            raise ValueError(f"Unknown adaptation method: {method}")
        self.method = method
        cfg = ModelConfig.from_dict(architecture)
        self.encoder = AtomEnvEmbeddings(cfg)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if method == "full":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        elif method == "lora" and attach_lora:
            self.add_lora()
        self.layer_logits = nn.Parameter(torch.zeros(4))
        self.pooler = nn.Linear(cfg.hidden_size, 1)
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_size),
                                  nn.Linear(cfg.hidden_size, 512), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(512, 1))

    def add_lora(self) -> None:
        for layer in self.encoder.encoder.layer:
            attention = layer.attention.self
            attention.query_proj = LoRALinear(attention.query_proj)
            attention.value_proj = LoRALinear(attention.value_proj)

    def forward(self, batch):
        input_ids, input_mask, relative_pos = encoder_inputs(batch)
        if self.method == "frozen":
            self.encoder.eval()
            with torch.no_grad():
                output = self.encoder(input_ids, input_mask,
                                      attention_mask=input_mask, relative_pos=relative_pos)
        else:
            output = self.encoder(input_ids, input_mask,
                                  attention_mask=input_mask, relative_pos=relative_pos)
        weights = torch.softmax(self.layer_logits, dim=0)
        hidden = sum(weights[i] * output["hidden_states"][layer - 1]
                     for i, layer in enumerate((4, 8, 10, 12)))
        atom_mask = input_mask.bool().clone()
        atom_mask[:, 0] = False
        logits = self.pooler(hidden).squeeze(-1).masked_fill(~atom_mask, float("-inf"))
        pooled = (torch.softmax(logits, dim=1).unsqueeze(-1) * hidden).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class GRASPPredictor:
    def __init__(self, model: PropertyNetwork, vocabulary: dict[int | str, int],
                 config: dict, device: str | torch.device):
        self.model = model.to(device)
        self.vocabulary = vocabulary
        self.config = config
        self.device = torch.device(device)

    @classmethod
    def from_encoder(cls, encoder: GRASPEncoder, *, task: str, method: str = "full") -> "GRASPPredictor":
        if task not in {"regression", "binary"}:
            raise ValueError("task must be regression or binary")
        model = PropertyNetwork(encoder.config["architecture"], method, attach_lora=False)
        model.encoder.load_state_dict(encoder.model.state_dict(), strict=True)
        if method == "lora":
            model.add_lora()
        config = {"architecture": encoder.config["architecture"],
                  "task": task, "method": method,
                  "source_checkpoint_sha256": encoder.config["source_checkpoint_sha256"],
                  "target_mean": 0.0, "target_std": 1.0,
                  "output": "probability" if task == "binary" else "original_target_units"}
        return cls(model, encoder.vocabulary, config, encoder.device)

    @classmethod
    def from_pretrained(cls, directory: str | Path, *,
                        device: str | torch.device | None = None) -> "GRASPPredictor":
        files = resolve_files(directory, ("predictor.safetensors", "predictor_config.json", "vocab.json"))
        config = json.loads(files["predictor_config.json"].read_text(encoding="utf-8"))
        model = PropertyNetwork(config["architecture"], config["method"])
        model.load_state_dict(load_file(files["predictor.safetensors"], device="cpu"), strict=True)
        model.eval()
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, read_vocab(files["vocab.json"]), config, chosen_device)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().contiguous() for key, value in self.model.state_dict().items()}
        save_file(state, directory / "predictor.safetensors")
        (directory / "predictor_config.json").write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")
        raw_vocab = {str(key): value for key, value in self.vocabulary.items()}
        (directory / "vocab.json").write_text(json.dumps(raw_vocab, sort_keys=True) + "\n", encoding="utf-8")

    @torch.no_grad()
    def predict(self, smiles: str | Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        values = [smiles] if isinstance(smiles, str) else list(smiles)
        self.model.eval()
        outputs = []
        for batch in make_loader(values, self.vocabulary, batch_size):
            outputs.append(self.model(batch.to(self.device)).float().cpu())
        values = torch.cat(outputs)
        if self.config["task"] == "binary":
            return torch.sigmoid(values).numpy()
        return values.numpy() * self.config["target_std"] + self.config["target_mean"]
