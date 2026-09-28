"""Checkpoint loading and molecular representations for GRASP."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from safetensors.torch import load_file
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from DeBERTa.deberta.config import ModelConfig
from mole.training.data.datasets import MolDataset
from mole.training.models.mole import AtomEnvEmbeddings


def resolve_files(source: str | Path, filenames: tuple[str, ...], revision: str | None = None) -> dict[str, Path]:
    local = Path(source)
    if local.is_dir():
        files = {name: local / name for name in filenames}
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model files in {local}: {missing}")
        return files
    from huggingface_hub import hf_hub_download

    return {name: Path(hf_hub_download(repo_id=str(source), filename=name, revision=revision))
            for name in filenames}


def read_vocab(path: Path) -> dict[int | str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab = {int(k) if k not in {"PAD", "MASK", "UNK", "CLS"} else k: int(v)
             for k, v in raw.items()}
    if len(vocab) != 211 or [vocab[k] for k in ("PAD", "MASK", "UNK", "CLS")] != [0, 208, 209, 210]:
        raise ValueError("GRASP vocabulary is missing or incompatible")
    return vocab


def validate_smiles(smiles: Iterable[str], max_atoms: int = 511) -> list[str]:
    values = list(smiles)
    if not values:
        raise ValueError("At least one SMILES is required")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SMILES row {index} is empty")
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"SMILES row {index} is invalid: {value!r}")
        if mol.GetNumAtoms() + 1 > max_atoms + 1:
            raise ValueError(f"SMILES row {index} has {mol.GetNumAtoms()} atoms; GRASP supports at most {max_atoms}")
    return values


def make_loader(smiles: Iterable[str], vocabulary: dict[int | str, int], batch_size: int,
                shuffle: bool = False) -> DataLoader:
    values = validate_smiles(smiles)
    dataset = MolDataset(pd.Series(values), vocabulary, radius_inp=0,
                         useFeatures_inp=False, cls_token=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def encoder_inputs(batch):
    input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
    relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
    return input_ids, input_mask, relative_pos


class GRASPEncoder:
    def __init__(self, model: AtomEnvEmbeddings, vocabulary: dict[int | str, int],
                 config: dict, source: str, device: str | torch.device):
        self.model = model.to(device)
        self.vocabulary = vocabulary
        self.config = config
        self.source = source
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(cls, source: str | Path = "caithmac/GRASP", *,
                        device: str | torch.device | None = None,
                        revision: str | None = None) -> "GRASPEncoder":
        files = resolve_files(source, ("model.safetensors", "config.json", "vocab.json"), revision)
        config = json.loads(files["config.json"].read_text(encoding="utf-8"))
        if config.get("representation") != "final_layer_cls":
            raise ValueError("Unsupported GRASP representation config")
        model = AtomEnvEmbeddings(ModelConfig.from_dict(config["architecture"]))
        model.load_state_dict(load_file(files["model.safetensors"], device="cpu"), strict=True)
        model.eval()
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, read_vocab(files["vocab.json"]), config, str(source), chosen_device)

    @torch.no_grad()
    def encode(self, smiles: str | Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        """Return final-layer CLS embeddings, one 768-vector per SMILES."""
        values = [smiles] if isinstance(smiles, str) else list(smiles)
        self.model.eval()
        embeddings = []
        for batch in make_loader(values, self.vocabulary, batch_size):
            batch = batch.to(self.device)
            input_ids, input_mask, relative_pos = encoder_inputs(batch)
            result = self.model(input_ids, input_mask,
                                attention_mask=input_mask, relative_pos=relative_pos)
            embeddings.append(result["hidden_states"][-1][:, 0].float().cpu())
        return torch.cat(embeddings).numpy()
