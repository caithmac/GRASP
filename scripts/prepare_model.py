"""Build the public GRASP model payload from the retained, trusted checkpoint.

This script is for release maintainers. It never downloads or unpickles remote files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp.encoder_arch import RTD25_STEP1_CONFIG

SOURCE_SHA256 = "7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != SOURCE_SHA256:
        raise ValueError("Checkpoint SHA-256 does not match the paper's fixed 50k encoder")

    # Both inputs are retained local research artifacts, never user-supplied files.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("Expected a plain tensor state dict")
    with args.vocabulary.open("rb") as stream:
        original_vocab = pickle.load(stream)
    if len(original_vocab) != 207 or not all(isinstance(k, int) for k in original_vocab):
        raise ValueError("Unexpected atom vocabulary")
    vocab = {str(k): int(v) for k, v in original_vocab.items()}
    vocab.update(PAD=0, MASK=208, UNK=209, CLS=210)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, model_path)
    restored = load_file(model_path, device="cpu")
    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
        raise AssertionError("SafeTensors conversion changed encoder tensors")
    (args.output / "config.json").write_text(
        json.dumps({"architecture": RTD25_STEP1_CONFIG,
                    "source_checkpoint_sha256": SOURCE_SHA256,
                    "representation": "final_layer_cls"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "vocab.json").write_text(json.dumps(vocab, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        f"{sha256(model_path)}  model.safetensors\n"
        f"{sha256(args.output / 'config.json')}  config.json\n"
        f"{sha256(args.output / 'vocab.json')}  vocab.json\n",
        encoding="utf-8",
    )
    print(f"Converted {len(state)} tensors; source and output verified: {model_path}")


if __name__ == "__main__":
    main()
