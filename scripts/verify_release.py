"""Check conversion, tokenizer identity, encoder parity, and predictor reload."""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from DeBERTa.deberta.config import ModelConfig
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp import GRASPEncoder, GRASPPredictor
from grasp.encoder import validate_smiles
from mole.training.data.utils import open_dictionary
from mole.training.models.mole import AtomEnvEmbeddings

EXPECTED_SHA = "7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("hf_model"))
    parser.add_argument("--predictor-dir", type=Path, default=Path("runs/frozen_smoke"))
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == EXPECTED_SHA
    original_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    safe_state = load_file(args.model_dir / "model.safetensors", device="cpu")
    assert original_state.keys() == safe_state.keys()
    assert all(torch.equal(original_state[key], safe_state[key]) for key in original_state)
    del safe_state
    gc.collect()
    released = GRASPEncoder.from_pretrained(args.model_dir, device="cpu")
    assert released.vocabulary == open_dictionary(str(args.vocabulary))
    original = AtomEnvEmbeddings(ModelConfig.from_dict(released.config["architecture"]))
    original.load_state_dict(original_state, strict=True)
    del original_state
    gc.collect()
    original.eval()
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    new_result = released.encode(smiles)
    # The retained model path uses the same graph tensors; compare output with the raw state.
    from grasp.encoder import encoder_inputs, make_loader
    with torch.no_grad():
        batch = next(iter(make_loader(smiles, open_dictionary(str(args.vocabulary)), len(smiles))))
        tokens, mask, distances = encoder_inputs(batch)
        old_result = original(tokens, mask, attention_mask=mask,
                              relative_pos=distances)["hidden_states"][-1][:, 0].numpy()
    np.testing.assert_allclose(new_result, old_result, rtol=0, atol=0)
    del original, released
    gc.collect()
    try:
        validate_smiles(["CCO", "not_smiles"])
    except ValueError as exc:
        assert "row 1" in str(exc)
    else:
        raise AssertionError("Invalid SMILES was accepted")
    try:
        validate_smiles(["C" * 512])
    except ValueError as exc:
        assert "511" in str(exc)
    else:
        raise AssertionError("Over-capacity molecule was accepted")
    predictor = GRASPPredictor.from_pretrained(args.predictor_dir, device="cpu")
    first = predictor.predict(smiles)
    del predictor
    gc.collect()
    second = GRASPPredictor.from_pretrained(args.predictor_dir, device="cpu").predict(smiles)
    np.testing.assert_array_equal(first, second)
    print("PASS: checkpoint, vocabulary, encoder outputs, invalid inputs, and predictor reload")


if __name__ == "__main__":
    main()
