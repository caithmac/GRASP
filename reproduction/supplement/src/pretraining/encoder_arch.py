"""Checkpoint-aware MolE encoder configuration and state-dict loading."""
from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any


BASE_CONFIG = dict(
    embedding_size=768,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    attention_head_size=64,
    attention_probs_dropout_prob=0.1,
    hidden_dropout_prob=0.1,
    hidden_act="gelu",
    initializer_range=0.02,
    layer_norm_eps=1e-7,
    max_position_embeddings=0,
    max_relative_positions=64,
    position_buckets=0,
    norm_rel_ebd="layer_norm",
    pos_att_type="p2c|c2p",
    position_biased_input=False,
    relative_attention=True,
    share_att_key=True,
    type_vocab_size=0,
    vocab_size=211,
)

LARGE_CONFIG = dict(
    BASE_CONFIG,
    embedding_size=1024,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    max_position_embeddings=512,
    max_relative_positions=512,
)

RTD25_STEP1_CONFIG = dict(
    BASE_CONFIG,
    max_position_embeddings=512,
    max_relative_positions=-1,
    position_buckets=-1,
    norm_rel_ebd="none",
    pos_att_type="c2p",
    position_biased_input=True,
    share_att_key=False,
)

_LAYER_NUMBER = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")


def _shape(state: Mapping[str, Any], suffix: str) -> tuple[int, ...] | None:
    for key, value in state.items():
        if key.endswith(suffix):
            return tuple(value.shape)
    return None


def encoder_state_dict(raw: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Normalize RTD, MLM, wrapped encoder, or clean encoder checkpoints."""
    state = raw.get("state_dict", raw.get("model", raw))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain a state-dict mapping")

    if any(key.startswith("model.generator.") for key in state):
        generator_key = "model.generator.embeddings.word_embeddings.weight"
        bias_key = "model.disc_word_bias"
        if generator_key not in state or bias_key not in state:
            raise KeyError("RTD checkpoint is missing GDES embedding tensors")
        output = OrderedDict()
        output["embeddings.word_embeddings.weight"] = state[generator_key] + state[bias_key]
        for key, value in state.items():
            if key.startswith("model.discriminator.") and ".embeddings.word_embeddings." not in key:
                output[key[len("model.discriminator."):]] = value
        return output

    for prefix in ("model.encoder.", "module.encoder."):
        if any(key.startswith(prefix) for key in state):
            return OrderedDict(
                (key[len(prefix):], value)
                for key, value in state.items()
                if key.startswith(prefix)
            )

    # Full Step-2 training state (as opposed to the exported clean snapshot).
    if any(key.startswith("encoder.embeddings.") for key in state):
        return OrderedDict(
            (key[len("encoder."):], value)
            for key, value in state.items()
            if key.startswith("encoder.")
        )

    if any(key.startswith("embeddings.") for key in state):
        return OrderedDict(state)
    raise KeyError("Unrecognized checkpoint: no RTD, MLM, wrapped, or clean encoder keys")


def load_encoder_state(path: str, *, weights_only: bool = False) -> OrderedDict[str, Any]:
    """Load and normalize an encoder checkpoint; torch is imported lazily."""
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=weights_only)
    return encoder_state_dict(raw)


def resolve_encoder_config(
    state: Mapping[str, Any] | None = None,
    requested: str = "auto",
    *,
    dropout: float | None = None,
) -> dict[str, Any]:
    """Resolve a MolE config, inferring all shape-bearing fields from weights."""
    requested = requested.lower()
    profiles = {
        "auto": BASE_CONFIG,
        "base": BASE_CONFIG,
        "transfer": BASE_CONFIG,
        "large": LARGE_CONFIG,
        "rtd25_step1": RTD25_STEP1_CONFIG,
    }
    if requested not in profiles:
        raise ValueError(
            f"Unknown ENCODER_ARCH={requested!r}; use auto, base, large, transfer, or rtd25_step1"
        )
    config = dict(profiles[requested])

    if state and requested == "auto":
        word_shape = _shape(state, "embeddings.word_embeddings.weight")
        query_shape = _shape(state, "encoder.layer.0.attention.self.query_proj.weight")
        intermediate_shape = _shape(state, "encoder.layer.0.intermediate.dense.weight")
        if not word_shape or not query_shape or not intermediate_shape:
            raise KeyError("Encoder state dict lacks embeddings or layer-0 projection weights")

        hidden_size = query_shape[1]
        embedding_size = word_shape[1]
        layer_numbers = {
            int(match.group(1))
            for key in state
            if (match := _LAYER_NUMBER.search(key))
        }
        if not layer_numbers:
            raise KeyError("Encoder state dict has no encoder.layer.N tensors")
        num_layers = max(layer_numbers) + 1
        if layer_numbers != set(range(num_layers)):
            raise ValueError("Encoder state dict has a non-contiguous layer sequence")
        if hidden_size % 64:
            raise ValueError(f"Cannot infer 64-wide attention heads from hidden_size={hidden_size}")

        config = dict(LARGE_CONFIG if hidden_size == 1024 and num_layers == 24 else BASE_CONFIG)
        config.update(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_shape[0],
            num_hidden_layers=num_layers,
            num_attention_heads=query_shape[0] // 64,
            attention_head_size=64,
            vocab_size=word_shape[0],
        )

        position_shape = _shape(state, "embeddings.position_embeddings.weight")
        config["position_biased_input"] = position_shape is not None
        if position_shape:
            config["max_position_embeddings"] = position_shape[0]

        relative_shape = _shape(state, "encoder.rel_embeddings.weight")
        if relative_shape:
            config["max_relative_positions"] = relative_shape[0] // 2
            config["position_buckets"] = 0
        config["norm_rel_ebd"] = (
            "layer_norm" if _shape(state, "encoder.LayerNorm.weight") else "none"
        )

        has_pos_key = any(key.endswith("attention.self.pos_key_proj.weight") for key in state)
        has_pos_query = any(key.endswith("attention.self.pos_query_proj.weight") for key in state)
        config["share_att_key"] = not (has_pos_key or has_pos_query)
        if not config["share_att_key"]:
            parts = []
            if has_pos_key:
                parts.append("c2p")
            if has_pos_query:
                parts.append("p2c")
            config["pos_att_type"] = "|".join(parts)

    if dropout is not None:
        config["attention_probs_dropout_prob"] = dropout
        config["hidden_dropout_prob"] = dropout
    return config


def architecture_summary(config: Mapping[str, Any]) -> str:
    return (
        f"{config['num_hidden_layers']}x{config['hidden_size']} "
        f"ffn={config['intermediate_size']} heads={config['num_attention_heads']}"
    )
