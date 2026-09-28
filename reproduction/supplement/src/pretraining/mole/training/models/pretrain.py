from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torchmetrics import MeanMetric

from DeBERTa.deberta.config import ModelConfig
from DeBERTa.deberta.ops import ACT2FN, LayerNorm

from mole.training.data.utils import TensorDict
from mole.training.models.base import Model
from mole.training.models.mole import AtomEnvEmbeddings
from mole.training.utils.metrics import MetricsDict

__all__ = ["MolERTDModel", "MolEPreTrain", "MolEMLMModel", "MolEMLMPreTrain"]


class MolEMLMHead(nn.Module):
    """Prediction head for the generator (predicts original atom env at masked positions)."""

    def __init__(self, config):
        super().__init__()
        # Project hidden_size → embedding_size so weight tying works when they differ
        embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, embedding_size)
        self.act = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(embedding_size, config.layer_norm_eps)
        
        # Enhanced Mask Decoder: Absolute position embeddings added before the decoder
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 512)
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(self.max_position_embeddings, embedding_size)
        
        self.decoder = nn.Linear(embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states, position_ids=None):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.LayerNorm(x)
        
        if position_ids is not None and hasattr(self, "position_embeddings"):
            pos_embeddings = self.position_embeddings(position_ids)
            x = x + pos_embeddings
            
        return self.decoder(x)  # [B, L, vocab_size]


class MolERTDHead(nn.Module):
    """Per-token binary discriminator head (real=0, replaced=1)."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        
        # Enhanced Mask Decoder for Discriminator
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 512)
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(self.max_position_embeddings, config.hidden_size)
            
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, position_ids=None):
        x = self.act(self.dense(hidden_states))
        
        if position_ids is not None and hasattr(self, "position_embeddings"):
            pos_embeddings = self.position_embeddings(position_ids)
            x = x + pos_embeddings
            
        return self.classifier(x).squeeze(-1)  # [B, L]


class MolERTDModel(nn.Module):
    """
    MolE generator + discriminator for RTD pre-training (DeBERTa-v3 style).

    Generator: small AtomEnvEmbeddings encoder + MLM head.
    Discriminator: full AtomEnvEmbeddings encoder + binary RTD head.
    GDES: discriminator word embeddings = detached generator embeddings + trainable bias.
    """

    def __init__(
        self,
        gen_deberta_config: dict,
        disc_deberta_config: dict,
        vocab_size_inp: Optional[int] = None,
        mask_token_id: int = 0,
        rtd_lambda: float = 50.0,
        embedding_sharing: str = "gdes",
        **kwargs,
    ):
        super().__init__()
        gen_cfg = ModelConfig.from_dict(gen_deberta_config)
        disc_cfg = ModelConfig.from_dict(disc_deberta_config)
        if vocab_size_inp is not None:
            gen_cfg.vocab_size = vocab_size_inp
            disc_cfg.vocab_size = vocab_size_inp

        self.generator = AtomEnvEmbeddings(gen_cfg)
        self.gen_head = MolEMLMHead(gen_cfg)
        # Tie generator MLM decoder weights to generator word embeddings
        self.gen_head.decoder.weight = self.generator.embeddings.word_embeddings.weight

        self.discriminator = AtomEnvEmbeddings(disc_cfg)
        self.disc_head = MolERTDHead(disc_cfg)

        self.mask_token_id = mask_token_id
        self.rtd_lambda = rtd_lambda
        self.embedding_sharing = embedding_sharing.lower()

        if self.embedding_sharing == "gdes":
            self._setup_gdes()

        self._register_disc_hook()

    def _setup_gdes(self):
        # Discriminator word embedding effective weight = gen_weight.detach() + disc_word_bias
        # Only disc_word_bias receives gradients from discriminator loss
        word_bias = torch.zeros_like(self.discriminator.embeddings.word_embeddings.weight)
        self.disc_word_bias = nn.Parameter(word_bias)

    def _register_disc_hook(self):
        def hook(module, *inputs):
            if self.embedding_sharing == "gdes":
                gen_w = self.generator.embeddings.word_embeddings.weight.detach()
                self._set_param(
                    self.discriminator.embeddings.word_embeddings,
                    "weight",
                    gen_w + self.disc_word_bias,
                )
            elif self.embedding_sharing == "es":
                gen_w = self.generator.embeddings.word_embeddings.weight
                self._set_param(
                    self.discriminator.embeddings.word_embeddings,
                    "weight",
                    gen_w,
                )

        self.discriminator.register_forward_pre_hook(hook)

    @staticmethod
    def _set_param(module, name, value):
        if hasattr(module, name):
            delattr(module, name)
        # persistent=False: this is recomputed every forward from gen weight + disc_word_bias,
        # so it must not be saved to the checkpoint (would bloat by ~vocab*hidden floats and
        # cause a parameter↔buffer kind mismatch on load before the first forward fires).
        module.register_buffer(name, value, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        mlm_labels: torch.Tensor,
        original_ids: torch.Tensor,
        relative_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_mask

        # Generate position_ids [0, 1, ..., L-1] for EMD
        B, L = input_ids.shape
        position_ids = torch.arange(L, device=input_ids.device).expand(B, L)

        # ── Generator ──────────────────────────────────────────────────────────
        gen_out = self.generator(
            input_ids, input_mask,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
        )
        gen_hidden = gen_out["hidden_states"][-1]
        gen_logits = self.gen_head(gen_hidden, position_ids=position_ids)  # [B, L, V]

        vocab_size = gen_logits.size(-1)
        gen_loss = F.cross_entropy(
            gen_logits.view(-1, vocab_size),
            mlm_labels.view(-1).long(),
            ignore_index=-100,
        )

        # ── Build discriminator input (no grad) ────────────────────────────────
        with torch.no_grad():
            gen_preds = gen_logits.argmax(dim=-1)          # [B, L]
            masked = mlm_labels != -100                     # positions that were masked
            disc_ids = original_ids.clone()
            disc_ids[masked] = gen_preds[masked]
            # RTD label: 1 where the token was replaced and differs from original
            disc_labels = ((disc_ids != original_ids) & input_mask.bool()).float()

        # ── Discriminator ──────────────────────────────────────────────────────
        disc_out = self.discriminator(
            disc_ids, input_mask,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
        )
        disc_hidden = disc_out["hidden_states"][-1]
        disc_logits = self.disc_head(disc_hidden, position_ids=position_ids)           # [B, L]

        bce = nn.BCEWithLogitsLoss(reduction="none")
        disc_loss = (bce(disc_logits, disc_labels) * input_mask.float()).sum() / input_mask.float().sum()

        total_loss = gen_loss + self.rtd_lambda * disc_loss

        return {
            "loss": total_loss,
            "gen_loss": gen_loss.detach(),
            "disc_loss": disc_loss.detach(),
            "logits": disc_logits.detach(),
            "gen_accuracy": (
                (gen_preds[masked] == original_ids[masked]).float().mean()
                if masked.any() else torch.zeros((), device=input_ids.device)
            ).detach(),
            "disc_accuracy": (
                ((disc_logits > 0) == disc_labels.bool()).float()
                * input_mask.float()
            ).sum().div(input_mask.float().sum().clamp_min(1)).detach(),
            "disc_precision": (
                (((disc_logits > 0) & disc_labels.bool()) * input_mask.bool()).sum().float()
                / (((disc_logits > 0) * input_mask.bool()).sum().clamp_min(1).float())
            ).detach(),
            "disc_recall": (
                (((disc_logits > 0) & disc_labels.bool()) * input_mask.bool()).sum().float()
                / ((disc_labels.bool() * input_mask.bool()).sum().clamp_min(1).float())
            ).detach(),
            "replacement_rate": (
                (disc_labels * input_mask.float()).sum()
                / input_mask.float().sum().clamp_min(1)
            ).detach(),
        }


class MolEPreTrain(Model):
    """PyTorch Lightning module for MolE RTD pre-training."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: partial,
        metrics: MetricsDict,
        lr_scheduler: Optional[partial] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimizer=optimizer, metrics=metrics, lr_scheduler=lr_scheduler, **kwargs)
        self.model = model
        self.metrics.update(
            MetricsDict(
                mean_loss=MeanMetric(),
                mean_gen_loss=MeanMetric(),
                mean_disc_loss=MeanMetric(),
                mean_gen_accuracy=MeanMetric(),
                mean_disc_accuracy=MeanMetric(),
                mean_disc_precision=MeanMetric(),
                mean_disc_recall=MeanMetric(),
                mean_replacement_rate=MeanMetric(),
            )
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore checkpoints saved after the GDES pre-forward hook ran.

        GDES replaces the discriminator word-embedding parameter with a
        non-persistent derived buffer (generator embedding + disc_word_bias).
        Consequently, that redundant tensor is intentionally absent from
        checkpoints written after the first forward.  A newly constructed
        model still has the placeholder parameter when Lightning performs its
        strict state-dict restore, so provide its current value solely to
        satisfy that restore.  The first forward recomputes the effective
        discriminator embedding from the restored learned tensors.
        """
        key = "model.discriminator.embeddings.word_embeddings.weight"
        state_dict = checkpoint.get("state_dict", {})
        if (
            self.model.embedding_sharing == "gdes"
            and key not in state_dict
        ):
            state_dict[key] = (
                self.model.discriminator.embeddings.word_embeddings.weight.detach().clone()
            )

    def setup(self, stage: str) -> None:
        if self.checkpoint_path is not None:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = [v for k, v in state_dict.items() if "state_dict" in k][0]
            state_dict = state_dict[0] if isinstance(state_dict, list) else state_dict
            self.load_state_dict(state_dict, strict=False)

        if stage == "fit":
            with torch.no_grad():
                pad = getattr(self.model.generator.config, "padding_idx", 0)
                self.model.generator.embeddings.word_embeddings.weight[pad].fill_(0)
                if self.model.generator.encoder.relative_attention:
                    self.model.generator.encoder.rel_embeddings.weight[0].fill_(0)
                self.model.discriminator.embeddings.word_embeddings.weight[pad].fill_(0)
                if self.model.discriminator.encoder.relative_attention:
                    self.model.discriminator.encoder.rel_embeddings.weight[0].fill_(0)

    def _step(self, batch: Data) -> Dict[str, Union[torch.Tensor, float]]:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        original_ids, _ = to_dense_batch(batch.original_ids, batch.batch, fill_value=0)
        mlm_labels, _ = to_dense_batch(batch.mlm_labels, batch.batch, fill_value=-100)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)

        return self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            mlm_labels=mlm_labels,
            original_ids=original_ids,
            relative_pos=relative_pos,
        )

    def training_step(self, batch: Data, batch_idx: int):
        return self._step(batch)

    def validation_step(self, batch: Data, batch_idx: int):
        return self._step(batch)

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        self.metrics["mean_loss"].update(outputs["loss"])
        self.metrics["mean_gen_loss"].update(outputs["gen_loss"])
        self.metrics["mean_disc_loss"].update(outputs["disc_loss"])
        self.metrics["mean_gen_accuracy"].update(outputs["gen_accuracy"])
        self.metrics["mean_disc_accuracy"].update(outputs["disc_accuracy"])
        self.metrics["mean_disc_precision"].update(outputs["disc_precision"])
        self.metrics["mean_disc_recall"].update(outputs["disc_recall"])
        self.metrics["mean_replacement_rate"].update(outputs["replacement_rate"])


# ============================================================================
# MolE-MLM: radius-0 input → radius-2 target prediction (true MolE objective)
# ============================================================================

class MolEMLMModel(nn.Module):
    """
    Full 12-layer encoder trained with MLM on radius-0 inputs predicting
    radius-2 atom environment targets (~141k classes).
    No generator, no GDES — pure MLM baseline matching MolE paper's Step 1.
    """

    def __init__(
        self,
        disc_deberta_config: dict,
        vocab_size_inp: int,
        vocab_size_target: int,
        mask_token_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        cfg = ModelConfig.from_dict(disc_deberta_config)
        cfg.vocab_size = vocab_size_inp
        self.encoder = AtomEnvEmbeddings(cfg)

        # MLM head: project hidden → radius-2 target vocab
        embedding_size = getattr(cfg, "embedding_size", cfg.hidden_size)
        self.dense = nn.Linear(cfg.hidden_size, embedding_size)
        self.act = ACT2FN[cfg.hidden_act] if isinstance(cfg.hidden_act, str) else cfg.hidden_act
        self.layer_norm = LayerNorm(embedding_size, cfg.layer_norm_eps)
        self.decoder = nn.Linear(embedding_size, vocab_size_target, bias=True)

        self.mask_token_id = mask_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        mlm_labels: torch.Tensor,
        relative_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_mask

        enc_out = self.encoder(
            input_ids, input_mask,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
        )
        hidden = enc_out["hidden_states"][-1]  # [B, L, H]

        x = self.act(self.dense(hidden))
        x = self.layer_norm(x)
        logits = self.decoder(x)  # [B, L, vocab_size_target]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            mlm_labels.view(-1).long(),
            ignore_index=-100,
        )

        return {"loss": loss, "mlm_loss": loss.detach()}


class MolEMLMPreTrain(Model):
    """PyTorch Lightning module for MolE-MLM pre-training (radius-2 target)."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: partial,
        metrics: MetricsDict,
        lr_scheduler: Optional[partial] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimizer=optimizer, metrics=metrics, lr_scheduler=lr_scheduler, **kwargs)
        self.model = model
        self.metrics.update(MetricsDict(mean_loss=MeanMetric()))

    def setup(self, stage: str) -> None:
        if self.checkpoint_path is not None:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = [v for k, v in state_dict.items() if "state_dict" in k][0]
            state_dict = state_dict[0] if isinstance(state_dict, list) else state_dict
            self.load_state_dict(state_dict, strict=False)

        if stage == "fit":
            with torch.no_grad():
                pad = getattr(self.model.encoder.config, "padding_idx", 0)
                self.model.encoder.embeddings.word_embeddings.weight[pad].fill_(0)
                if self.model.encoder.encoder.relative_attention:
                    self.model.encoder.encoder.rel_embeddings.weight[0].fill_(0)

    def _step(self, batch: Data) -> Dict[str, Union[torch.Tensor, float]]:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        # r2_labels: radius-2 token IDs at masked positions, -100 elsewhere
        mlm_labels, _ = to_dense_batch(batch.r2_mlm_labels, batch.batch, fill_value=-100)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)

        return self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            mlm_labels=mlm_labels,
            relative_pos=relative_pos,
        )

    def training_step(self, batch: Data, batch_idx: int):
        return self._step(batch)

    def validation_step(self, batch: Data, batch_idx: int):
        return self._step(batch)

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        self.metrics["mean_loss"].update(outputs["loss"])
