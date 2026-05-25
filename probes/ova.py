"""
OVA (One-vs-All) deferral heads.

This is the load-bearing module for the paper. Two independent linear probes
on frozen LLM hidden states:

    f_pred(h)  = σ(w_pred · h + b_pred)    — predicts model correctness
    f_defer(h) = σ(w_defer · h + b_defer)  — predicts expert reliability

Each has its own sigmoid. NO shared softmax, NO shared normalization.
This is what distinguishes OVA L2D (Verma 2023) from softmax L2D
(Mozannar-Sontag 2020): calibration errors don't propagate between the
two heads, because they aren't tied through normalization.

The gap signal Δ(x) is then the difference of the two heads' logits.
See `gap/signal.py` for that computation.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OVATrainConfig:
    """Hyperparameters for OVA head training. See configs/experiments/*.yaml."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 256
    balance_classes: bool = True   # weight loss by class frequency


class OVAHeads(nn.Module):
    """
    Two independent linear binary classifiers over the same frozen feature space.

    Forward returns BOTH logits (in logit space, before sigmoid). The gap is
    the difference of these logits. Don't apply sigmoid before computing the
    gap — it loses the additive structure.

    Architecture: linear only. The paper claims "linear probes on hidden
    states." Adding nonlinearity changes that claim. Don't add a hidden layer.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.pred  = nn.Linear(input_dim, 1)
        self.defer = nn.Linear(input_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, d] frozen hidden states.

        Returns:
          logit_pred:  [B] — model-correctness logit
          logit_defer: [B] — expert-reliability logit

        Both are RAW logits (no sigmoid). Caller computes σ() if needed.
        """
        return self.pred(h).squeeze(-1), self.defer(h).squeeze(-1)

    def predict_probs(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return σ(logits) instead of logits."""
        lp, ld = self(h)
        return torch.sigmoid(lp), torch.sigmoid(ld)


def ova_loss(
    logit_pred: torch.Tensor,
    logit_defer: torch.Tensor,
    y_model: torch.Tensor,
    y_expert: torch.Tensor,
    pos_weight_pred: torch.Tensor | None = None,
    pos_weight_defer: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    OVA loss = BCE(f_pred, y_model) + BCE(f_defer, y_expert).

    Independent BCE per head. Each head only sees its own target. This is
    the property §3.2 of the paper depends on for "calibration errors do
    not propagate between the two estimates."

    pos_weight_*: scalar tensors for class balancing. Compute as
                  (n_negative / n_positive) on the training set.
    """
    loss_pred = F.binary_cross_entropy_with_logits(
        logit_pred, y_model.float(), pos_weight=pos_weight_pred,
    )
    loss_defer = F.binary_cross_entropy_with_logits(
        logit_defer, y_expert.float(), pos_weight=pos_weight_defer,
    )
    return loss_pred + loss_defer


def compute_pos_weights(
    y_model: torch.Tensor, y_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pos_weight values for class-balanced BCE.
    pos_weight = n_negative / n_positive. Floor at 1e-3, ceiling at 1e3
    to prevent collapse on near-degenerate labels.
    """
    def _safe_ratio(y: torch.Tensor) -> torch.Tensor:
        n_pos = y.float().sum().clamp_min(1.0)
        n_neg = (1 - y.float()).sum().clamp_min(1.0)
        return torch.clamp(n_neg / n_pos, min=1e-3, max=1e3)
    return _safe_ratio(y_model), _safe_ratio(y_expert)
