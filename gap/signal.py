"""
The generation-discrimination gap Δ(x).

This is where THE paper's central contribution is computed. It's a thin
module on purpose — the conceptual weight is in the OVA heads + the
conformal calibrator. Δ is just their difference in logit space.

Δ(h) = logit(σ(f_pred(h))) - logit(σ(f_defer(h)))
     = f_pred(h) - f_defer(h)            (since logit(σ(x)) = x)

So in implementation terms, the gap is just the difference of the two raw
logits from OVAHeads. No sigmoid needed. The simplicity here is the point.

Decision rule (after conformal calibration):
    defer if Δ(h) < -θ_hat,
where θ_hat is set by `gap/conformal.py` to give the desired coverage.
"""
from __future__ import annotations
from typing import NamedTuple

import numpy as np
import torch

from probes.ova import OVAHeads
from utils.device import configure_torch_threads, get_torch_device


class GapSignals(NamedTuple):
    """All per-example signals derived from the OVA heads.

    Saved to disk as `signals.pt` for downstream consumption.
    """
    example_ids: np.ndarray   # [2N] str — h^+ rows then h^- rows
    is_pos:      np.ndarray   # [2N] bool — True for h^+, False for h^-
    logit_pred:  np.ndarray   # [2N] f_pred logits
    logit_defer: np.ndarray   # [2N] f_defer logits
    gap:         np.ndarray   # [2N] = logit_pred - logit_defer
    sigma_pred:  np.ndarray   # [2N] = σ(logit_pred), for probe-only baselines
    sigma_defer: np.ndarray   # [2N] = σ(logit_defer)


def gap_signal(
    logit_pred: torch.Tensor,
    logit_defer: torch.Tensor,
) -> torch.Tensor:
    """
    Δ = logit(f_pred) - logit(f_defer), computed directly from raw logits.

    Equivalent to logit(σ(f_pred)) - logit(σ(f_defer)) because logit ∘ σ = id.
    Stays in logit space so positive/negative magnitudes have the right
    units for conformal calibration.
    """
    return logit_pred - logit_defer


def compute_signals(
    model: OVAHeads,
    h: torch.Tensor,           # [2N, d]
    is_pos: np.ndarray,        # [2N] bool
    example_ids: np.ndarray,   # [2N] str
    device: str = "auto",
    batch_size: int = 1024,
) -> GapSignals:
    """
    Run the trained OVA heads over (stacked h^+ and h^-) and return all
    per-example signals in one bundle.
    """
    configure_torch_threads()
    device = get_torch_device(device)
    model = model.to(device).eval()

    logit_p_chunks: list[torch.Tensor] = []
    logit_d_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(h), batch_size):
            end = start + batch_size
            h_batch = h[start:end].to(device)
            lp, ld = model(h_batch)
            logit_p_chunks.append(lp.cpu())
            logit_d_chunks.append(ld.cpu())

    lp = torch.cat(logit_p_chunks).numpy()
    ld = torch.cat(logit_d_chunks).numpy()

    return GapSignals(
        example_ids = example_ids,
        is_pos      = is_pos.astype(bool),
        logit_pred  = lp,
        logit_defer = ld,
        gap         = lp - ld,
        sigma_pred  = _sigmoid(lp),
        sigma_defer = _sigmoid(ld),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def save_signals(signals: GapSignals, path) -> None:
    """Save GapSignals to a .pt file for downstream consumption."""
    torch.save({
        "example_ids": signals.example_ids.tolist(),
        "is_pos":      signals.is_pos,
        "logit_pred":  signals.logit_pred,
        "logit_defer": signals.logit_defer,
        "gap":         signals.gap,
        "sigma_pred":  signals.sigma_pred,
        "sigma_defer": signals.sigma_defer,
    }, path)


def load_signals(path) -> GapSignals:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return GapSignals(
        example_ids = np.array(blob["example_ids"]),
        is_pos      = np.asarray(blob["is_pos"]),
        logit_pred  = np.asarray(blob["logit_pred"]),
        logit_defer = np.asarray(blob["logit_defer"]),
        gap         = np.asarray(blob["gap"]),
        sigma_pred  = np.asarray(blob["sigma_pred"]),
        sigma_defer = np.asarray(blob["sigma_defer"]),
    )
