"""rsuq.core.signals — The uncertainty signals, frame-type agnostic.

Sizes may be per-position (ContextFrame), so width coefficients are computed
from the sizes actually in force at each position. A comparability NOTE the
AAAI paper must carry: under the ContextFrame, |F_{k,t}| varies, so W_t
mixes 'mass spread' with 'ruler change'. Mitigation implemented here:
report BOTH raw W and size-controlled W_base (width computed with base sizes
but contextual membership) — if conclusions hold under both, the ruler
objection is closed empirically.
"""

from __future__ import annotations
import torch

EPS = 1e-12


def credal_width(m: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """W = sum_k m_k (1 - 1/|F_k|), sizes shared (K,) or per-position (N,K)."""
    coef = 1.0 - 1.0 / sizes.clamp(min=1.0)
    return (m * coef).sum(-1) if coef.dim() == m.dim() else m @ coef


def per_cluster_entropy(q: torch.Tensor, kappa: torch.Tensor, K: int):
    kap = kappa.expand_as(q) if kappa.dim() < q.dim() else kappa
    h = -q * (q + EPS).log()
    H = torch.zeros(*q.shape[:-1], K, device=q.device, dtype=q.dtype)
    H.scatter_add_(-1, kap, h)
    return H


def token_choice_axis(m, q, kappa, sizes, K: int,
                      mode: str = "mass_weighted", normalised: bool = True):
    H_k = per_cluster_entropy(q, kappa, K)
    if normalised:
        sz = sizes if sizes.dim() == H_k.dim() else sizes.expand_as(H_k)
        H_k = H_k / sz.clamp(min=2.0).log()        # singletons: H=0 anyway
    if mode == "mass_weighted":
        return (m * H_k).sum(-1)
    if mode == "committed":
        return H_k.gather(-1, m.argmax(-1, keepdim=True)).squeeze(-1)
    raise ValueError(mode)


def pignistic_entropy(betp: torch.Tensor) -> torch.Tensor:
    """AU signal H(BetP) of the NeurIPS EU/AU framing — included so the
    library serves that paper too."""
    return -(betp * (betp + EPS).log()).sum(-1)


def all_signals(m, q, p, kappa, sizes, K: int, base_sizes=None) -> dict:
    from .beliefs import ranked_pignistic
    bp = ranked_pignistic(m, q, kappa, sizes, lam=1.0)
    out = {
        "W": credal_width(m, sizes),
        "Htc_mw": token_choice_axis(m, q, kappa, sizes, K, "mass_weighted"),
        "Htc_com": token_choice_axis(m, q, kappa, sizes, K, "committed"),
        "betp_r_max": bp.max(-1).values,
        "betp_entropy": pignistic_entropy(bp),
        "m_max": m.max(-1).values,
        "softmax_max": p.max(-1).values,
        "softmax_entropy": -(p * (p + EPS).log()).sum(-1),
    }
    if base_sizes is not None:                      # size-controlled width
        out["W_basecoef"] = credal_width(m, base_sizes)
    return out
