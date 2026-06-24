"""rsuq.core.beliefs — Mass heads, within-cluster q, pignistics.

Generalised over the FrameProtocol: every function takes (kappa, sizes)
which may be per-position (N, V)/(N, K) [ContextFrame] or shared (V,)/(K,)
[FixedFrame]. Formulas are unchanged in form — only inputs vary. This is the
controlled-comparison property the AAAI paper leans on.

Design theorem (Prop. 1 of the Stage II story, verified): logit-induced m
AND logit-induced q reconstruct the softmax exactly: BetP_R == p. Hence the
mass is trained wherever non-redundancy with the softmax is claimed.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


def _expand_kappa(kappa: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast (V,) kappa to like's batch shape; pass through (N, V)."""
    return kappa.expand_as(like) if kappa.dim() < like.dim() else kappa


class MassHead(nn.Module):
    """h -> mass logits over K clusters. Softmax keeps m on the simplex —
    validity by construction (no Mobius inversion / clipping)."""

    def __init__(self, d_model: int, K: int, hidden: int | None = None):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                  nn.Linear(hidden, K))
                    if hidden else nn.Linear(d_model, K))

    def forward(self, h):  return self.net(h)
    def masses(self, h):   return F.softmax(self.net(h), dim=-1)


def cluster_prob_mass(p: torch.Tensor, kappa: torch.Tensor, K: int):
    """Collapsed baseline: m_k = sum_{w in F_k} p(w)."""
    kap = _expand_kappa(kappa, p)
    s = torch.zeros(*p.shape[:-1], K, device=p.device, dtype=p.dtype)
    s.scatter_add_(-1, kap, p)
    return s


def within_cluster_q(p: torch.Tensor, kappa: torch.Tensor, K: int):
    """q(w) = p(w) / p(F_{kappa(w)}); all tokens at once."""
    kap = _expand_kappa(kappa, p)
    s = cluster_prob_mass(p, kappa, K)
    return p / (s.gather(-1, kap) + EPS)


def pignistic(m: torch.Tensor, kappa: torch.Tensor, sizes: torch.Tensor):
    """Uniform pignistic BetP(w) = m_{kappa(w)} / |F_{kappa(w)}|.
    Accepts shared or per-position (kappa, sizes)."""
    kap = _expand_kappa(kappa, m.new_empty(*m.shape[:-1], kappa.shape[-1]))
    sz = sizes if sizes.dim() == m.dim() else sizes.expand(*m.shape[:-1], -1)
    return m.gather(-1, kap) / sz.gather(-1, kap)


def ranked_pignistic(m, q, kappa, sizes, lam: float = 1.0):
    """BetP_R = m_{kappa(w)} * [(1-lam)/|F| + lam q(w)]. lam=0 -> Stage I."""
    kap = _expand_kappa(kappa, q)
    sz = sizes if sizes.dim() == m.dim() else sizes.expand(*m.shape[:-1], -1)
    inv = 1.0 / sz.gather(-1, kap)
    return m.gather(-1, kap) * ((1 - lam) * inv + lam * q)
