"""rsuq.core.frame — The frame abstraction.

Every paper in the portfolio consumes a frame through ONE interface:

    frame.assignments(h)  -> (kappa_t, sizes_t)   per-position membership + sizes
    frame.K, frame.V

Two implementations:

  FixedFrame    kappa_t == kappa for all positions. Stage I/II, the NeurIPS
                paper, the Token-Barycenter paper. The control condition.

  ContextFrame  the AAAI contribution. The SET of clusters stays fixed; only
                the membership of designated polysemous tokens moves, selected
                by the model's own hidden state at the position (locked
                decision: single-position state, last-four-layer mean-pool).
                The partition remains DISJOINT at every position, so every
                downstream formula (W, BetP, BetP_R) is unchanged in form —
                only its inputs (kappa_t, sizes_t) vary.

Verified invariants (tests/test_core.py, numerically validated first):
  I6  incremental size bookkeeping == full recount; width identical
  I7  per-position partition is disjoint and exhaustive
  I8  pignistic normalises under per-position sizes
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------- interface
@runtime_checkable
class FrameProtocol(Protocol):
    K: int
    V: int

    def assignments(self, h: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (kappa, sizes) for the position(s) whose pooled hidden
        state(s) are h. h: (d,) or (N, d) or None (fixed frames ignore it).
        kappa: (V,) or (N, V) long; sizes: (K,) or (N, K) float."""
        ...


# ------------------------------------------------------------------- shared
def _width_coef(sizes: torch.Tensor) -> torch.Tensor:
    return 1.0 - 1.0 / sizes.clamp(min=1.0)


# -------------------------------------------------------------- fixed frame
@dataclass
class FixedFrame:
    kappa: torch.Tensor              # (V,)
    K: int

    def __post_init__(self):
        counts = torch.bincount(self.kappa, minlength=self.K)
        if (counts == 0).any():
            raise ValueError("empty clusters in fixed frame")
        self.V = int(self.kappa.numel())
        self.sizes = counts.float()
        self.width_coef = _width_coef(self.sizes)
        self.log_sizes = self.sizes.log()

    def assignments(self, h=None):
        return self.kappa, self.sizes

    # construction from a static embedding matrix (wte) — context-free
    @staticmethod
    def from_model(model, K: int = 200, pca_dim: int = 64,
                   seed: int = 0) -> "FixedFrame":
        """Model-agnostic: works for GPT-2, Llama, etc. via the HF API."""
        emb = model.get_input_embeddings().weight.detach().cpu().numpy()
        return FixedFrame.from_embeddings(emb, K, pca_dim, seed)

    @staticmethod
    def from_embeddings(emb: np.ndarray, K: int = 200, pca_dim: int = 64,
                        seed: int = 0) -> "FixedFrame":
        from sklearn.decomposition import PCA
        from sklearn.cluster import MiniBatchKMeans
        Z = PCA(n_components=min(pca_dim, emb.shape[1]),
                random_state=seed).fit_transform(emb)
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3,
                             batch_size=4096).fit(Z)
        return FixedFrame(kappa=torch.from_numpy(km.labels_).long(), K=K)

    def save(self, path):  torch.save({"kappa": self.kappa, "K": self.K}, path)
    @staticmethod
    def load(path):        d = torch.load(path); return FixedFrame(d["kappa"], d["K"])


# ------------------------------------------------------------ context frame
@dataclass
class ContextFrame:
    """AAAI frame: fixed cluster set, per-position membership for polysemous
    tokens, selected by the hidden state h_t against per-token sense centroids.

    base:            the underlying FixedFrame (the control condition)
    poly_tokens:     (P,) long — designated polysemous token ids
    sense_centroids: dict w -> (S_w, d) float — centroids of w's contextual
                     occurrence states (built by SenseInventory)
    sense_cluster:   dict w -> (S_w,) long — host cluster of each sense

    Selection rule (LOCKED, see DECISION_context_vector.md): nearest sense
    centroid to the single-position pooled hidden state. No bag-average, no
    learned aggregator (faithfulness/endogeneity ground).
    """
    base: FixedFrame
    poly_tokens: torch.Tensor
    sense_centroids: dict = field(default_factory=dict)
    sense_cluster: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K, self.V = self.base.K, self.base.V
        self._base_out = self.base.kappa[self.poly_tokens]   # (P,) outflow

    def _select(self, h: torch.Tensor) -> torch.Tensor:
        """h: (N, d) -> (N, P) chosen cluster per polysemous token."""
        cols = []
        for w in self.poly_tokens.tolist():
            C = self.sense_centroids[w].to(h.device)          # (S_w, d)
            s = torch.cdist(h, C).argmin(-1)                  # (N,)
            cols.append(self.sense_cluster[w].to(h.device)[s])
        return torch.stack(cols, dim=-1)                      # (N, P)

    def assignments(self, h: torch.Tensor):
        if h is None:
            raise ValueError("ContextFrame requires the hidden state h")
        if h.dim() == 1:
            h = h.unsqueeze(0)
        N = h.shape[0]
        new = self._select(h)                                 # (N, P)
        # kappa_t: base copy with polysemous columns replaced (dense path;
        # sparse-delta path is a documented optimisation for large N*V)
        kappa_t = self.base.kappa.to(h.device).unsqueeze(0).repeat(N, 1)
        kappa_t[:, self.poly_tokens.to(h.device)] = new
        # incremental sizes: base - outflow + inflow  (verified identity I6)
        sizes_t = self.base.sizes.to(h.device).unsqueeze(0).repeat(N, 1)
        ones = torch.ones(N, len(self.poly_tokens), device=h.device)
        sizes_t.scatter_add_(-1, self._base_out.to(h.device).expand(N, -1), -ones)
        sizes_t.scatter_add_(-1, new, ones)
        if (sizes_t <= 0).any():
            raise RuntimeError("context reassignment emptied a cluster; "
                               "guard polysemous selection or merge clusters")
        return kappa_t.squeeze(0) if N == 1 else kappa_t, \
               sizes_t.squeeze(0) if N == 1 else sizes_t
