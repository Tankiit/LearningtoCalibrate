"""rsuq.core.senses — Sense inventories for the ContextFrame (AAAI mechanism).

Pipeline (matches the AAAI design doc):
  1. Identify polysemous tokens (the only tokens whose membership will move).
  2. For each, collect contextual occurrence states — single-position hidden
     states, last-four-layer mean-pool (locked context-vector decision; each
     occurrence IS a single-position state, so the selector is geometrically
     matched to how the senses are built).
  3. Cluster occurrences into S_w sense centroids.
  4. Map each sense to a HOST CLUSTER of the fixed frame.

OPEN DESIGN DECISIONS — marked, not hidden (resolve before AAAI freeze):

  [OD-1] Polysemy detection. v1: occurrence-state dispersion (mean pairwise
         distance) above a percentile threshold, intersected with a frequency
         floor. Alternatives: WordNet sense counts (English-only, citeable),
         silhouette-gap on occurrence clusters. The choice is an ablation
         axis, not a hyperparameter to tune on test.

  [OD-2] Number of senses S_w. v1: fixed S_w = 2..4 chosen by silhouette.
         Reviewers will ask; report sensitivity.

  [OD-3] Sense -> host cluster mapping. The fixed clusters live in static-
         embedding space; senses live in hidden-state space. v1 bridges by
         building HIDDEN-SPACE cluster prototypes: prototype(k) = mean of
         occurrence states of tokens in F_k (sampled). Sense's host =
         nearest prototype. This keeps both sides of the nearest-neighbour
         comparison in the same space (the geometric-match ground, applied
         twice). Alternative: host = cluster of the sense's nearest
         neighbouring TOKEN by occurrence-state similarity.
"""

from __future__ import annotations
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@torch.no_grad()
def pool_last4(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Last-four-layer mean-pool at every position. hidden_states: tuple of
    (1, T, d) from output_hidden_states=True. Returns (T, d)."""
    return torch.stack(hidden_states[-4:], 0).mean(0)[0]


def detect_polysemous(occ_states: dict[int, torch.Tensor],
                      disp_pct: float = 90.0, min_occ: int = 30
                      ) -> torch.Tensor:
    """[OD-1] v1: dispersion-percentile rule over occurrence states."""
    disp = {}
    for w, X in occ_states.items():
        if X.shape[0] < min_occ:
            continue
        c = X.mean(0, keepdim=True)
        disp[w] = float((X - c).norm(dim=-1).mean())
    if not disp:
        return torch.empty(0, dtype=torch.long)
    thr = torch.quantile(torch.tensor(list(disp.values())), disp_pct / 100.0)
    return torch.tensor([w for w, d in disp.items() if d > thr],
                        dtype=torch.long)


def build_senses(occ_states: dict[int, torch.Tensor],
                 poly: torch.Tensor, s_range=(2, 3, 4), seed: int = 0
                 ) -> dict[int, torch.Tensor]:
    """[OD-2] Cluster each polysemous token's occurrences; S_w by silhouette."""
    out = {}
    for w in poly.tolist():
        X = occ_states[w].numpy()
        best, best_s = None, -1.0
        for S in s_range:
            if X.shape[0] <= S:
                continue
            km = KMeans(n_clusters=S, random_state=seed, n_init=5).fit(X)
            sc = silhouette_score(X, km.labels_)
            if sc > best_s:
                best, best_s = km, sc
        out[w] = torch.from_numpy(best.cluster_centers_).float()
    return out


@torch.no_grad()
def cluster_prototypes(fixed_frame, occ_states: dict[int, torch.Tensor],
                       d_model: int) -> torch.Tensor:
    """[OD-3] Hidden-space prototype per fixed cluster: mean occurrence state
    of (sampled) member tokens. Returns (K, d)."""
    K = fixed_frame.K
    proto = torch.zeros(K, d_model)
    count = torch.zeros(K)
    for w, X in occ_states.items():
        k = int(fixed_frame.kappa[w])
        proto[k] += X.mean(0)
        count[k] += 1
    missing = count == 0
    proto[~missing] /= count[~missing].unsqueeze(-1)
    if missing.any():
        proto[missing] = proto[~missing].mean(0)   # fallback; log this
    return proto


def map_senses_to_clusters(sense_centroids: dict[int, torch.Tensor],
                           prototypes: torch.Tensor
                           ) -> dict[int, torch.Tensor]:
    """[OD-3] Host cluster of each sense = nearest hidden-space prototype."""
    return {w: torch.cdist(C, prototypes).argmin(-1)
            for w, C in sense_centroids.items()}
