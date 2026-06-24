"""rsuq.core.sense_inventory — Build the ContextFrame's sense inventory.

Resolves the three open decisions, now data-driven:

  OD-1  Polysemy by MEASURED SEPARATION (not a hand list). A token enters
        Stratum P iff its occurrence states split into senses with silhouette
        > tau. Validated: poly~0.9, mono~0.06 on separable/unseparable blobs.
  OD-2  Sense count S_w by best silhouette over S in 2..max_senses.
  OD-3  Sense -> host cluster via HIDDEN-SPACE prototypes (both sides of the
        nearest-neighbour comparison in the same space).

Pipeline:
  1. collect_occurrences   : gather single-position last4-mean states for each
                             candidate token across a corpus (offset-aligned)
  2. score_and_build       : separation score -> Stratum P; KMeans senses;
                             host-cluster mapping; assemble ContextFrame inputs
"""

from __future__ import annotations
import torch
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@torch.inference_mode()
def collect_occurrences(model, tokenizer, blocks, candidate_ids: set[int],
                        device="cuda", pool="last4_mean",
                        max_per_token: int = 200) -> dict[int, torch.Tensor]:
    """For each candidate token id, collect its single-position hidden states
    wherever it appears in the corpus. Offset-free here (we already have token
    ids in blocks), so this is just a masked gather over batched states."""
    model.eval()
    base = model.base_model
    store = defaultdict(list)
    for i in range(blocks.shape[0]):
        ids = blocks[i:i+1].to(device)
        out = base(ids, output_hidden_states=(pool == "last4_mean"))
        if pool == "last4_mean":
            h = torch.stack(out.hidden_states[-4:], 0).mean(0)[0]   # (T, d)
        else:
            h = out.last_hidden_state[0]
        row = ids[0]
        for t in range(row.shape[0]):
            tid = int(row[t])
            if tid in candidate_ids and len(store[tid]) < max_per_token:
                store[tid].append(h[t].cpu())
    return {k: torch.stack(v) for k, v in store.items() if len(v) >= 30}


def separation_score(X: np.ndarray, max_senses=4, seed=0):
    """OD-1/OD-2: best silhouette over S in 2..max_senses; returns
    (score, best_S, labels). High score => genuine senses."""
    best = (-1.0, 1, None)
    for S in range(2, max_senses + 1):
        if X.shape[0] <= S:
            break
        lab = KMeans(S, random_state=seed, n_init=5).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if s > best[0]:
            best = (s, S, lab)
    return best


@torch.inference_mode()
def cluster_prototypes(fixed_frame, occ: dict[int, torch.Tensor], d_model):
    """OD-3: hidden-space prototype per fixed cluster = mean occurrence state
    of its member tokens. (K, d)."""
    K = fixed_frame.K
    proto = torch.zeros(K, d_model)
    cnt = torch.zeros(K)
    for tid, X in occ.items():
        k = int(fixed_frame.kappa[tid])
        proto[k] += X.mean(0)
        cnt[k] += 1
    miss = cnt == 0
    proto[~miss] /= cnt[~miss].unsqueeze(-1)
    if miss.any():
        proto[miss] = proto[~miss].mean(0)
    return proto


def build_inventory(fixed_frame, occ: dict[int, torch.Tensor], d_model,
                    tau: float = 0.5, max_senses=4, seed=0):
    """Assemble ContextFrame inputs. Returns dict with poly_tokens,
    sense_centroids, sense_cluster, plus the separation report for Stratum P."""
    proto = cluster_prototypes(fixed_frame, occ, d_model)
    poly, cents, s2c, report = [], {}, {}, {}
    for tid, X in occ.items():
        Xn = X.numpy()
        score, S, lab = separation_score(Xn, max_senses, seed)
        report[tid] = {"separation": float(score), "n_senses": int(S),
                       "n_occ": int(X.shape[0]),
                       "in_stratum_P": bool(score > tau)}
        if score <= tau:
            continue                       # monosemous in THIS model -> skip
        poly.append(tid)
        centroids = torch.stack([X[lab == s].mean(0) for s in range(S)])
        cents[tid] = centroids
        s2c[tid] = torch.cdist(centroids, proto).argmin(-1)   # host clusters
    return {"poly_tokens": torch.tensor(poly, dtype=torch.long),
            "sense_centroids": cents, "sense_cluster": s2c,
            "prototypes": proto, "separation_report": report,
            "n_stratum_P": len(poly)}
