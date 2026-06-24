# RSUQ deferral — collaborator handoff

This note tells you how to bind your deferral cache to our screen, run one
cell, and report a verdict that is comparable to ours. **Two files. Read this,
then edit `adapter.py`.**

---

## What's in this folder

| File            | You edit? | What it does |
|-----------------|-----------|--------------|
| `adapter.py`    | **YES**   | The one file you edit. Binds your cache, frame builder, and mass head to the protocol. |
| `screen.py`     | NO        | The protocol: AURC, bootstrap CI, partial-corr, verdict. Frozen. Identical for every cell. |
| `run_repspace_deferral.py` | NO (reference) | A fully worked adapter+driver against the HaluEval/llama3_8b cache. Read it as a worked example. |

`adapter.py` exports a `Cell` dataclass and four functions you fill in. The
screen imports `adapter` and never the other way around.

---

## The contract (read once)

One **cell** = one (model, dataset) pair. For each cell, you provide three
per-item arrays of the same length `N`:

| Field   | Shape | Meaning |
|---------|-------|---------|
| `h`     | (N, d)| Pooled hidden states. **Pinned:** last-4-layer mean at the first answer token. If your cache is pre-pooled, ship it as-is. |
| `conf`  | (N,)  | The logprob foil `lp_*`. **Higher = more confident.** This is the baseline you have to beat. |
| `err`   | (N,)  | Binary, 1 = model was wrong. The deferral target. |

From `h` you build a frame, run a mass head, and the screen scores W = the
random-set width `Σ_k m_k (1 − 1/|F_k|)`. The screen then answers three
questions, every time:

1. **Q1 (does W help?)** Bootstrap CI on `AURC(conf) − AURC(conf+W)`.
   Verdict `HELPS` only if CI excludes 0 **and** the mean gap beats the
   0.02 AURC noise floor.
2. **Q2 (is W redundant with conf?)** Partial correlation of W and err
   given conf. The W ⊥ BetP dissociation check.
3. **Q3 (does W see the error at all?)** Point-biserial between err and a
   median split of W.

The screen is `screen.py`. Do not reimplement it per cell.

---

## The four functions in `adapter.py`

```python
def load_cell(cache_path: str) -> Cell:
    """Load one (model, dataset) cell from your cache. Return a Cell(h, conf, err, name).
    Pin: conf is the real lp_*, err is the deferral error label. If your cache
    stores contrastive h+/h- pairs, collapse to per-item h here (first answer
    token)."""

def build_frame(h, K, seed):
    """KMeans (or your RS-LLM frame builder) over pooled h. MUST return an
    object with `.centroids` (K, d) and `.sizes` (K,) — the |F_k| that W
    weights by. Pin: disjoint partition."""

def mass_head(h, frame, mode, seed, tau=1.0):
    """h -> mass m ∈ Δ^{K-1}, shape (N, K).
      mode='geometric' : m_k ∝ exp(-||h - c_k||^2 / tau)   [HEADLINE arm, no err]
      mode='trained'   : small MLP head, may use err labels  [ABLATION arm]
    Pin: rows sum to 1; geometric sees NO error labels."""

def frame_sizes(frame):
    """|F_k| as (K,). Usually just `frame.sizes`. Override only if your frame
    builder's cardinality accounting differs from a plain KMeans partition."""
```

`credal_width(m, sizes)` is already wired: it tries `rsuq.core.signals.credal_width`
(the library version), and falls back to an inline `Σ_k m_k (1 − 1/|F_k|)`
**with a loud banner** if rsuq isn't importable. **Results from the stub are
not paper-grade.** Either install rsuq on your path, or re-run inside this repo.

---

## Minimal cell runner (you write this, ~30 lines)

```python
import numpy as np
from adapter import load_cell, build_frame, mass_head, frame_sizes, credal_width
from screen   import (assert_cell_ok, single_split_indices, compose_aurc_fixed,
                      aurc_gap_bootstrap, verdict_from_ci, stc, partial_corr_W_given_conf)

cell = load_cell("path/to/your/cache.pt")
assert_cell_ok(cell.h, cell.conf, cell.err)

K, seed = 50, 0
tr, te = single_split_indices(len(cell.err), seed)

frame = build_frame(cell.h[tr], K, seed)
sizes = frame_sizes(frame)

m_te  = mass_head(cell.h[te], frame, mode="geometric", seed=seed)
W_te  = credal_width(m_te, sizes)

aurc_conf,  unc_conf  = compose_aurc_fixed([cell.conf[tr]],            cell.err[tr],
                                           [cell.conf[te]],            cell.err[te])
aurc_confW, unc_confW = compose_aurc_fixed([cell.conf[tr], W_te_like], cell.err[tr],
                                           [cell.conf[te], W_te],      cell.err[te])
# (NOTE: build W_tr the same way as W_te — same frame, mass_head on the train split.)

gap_mean, lo, hi = aurc_gap_bootstrap(unc_conf, unc_confW, cell.err[te], seed=seed)
print(verdict_from_ci(gap_mean, lo, hi))
print("STC:", stc(W_te, cell.err[te]),
      " partial_corr:", partial_corr_W_given_conf(W_te, cell.err[te], cell.conf[te]))
```

`run_repspace_deferral.py` in this folder is the fully-worked version of this
sketch — three arms (`geometric`, `trained_supervised`, `trained_unsupervised`),
multi-seed aggregation, and a K-sweep. Read it, don't ship it as your driver.

---

## How to report a cell back

A per-cell JSON with this shape is all we need:

```json
{
  "name": "llama3_8b/halueval_qa",
  "N": 6554,
  "K": 50,
  "seeds": [0, 1],
  "arms": {
    "geometric": {
      "aurc_conf":     0.1834,
      "aurc_conf_plus_W": 0.1812,
      "gap_mean":      0.0023,
      "gap_ci":        [-0.0011, 0.0058],
      "verdict":       "NULL (CI includes 0)",
      "STC":          -0.0370,
      "partial_corr":  0.014
    }
  },
  "library": true,
  "screen_version": "screen.py@<git-sha>"
}
```

The two fields that make cells comparable across labs:

- **`library: true`** — rsuq was importable and `credal_width` came from the
  library. If you shipped a stubbed W, set this false and flag it; we'll
  re-run.
- **`screen_version`** — the git short SHA of `screen.py` so we know the
  protocol hasn't drifted.

---

## Hard pins (do not negotiate per cell)

1. **Pooling**: last-4-layer mean at the first answer token. If you suspect
   pooling is doing the work, that's a separate ablation — don't change the
   pin.
2. **Foil**: `conf` is the real `lp_*` logprob. Not a proxy, not the
   generation prob, not a softmax max.
3. **Frame**: disjoint partition. `sizes` are the cluster member counts.
4. **Geometric arm sees no error labels.** The headline result is the
   unsupervised width; the trained arm is an ablation, not the headline.
5. **One split, train-only standardization, paired bootstrap on fixed scores.**
   Do not refit inside the bootstrap.
6. **Noise floor 0.02 AURC.** Below that, even a CI-positive gap is
   `SIGNIFICANT-BUT-TINY`, not `HELPS`.

If any of these are wrong for your setup, open an issue before running —
don't silently patch the screen.

---

## Honest result from our reference cell

For transparency, here's what the reference adapter gets on the one cell we
have cached (`llama3_8b/halueval_qa`, K-sweep {20, 50}, 2 seeds):

```
K=20 geometric           gap_mean -0.0003   STC -0.13   ABSTAIN
K=20 trained_supervised  gap_mean +0.0006   STC -0.10   MARGINAL
K=20 trained_unsupervised gap_mean -0.0481  STC -0.00   ABSTAIN
K=50 geometric           gap_mean -0.0051   STC -0.04   ABSTAIN
K=50 trained_supervised  gap_mean -0.0077   STC -0.06   ABSTAIN
K=50 trained_unsupervised gap_mean -0.0329  STC -0.06   ABSTAIN
```

**This is a NULL cell.** W does not consistently beat logprob-only deferral
here. We're shipping the screen so that when you get a non-NULL on your
cells, we can trust the comparison.

---

## Versioning

- `screen.py` carries the protocol. Its git SHA is the protocol version.
- `adapter.py` is yours; version it however you like.
- If we ever need to change the screen, we'll bump a `PROTOCOL_VERSION`
  constant at the top of `screen.py` and bump it here too.

Current protocol version: **1**.
