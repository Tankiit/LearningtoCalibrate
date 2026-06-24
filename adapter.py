"""adapter.py — THE ONE FILE YOU EDIT.

Everything else in this deliverable is protocol-pinned and should run unchanged.
This file binds *your* primitives (frame builder, mass head, pooled hidden
states, logprob foil, error labels) to the interface that `screen.py` expects.
Fill the four functions below; do not change their signatures.

Design intent (so divergence doesn't creep in between you and us):
  - The SCREEN (AURC, bootstrap CI, partial-corr, noise-floor gate) is fixed in
    `screen.py`. Do NOT reimplement it per cell — that's how cells stop being
    comparable across labs.
  - W is computed by `credal_width` from the RSUQ library when it imports. If
    `rsuq` is on your path you get the library version and we assert it. If you
    must stub it (e.g. quick local test without rsuq installed), the banner at
    import time will SHOUT, and any results from a stubbed W must NOT go in the
    paper — re-run with the real library.
  - Pooling is last-4-layer mean at the first answer token
    (DECISION_context_vector doc, LOCKED). If your cache is already pre-pooled,
    `load_cell` just returns it as-is and `has_raw=False`.

Reference implementation: see `run_repspace_deferral.py` in this repo for a
fully worked example of all four stubs against the HaluEval/llama3_8b cache.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# 0. The W implementation — library, never silently stubbed.
# ---------------------------------------------------------------------------
try:
    from rsuq.core.signals import credal_width as _credal_width   # the real one
    RSUQ_LIB = True
except Exception:                                                  # pragma: no cover
    RSUQ_LIB = False

    def _credal_width(m: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """FALLBACK. Disjoint-frame width  W = Σ_k m_k (1 − 1/|F_k|).
        Mathematically correct for the disjoint case, but if this path runs,
        results are NOT paper-grade until re-run inside the RSUQ repo. The
        import banner below warns loudly; screen.py will also refuse to write
        a verdict JSON with `library=False`."""
        sizes = np.asarray(sizes, dtype=float)
        return (np.asarray(m, dtype=float) * (1.0 - 1.0 / sizes)).sum(axis=-1)


def credal_width(m: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Public alias. m: (N, K) mass, sizes: (K,) |F_k|. Returns (N,) W."""
    return _credal_width(m, sizes)


if not RSUQ_LIB:
    _W = "!" * 78
    print(f"\n{_W}\n!! RSUQ LIBRARY NOT FOUND — running STUB credal_width.\n"
          f"!! Results from this run are NOT paper-grade. Install rsuq or\n"
          f"!! re-run inside the rsuq repo before reporting numbers.\n{_W}\n")


# ---------------------------------------------------------------------------
# 1. The data contract for one cell.
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    """One (model, dataset) cell from the deferral cache.

    h        : (N, d) pooled hidden states  (last-4-mean @ first answer tok)
    conf     : (N,)   the logprob foil  (lp_*), HIGHER = more confident
    err      : (N,)   binary error indicator, 1 = model was wrong (deferral target)
    name     : str    e.g. 'llama3_8b/popqa'
    has_raw  : bool   True if raw per-layer hidden states are available
                      (only needed if you plan to re-pool for ablations)
    """
    h: np.ndarray
    conf: np.ndarray
    err: np.ndarray
    name: str
    has_raw: bool = False


# ---------------------------------------------------------------------------
# 2. THE FOUR FUNCTIONS YOU IMPLEMENT.  (signatures are frozen)
# ---------------------------------------------------------------------------
def load_cell(cache_path: str) -> Cell:
    """Load one (model,dataset) cell. Return a Cell.

    Pin: `conf` is the REAL lp_* logprob (not a proxy), `err` is the deferral
    error label. If your cache stores contrastive h+/h- pairs, collapse to a
    per-item h here (first answer token). Set has_raw=True only if the cache
    also has per-layer states for the optional re-pool ablation.
    """
    raise NotImplementedError("bind to your cache loader; return a Cell")


def build_frame(h: np.ndarray, K: int, seed: int):
    """KMeans (or your RS-LLM frame builder) over pooled h.

    MUST return an object `frame` with at least:
        frame.centroids -> (K, d)
        frame.sizes     -> (K,) cluster cardinalities |F_k|
                           (how many vocab/rep atoms fall in F_k — the thing
                           W needs)
    Pin: disjoint partition. `sizes` are the |F_k| used in W.
    """
    raise NotImplementedError("bind to your frame builder; disjoint partition")


def mass_head(h: np.ndarray, frame, mode: str, seed: int, tau: float = 1.0) -> np.ndarray:
    """h -> mass m ∈ Δ^{K-1}, shape (N, K).

    mode='geometric' : m_k ∝ exp(-||h - c_k||^2 / tau)   [HEADLINE arm, no err labels]
    mode='trained'   : small MLP head, may use err labels  [ABLATION arm]

    Pin: rows sum to 1; the geometric arm sees NO error labels.
    """
    raise NotImplementedError("bind to your mass head; respect the two modes")


def frame_sizes(frame) -> np.ndarray:
    """Return |F_k| as (K,) — the cardinalities W weights by. Usually
    `frame.sizes`. Provided as a function so non-KMeans frame builders can
    override the cardinality accounting without touching the screen."""
    return np.asarray(frame.sizes, dtype=float)
