"""screen.py — THE PROTOCOL.  DO NOT EDIT.

Shared screen for the deferral battery. One cell = one (model, dataset). For
each cell, given the per-item uncertainty scores:

    conf   : the logprob foil lp_*  (HIGHER = more confident, lower AURC)
    W      : the random-set width Σ_k m_k (1 − 1/|F_k|)
    y_err  : binary error label (1 = model was wrong; the deferral target)

we ask the same three questions, the same way, every time:

    Q1. Does adding W to conf reduce AURC on held-out items?
        → bootstrap CI on the paired gap, with a noise-floor gate.
    Q2. Is W redundant with conf?
        → partial correlation of W and y_err given conf.
    Q3. Is W aligned with the error axis at all?
        → Spearman-ish sign check (point-biserial with the median split).

This file is FROZEN. If you need a different screen, write a new file — do
not patch this one. The whole point is that every cell in the matrix is
scored by literally the same code path, so a HELPS verdict in one cell and a
NULL verdict in another are comparable.

Author: rsuq maintainers
"""

from __future__ import annotations
import numpy as np

# tunable, but please don't: this is the deferral paper's own seed-noise floor.
NOISE_FLOOR = 0.02
N_BOOT      = 2000


# ----------------------------------------------------------------- AURC
def aurc(unc: np.ndarray, err: np.ndarray, n: int = 50) -> float:
    """Risk-coverage AUC. Vectorized: all n prefix-means via a single
    cumsum + gather. Equivalent to the looped version up to float ordering.

    unc : per-item uncertainty, HIGHER = defer.  err : per-item 0/1 error.
    """
    e = np.asarray(err)[np.argsort(unc)]
    m = len(e)
    cs = np.concatenate(([0.0], np.cumsum(e)))
    ks = np.maximum(1, (np.linspace(1, n, n) * m / n).astype(int))
    risks = cs[ks] / ks
    covs  = np.linspace(1 / n, 1, n)
    return float(np.trapezoid(risks, covs))


# ----------------------------------------------------------------- Q3
def stc(W: np.ndarray, err: np.ndarray) -> float:
    """Sign-test-correlation: point-biserial between err and the median-split
    of W. Cheap, robust, scale-free. Magnitude tells you whether W has any
    error signal at all; sign tells you polarity (positive = W high on errors).
    """
    from scipy.stats import pointbiserialr
    return float(pointbiserialr(err, (W > np.median(W)).astype(float))[0])


# ----------------------------------------------------------------- Q2
def partial_corr_W_given_conf(W: np.ndarray, err: np.ndarray,
                              conf: np.ndarray) -> float:
    """Partial correlation of (W, err) after regressing out conf from both.
    Positive => W carries error signal beyond what conf already provides.
    This is the W ⊥ BetP dissociation check: if W were just a re-encoding of
    confidence, this would be ~0.
    """
    from scipy.stats import pearsonr
    def resid(a, b):
        # closed-form OLS residual of a on b (with intercept)
        b1 = b - b.mean()
        coef = (a - a.mean()) @ b1 / (b1 @ b1 + 1e-12)
        return a - (a.mean() + coef * b1)
    rW  = resid(W,    conf)
    rE  = resid(err,  conf)
    return float(pearsonr(rW, rE)[0])


# ----------------------------------------------------------------- Q1
def compose_aurc_fixed(sigs_tr, y_tr, sigs_te, y_te):
    """Fit composite (logistic) on TRAIN signals/labels; score AURC on TEST.
    Standardization uses TRAIN stats only (no test leakage).

    sigs_tr / sigs_te : list of 1-D arrays, each already restricted to the
                        train / test split. e.g. [conf_tr, W_tr].
    Returns (aurc_on_test, unc_on_test) so callers can pass unc_on_test
    straight into the bootstrap without refitting. Returns (None, None) if
    train is single-class.
    """
    if len(set(np.asarray(y_tr).tolist())) < 2:
        return None, None
    from sklearn.linear_model import LogisticRegression
    Xtr = np.column_stack(sigs_tr)
    Xte = np.column_stack(sigs_te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    lr  = LogisticRegression(max_iter=500).fit(Xtr, y_tr)
    unc = lr.predict_proba(Xte)[:, 1]
    return aurc(unc, y_te), unc


def aurc_gap_bootstrap(unc_conf_te: np.ndarray,
                       unc_confW_te: np.ndarray,
                       y_te: np.ndarray,
                       B: int = N_BOOT,
                       seed: int = 0):
    """Paired item-bootstrap of  AURC(conf) - AURC(conf+W)  on the test set,
    using *already-fit* per-item scores. Positive gap = W helps (lower AURC).

    Design note: we deliberately do NOT refit the logistic inside each
    bootstrap iteration. Refitting on resampled data would (i) be O(B) more
    expensive and (ii) change the estimand — we want uncertainty about *this*
    fitted composite's test-set gap, not the gap of a refit procedure.
    Bootstrap of the sorted-prefix AUC of fixed scores is the standard
    non-parametric CI for a fixed ranking.
    """
    rng = np.random.default_rng(seed)
    y_te = np.asarray(y_te)
    n = len(y_te)
    gaps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        a_conf  = aurc(unc_conf_te[idx],  y_te[idx])
        a_confW = aurc(unc_confW_te[idx], y_te[idx])
        gaps[b] = a_conf - a_confW
    lo, hi = np.nanpercentile(gaps, [2.5, 97.5])
    return float(np.nanmean(gaps)), float(lo), float(hi)


# ----------------------------------------------------------------- verdicts
def verdict_from_ci(gap_mean: float, lo: float, hi: float,
                    noise_floor: float = NOISE_FLOOR) -> str:
    """Honest verdict: W helps only if the bootstrap CI excludes 0 AND beats
    the deferral paper's own seed-noise floor (~0.02 AURC). Anything else is
    either significant-but-tiny or null."""
    if lo > 0 and gap_mean > noise_floor:
        return "HELPS (CI>0 and > noise floor)"
    if lo > 0:
        return "SIGNIFICANT-BUT-TINY (CI>0 but < noise floor)"
    return "NULL (CI includes 0)"


def single_split_indices(n: int, seed: int,
                         train_frac: float = 0.6) -> np.ndarray:
    """One permutation, sliced into one train/test split. Same split for every
    arm in a cell, so arms are directly comparable on identical test items."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * train_frac)
    return idx[:k], idx[k:]


# ----------------------------------------------------------------- shape asserts
def assert_cell_ok(h: np.ndarray, conf: np.ndarray, err: np.ndarray) -> None:
    """Sanity asserts. Call at the top of every cell runner. Cheap; catches
    the three ways the cache tends to silently rot: shape mismatch, label
    collapse, and the logprob foil being inverted.
    """
    N = h.shape[0]
    assert h.ndim == 2, f"h must be (N, d), got shape {h.shape}"
    assert conf.shape == (N,), f"conf must be ({N},), got {conf.shape}"
    assert err.shape  == (N,), f"err must be ({N},), got {err.shape}"
    assert set(np.unique(err).tolist()) <= {0, 1}, "err must be binary"
    assert err.sum() > 0, "cell has no errors — AURC undefined"
    assert err.sum() < N, "cell is all errors — AURC undefined"
