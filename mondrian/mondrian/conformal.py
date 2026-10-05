"""One common floor rule: use +infinity, retain every test observation."""
from decimal import Decimal, ROUND_CEILING
import numpy as np


def conformal_rank(n, alpha):
    if not 0 < alpha < 1 or n < 0 or int(n) != n:
        raise ValueError("n must be a nonnegative integer and alpha in (0,1)")
    return int((Decimal(int(n) + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))


def _labels(cell, M):
    c = np.asarray(cell)
    if c.ndim != 1 or c.dtype.kind not in "iu" or np.any((c < 0) | (c >= M)):
        raise ValueError("Cell labels must be integer indices in [0,M)")
    return c


def per_cell_quantile(cell_cal, s_cal, M, alpha, floor):
    if M < 1 or int(M) != M or floor < 0 or int(floor) != floor:
        raise ValueError("Positive integer M and nonnegative integer floor required")
    conformal_rank(0, alpha)
    c, s = _labels(cell_cal, M), np.asarray(s_cal, float)
    if s.shape != c.shape or not np.isfinite(s).all():
        raise ValueError("One finite score per calibration row required")
    counts = np.bincount(c, minlength=M)
    q = np.full(M, np.inf)
    for j, n in enumerate(counts):
        k = conformal_rank(n, alpha)
        if n >= floor and k <= n:
            q[j] = np.partition(s[c == j], k - 1)[k - 1]
    q.setflags(write=False)
    counts.setflags(write=False)
    return q, counts


def coverage_by_cell(cell_test, s_test, q):
    q, s = np.asarray(q, float), np.asarray(s_test, float)
    if q.ndim != 1 or np.isnan(q).any() or np.isneginf(q).any():
        raise ValueError("Thresholds must be finite or positive infinity")
    c = _labels(cell_test, len(q))
    if s.shape != c.shape or not np.isfinite(s).all():
        raise ValueError("One finite score per test row required")
    counts = np.bincount(c, minlength=len(q))
    successes = np.bincount(c, weights=(s <= q[c]), minlength=len(q))
    cov = np.divide(successes, counts, out=np.full(len(q), np.nan), where=counts > 0)
    return cov, counts


def functionals(cov_by_cell, n_test_by_cell, alpha):
    """PROVISIONAL definitions; see reports/theory.md, not the missing paper.

    msce: mass-weighted squared cell-coverage error (test-noise-biased plug-in).
    spread: max minus min observed cell coverage.
    concentrated: largest cell undercoverage, optimizing over cell mass weights.
    Empty test cells are undefined and their count is always reported.
    """
    c, n = np.asarray(cov_by_cell, float), np.asarray(n_test_by_cell)
    if c.shape != n.shape or c.ndim != 1 or np.any(n < 0) or not 0 < alpha < 1:
        raise ValueError("Invalid cell coverage inputs")
    mask = n > 0
    if not mask.any() or not np.isfinite(c[mask]).all() or np.any((c[mask] < 0) | (c[mask] > 1)):
        raise ValueError("Observed cells must have finite coverages in [0,1]")
    weights = n[mask] / n.sum()
    delta = c[mask] - (1 - alpha)
    return {"msce": float(weights @ delta**2), "spread": float(np.ptp(c[mask])),
            "concentrated": float(max(0, -delta.min())), "coverage": float(weights @ c[mask]),
            "n_empty_test_cells": int((~mask).sum()), "definition_status": "provisional"}
