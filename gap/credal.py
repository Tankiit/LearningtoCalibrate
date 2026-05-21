"""Conformal credal intervals for calibrated binary correctness probabilities.

The current OVA artifacts are binary: p0(x) is the Platt-calibrated probability
that the model answer is correct. A KL credal set around p0 is therefore a
Bernoulli interval:

    C(x) = {q in [0, 1] : KL(Bern(q) || Bern(p0(x))) <= r(x)}.

Coverage of the observed label y is equivalent to

    -log p0(x)       if y = 1
    -log(1 - p0(x))  if y = 0

being no larger than r(x). The adaptive beta method fits r(x) as an isotonic
function of an uncertainty score and then conformalizes the residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression


ScoreTransform = Literal["abs", "neg", "pos", "raw"]


@dataclass
class CredalIntervals:
    """Credal intervals and radii for a batch of examples."""

    p0: np.ndarray
    radius: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def contains(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=float).reshape(-1)
        if len(labels) != len(self.lower):
            raise ValueError(f"length mismatch: {len(labels)} != {len(self.lower)}")
        return label_kl_scores(self.p0, labels) <= self.radius + 1e-12


@dataclass
class FixedRadiusCredalCalibrator:
    """Split-conformal fixed KL radius."""

    alpha: float = 0.10
    radius: float | None = None

    def fit(self, p0_cal: np.ndarray, labels_cal: np.ndarray) -> float:
        scores = label_kl_scores(p0_cal, labels_cal)
        self.radius = conformal_quantile(scores, self.alpha)
        return self.radius

    def radii(self, p0: np.ndarray) -> np.ndarray:
        if self.radius is None:
            raise RuntimeError("Call fit() before radii().")
        return np.full(len(np.asarray(p0).reshape(-1)), self.radius, dtype=float)

    def intervals(self, p0: np.ndarray) -> CredalIntervals:
        return build_kl_intervals(p0, self.radii(p0))


@dataclass
class IsotonicCredalCalibrator:
    """Conformalized monotone radius r(u), where u is an uncertainty signal."""

    alpha: float = 0.10
    transform: ScoreTransform = "abs"
    min_radius: float = 0.0
    _iso: IsotonicRegression | None = None
    _offset: float | None = None

    def fit(self, p0_cal: np.ndarray, labels_cal: np.ndarray, uncertainty_cal: np.ndarray) -> float:
        scores = label_kl_scores(p0_cal, labels_cal)
        u = transform_uncertainty(uncertainty_cal, self.transform)
        self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        fitted = self._iso.fit_transform(u, scores)
        residual = scores - fitted
        self._offset = conformal_quantile(residual, self.alpha)
        return float(self._offset)

    def radii(self, p0: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        if self._iso is None or self._offset is None:
            raise RuntimeError("Call fit() before radii().")
        u = transform_uncertainty(uncertainty, self.transform)
        radius = self._iso.predict(u) + self._offset
        return np.maximum(np.asarray(radius, dtype=float), self.min_radius)

    def intervals(self, p0: np.ndarray, uncertainty: np.ndarray) -> CredalIntervals:
        return build_kl_intervals(p0, self.radii(p0, uncertainty))


def label_kl_scores(p0: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Minimal KL radius needed for the credal set to contain each binary label."""
    p = np.clip(np.asarray(p0, dtype=float).reshape(-1), 1e-12, 1.0 - 1e-12)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if len(p) != len(y):
        raise ValueError(f"length mismatch: p0={len(p)} labels={len(y)}")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("labels must be binary 0/1")
    return np.where(y == 1.0, -np.log(p), -np.log1p(-p))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with finite-sample correction."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) == 0:
        raise ValueError("cannot fit conformal quantile on empty scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    q_level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def transform_uncertainty(values: np.ndarray, transform: ScoreTransform = "abs") -> np.ndarray:
    """Convert beta or another score into a monotone uncertainty covariate."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if transform == "abs":
        return np.abs(arr)
    if transform == "neg":
        return np.maximum(0.0, -arr)
    if transform == "pos":
        return np.maximum(0.0, arr)
    if transform == "raw":
        return arr
    raise ValueError(f"unknown transform {transform!r}")


def build_kl_intervals(p0: np.ndarray, radius: np.ndarray) -> CredalIntervals:
    """Build Bernoulli KL intervals around p0 for each radius."""
    p = np.clip(np.asarray(p0, dtype=float).reshape(-1), 1e-12, 1.0 - 1e-12)
    r = np.maximum(np.asarray(radius, dtype=float).reshape(-1), 0.0)
    if len(p) != len(r):
        raise ValueError(f"length mismatch: p0={len(p)} radius={len(r)}")

    lower = p.copy()
    upper = p.copy()
    nonzero = r > 0.0
    full = r >= np.maximum(_bernoulli_kl_array(np.zeros_like(p), p), _bernoulli_kl_array(np.ones_like(p), p))
    active = nonzero & ~full

    lower[full] = 0.0
    upper[full] = 1.0

    if np.any(active):
        pa = p[active]
        ra = r[active]

        left = np.zeros_like(pa)
        right = pa.copy()
        for _ in range(60):
            mid = (left + right) / 2.0
            ok = _bernoulli_kl_array(mid, pa) <= ra
            right = np.where(ok, mid, right)
            left = np.where(ok, left, mid)
        lower[active] = right

        left = pa.copy()
        right = np.ones_like(pa)
        for _ in range(60):
            mid = (left + right) / 2.0
            ok = _bernoulli_kl_array(mid, pa) <= ra
            left = np.where(ok, mid, left)
            right = np.where(ok, right, mid)
        upper[active] = left

    return CredalIntervals(p0=p, radius=r, lower=lower, upper=upper)


def evaluate_intervals(intervals: CredalIntervals, labels: np.ndarray) -> dict[str, float]:
    """Return empirical coverage and width metrics for credal intervals."""
    covered = intervals.contains(labels)
    width = intervals.width
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "mean_radius": float(np.mean(intervals.radius)),
        "n": int(len(width)),
    }


def _kl_interval_one(p: float, radius: float) -> tuple[float, float]:
    if radius <= 0.0:
        return p, p
    if radius >= max(_bernoulli_kl(0.0, p), _bernoulli_kl(1.0, p)):
        return 0.0, 1.0

    lo_left, lo_right = 0.0, p
    for _ in range(60):
        mid = (lo_left + lo_right) / 2.0
        if _bernoulli_kl(mid, p) <= radius:
            lo_right = mid
        else:
            lo_left = mid

    hi_left, hi_right = p, 1.0
    for _ in range(60):
        mid = (hi_left + hi_right) / 2.0
        if _bernoulli_kl(mid, p) <= radius:
            hi_left = mid
        else:
            hi_right = mid

    return float(lo_right), float(hi_left)


def _bernoulli_kl(q: float, p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    q = float(np.clip(q, 0.0, 1.0))
    if q == 0.0:
        return float(-np.log1p(-p))
    if q == 1.0:
        return float(-np.log(p))
    return float(q * np.log(q / p) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p)))


def _bernoulli_kl_array(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    out = np.empty_like(p, dtype=float)
    zero = q == 0.0
    one = q == 1.0
    middle = ~(zero | one)
    out[zero] = -np.log1p(-p[zero])
    out[one] = -np.log(p[one])
    out[middle] = (
        q[middle] * np.log(q[middle] / p[middle])
        + (1.0 - q[middle]) * np.log((1.0 - q[middle]) / (1.0 - p[middle]))
    )
    return out
