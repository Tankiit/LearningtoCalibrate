"""
Split conformal calibration on the gap signal.

The deferral rule from §4.4 of the paper:
    Defer iff Δ(h) < -θ̂,
where θ̂ is the (1-α)(1+1/n)-level empirical quantile of {-Δ(h^+)} on
held-out correct-answer representations from the calibration split.

Why correct-answer-only calibration: we want the threshold to satisfy
"among non-deferred predictions, at most α are wrong." So we calibrate
the score distribution on examples where the ground truth is "should NOT
defer" (i.e., h^+ representations). The conformal quantile of -Δ on those
becomes the abstention threshold.

This is the standard split-conformal selective-classification setup
(Angelopoulos & Bates 2023, §3.3); not novel to this paper. The only
paper-specific choice is using -Δ as the non-conformity score.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalDeferral:
    """
    Calibrated deferral rule. After `fit()`, `should_defer(gap)` returns a
    boolean mask of which examples to defer.

    alpha: target miscoverage rate. Lower α → fewer deferrals but stronger
           guarantee on commits.
    """
    alpha: float = 0.10
    threshold: float | None = None  # set by fit()

    def fit(self, gap_pos_cal: np.ndarray) -> float:
        """
        Fit threshold on calibration split.

        gap_pos_cal: 1-D array of Δ(h^+) values from the cal split.
                     ONLY h^+ values — these are the ground-truth-correct examples.

        Sets self.threshold and returns it.
        """
        if len(gap_pos_cal) == 0:
            raise ValueError("Cannot calibrate on empty cal split")

        # Non-conformity score is -Δ on h^+. We want the (1-α)(1+1/n) quantile.
        scores = -np.asarray(gap_pos_cal, dtype=float)
        n = len(scores)
        q_level = min((1 - self.alpha) * (1 + 1 / n), 1.0)
        self.threshold = float(np.quantile(scores, q_level))
        return self.threshold

    def should_defer(self, gap: np.ndarray) -> np.ndarray:
        """
        Return boolean mask: True where we should defer.

        Decision rule: defer iff Δ < -threshold,
        equivalently, defer iff -Δ > threshold,
        which is the standard "non-conformity score above threshold = abstain"
        convention.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before should_defer()")
        return np.asarray(gap, dtype=float) < -self.threshold

    def coverage(self, gap_pos_test: np.ndarray) -> float:
        """
        Empirical commit rate on h^+ test examples (sanity check).
        Should be approximately 1 - α.
        """
        commits = ~self.should_defer(gap_pos_test)
        return float(commits.mean())

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalDeferral":
        return cls(alpha=d["alpha"], threshold=d["threshold"])
