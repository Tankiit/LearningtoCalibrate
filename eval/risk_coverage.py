"""
Risk-coverage analysis. The headline metric for the paper is AURC
(Area Under the Risk-Coverage curve, lower is better).

Concepts (Geifman & El-Yaniv 2017):
  Coverage(τ) = fraction of examples with score ≥ τ (i.e., NOT deferred)
  Risk(τ)     = error rate on those covered examples
  RC curve    = {(coverage(τ), risk(τ)) : τ}
  AURC        = ∫ Risk dCoverage

A score with AURC strictly below the score-only baseline is doing
something better than thresholding the prediction probability.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class RiskCoverageCurve:
    """One curve. Use .aurc for the headline number, .as_dataframe() for plots."""
    coverages: np.ndarray
    risks: np.ndarray
    thresholds: np.ndarray

    @property
    def aurc(self) -> float:
        """Trapezoidal integral of risk over coverage."""
        # Coverages are decreasing-as-threshold-rises; ensure increasing for trapezoid
        order = np.argsort(self.coverages)
        cov = self.coverages[order]
        risk = self.risks[order]
        # np.trapezoid in NumPy 2.0+, np.trapz in older. Try newer first.
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz_fn(risk, cov))

    def as_dataframe(self):
        """For matplotlib / seaborn plotting."""
        import pandas as pd
        return pd.DataFrame({
            "coverage":  self.coverages,
            "risk":      self.risks,
            "threshold": self.thresholds,
        })


def risk_coverage_curve(
    score: np.ndarray,
    correct: np.ndarray,
    higher_is_more_confident: bool = True,
) -> RiskCoverageCurve:
    """
    Compute the full risk-coverage curve.

    score:    [N] real-valued confidence/score for each example.
              Higher = more confident, by default.
    correct:  [N] bool — was the prediction actually correct?
    higher_is_more_confident:
              If False, we negate `score` so the convention is uniform.

    Returns the curve at every distinct threshold value.
    """
    score = np.asarray(score, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if not higher_is_more_confident:
        score = -score
    if len(score) != len(correct):
        raise ValueError(f"Length mismatch: score={len(score)} correct={len(correct)}")

    # Sort by score descending: most confident commits first
    order = np.argsort(-score, kind="stable")
    sorted_correct = correct[order]
    sorted_score = score[order]

    # At threshold = score[i], we commit to the top (i+1) examples.
    # Coverage = (i+1) / N. Risk = error rate on those.
    n = len(score)
    cum_correct = np.cumsum(sorted_correct.astype(int))
    coverages = np.arange(1, n + 1) / n
    risks = 1.0 - cum_correct / np.arange(1, n + 1)

    return RiskCoverageCurve(
        coverages  = coverages,
        risks      = risks,
        thresholds = sorted_score,
    )


def aurc(
    score: np.ndarray,
    correct: np.ndarray,
    higher_is_more_confident: bool = True,
) -> float:
    """Headline number. Lower = better."""
    return risk_coverage_curve(score, correct, higher_is_more_confident).aurc


def coverage_at_acc(
    score: np.ndarray,
    correct: np.ndarray,
    target_acc: float,
    higher_is_more_confident: bool = True,
) -> float:
    """Return the coverage (fraction committed) when risk <= target_acc.

    If no threshold achieves the target accuracy, returns 0.0.
    """
    rc = risk_coverage_curve(score, correct, higher_is_more_confident)
    mask = rc.risks <= target_acc
    if not mask.any():
        return 0.0
    return float(rc.coverages[mask].max())


def coverage_at_accuracy(
    score: np.ndarray,
    correct: np.ndarray,
    target_accuracy: float,
    higher_is_more_confident: bool = True,
) -> float:
    """Return max coverage with selective accuracy >= target_accuracy."""
    if not 0 <= target_accuracy <= 1:
        raise ValueError(f"target_accuracy must be in [0, 1], got {target_accuracy}")
    target_risk = 1.0 - target_accuracy
    return coverage_at_acc(
        score,
        correct,
        target_acc=target_risk,
        higher_is_more_confident=higher_is_more_confident,
    )


def aurc_table(
    scores_by_method: dict[str, np.ndarray],
    correct: np.ndarray,
    higher_is_more_confident_by_method: dict[str, bool] | None = None,
):
    """
    Compute AURC for several scoring methods on the same correctness vector.

    Convenience for the headline table: pass a dict like
        {"gap": gap_scores, "sigma_pred": sigma_pred, "logprob": logprobs}
    and get back a DataFrame with one AURC per method.
    """
    import pandas as pd
    rows = []
    for name, score in scores_by_method.items():
        higher = (higher_is_more_confident_by_method or {}).get(name, True)
        rows.append({
            "method": name,
            "aurc": aurc(score, correct, higher_is_more_confident=higher),
            "coverage_at_acc_80": coverage_at_accuracy(
                score, correct, 0.80, higher_is_more_confident=higher,
            ),
            "coverage_at_acc_90": coverage_at_accuracy(
                score, correct, 0.90, higher_is_more_confident=higher,
            ),
            "n": len(score),
        })
    return pd.DataFrame(rows).sort_values("aurc")
