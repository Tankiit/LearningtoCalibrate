"""
Baseline scoring methods to compare against the gap.

Per the paper's headline empirical claim: "the gap dominates both components."
That requires running probe-only, logprob-only, and naive-sum baselines
on the same test data and computing AURC for each.

Each baseline maps (signals, logprobs) → score per example. Higher = more
confident-correct, by convention. Pass to `risk_coverage.aurc()`.
"""
from __future__ import annotations
import numpy as np

from gap.signal import GapSignals


def baseline_gap(signals: GapSignals) -> np.ndarray:
    """The paper's proposed score. Δ(h) = logit_pred - logit_defer."""
    return signals.gap


def baseline_sigma_pred(signals: GapSignals) -> np.ndarray:
    """Probe-only: σ(f_pred). The 'just use the probe' baseline."""
    return signals.sigma_pred


def baseline_logit_pred(signals: GapSignals) -> np.ndarray:
    """Probe-only in logit space, for fair comparison with the gap."""
    return signals.logit_pred


def baseline_logprob(logprobs: np.ndarray) -> np.ndarray:
    """Generation logprob — the 'just use the LLM's confidence' baseline.

    Higher logprob = more fluent generation, used as a confidence proxy.
    Aligned with the convention for higher_is_more_confident.
    """
    return np.asarray(logprobs, dtype=float)


def baseline_phillips_sum(signals: GapSignals) -> np.ndarray:
    """Phillips et al. 2026 — closest competitor.

    Adds the probe score to the entropy/logprob signal as a feature.
    Here we approximate by summing logit_pred and logit_defer; replace
    with the actual Phillips formulation when you port their reference
    implementation.

    PORT NOTE: replace this with the actual Phillips score once that
    reference implementation is available.
    """
    return signals.logit_pred + signals.logit_defer


def baseline_random(signals: GapSignals, seed: int = 0) -> np.ndarray:
    """Sanity baseline: random scores. AURC ≈ class-prior risk."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(len(signals.gap))


# Convenience dict for sweeping all baselines at once
ALL_BASELINES = {
    "gap":            baseline_gap,
    "sigma_pred":     baseline_sigma_pred,
    "logit_pred":     baseline_logit_pred,
    "phillips_sum":   baseline_phillips_sum,
    # logprob, random handled separately because of different signatures
}
