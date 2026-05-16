"""
Synthetic-data test for ConformalDeferral.

Coverage guarantee: with α=0.10, on synthetic h^+ test points drawn from
the same distribution as cal, empirical coverage should be ≥ 0.90 with
probability ≥ 0.90.
"""
import numpy as np
import pytest

from gap.conformal import ConformalDeferral


def test_threshold_set_after_fit():
    cd = ConformalDeferral(alpha=0.10)
    rng = np.random.default_rng(0)
    cd.fit(rng.standard_normal(500))
    assert cd.threshold is not None


def test_should_defer_returns_bool_array():
    cd = ConformalDeferral(alpha=0.10)
    rng = np.random.default_rng(0)
    cd.fit(rng.standard_normal(500))
    mask = cd.should_defer(rng.standard_normal(100))
    assert mask.dtype == bool
    assert len(mask) == 100


def test_empty_calibration_raises():
    cd = ConformalDeferral(alpha=0.10)
    with pytest.raises(ValueError):
        cd.fit(np.array([]))


def test_apply_before_fit_raises():
    cd = ConformalDeferral(alpha=0.10)
    with pytest.raises(RuntimeError):
        cd.should_defer(np.zeros(10))


def test_coverage_meets_guarantee_in_aggregate():
    """Empirical coverage on h^+ test data should be near 1-α on average."""
    alpha = 0.10
    n_trials = 50
    n_cal, n_test = 500, 500
    coverages = []

    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        # h^+ gap distribution: positive mean, normal noise
        gap_pos_cal  = rng.normal(loc=2.0, scale=1.0, size=n_cal)
        gap_pos_test = rng.normal(loc=2.0, scale=1.0, size=n_test)

        cd = ConformalDeferral(alpha=alpha)
        cd.fit(gap_pos_cal)
        coverages.append(cd.coverage(gap_pos_test))

    avg_coverage = np.mean(coverages)
    # Should be approximately 1 - α = 0.90
    # Allow 3 percentage points slack for finite-sample fluctuation
    assert abs(avg_coverage - (1 - alpha)) < 0.03, (
        f"Average coverage {avg_coverage:.4f} too far from {1-alpha}"
    )
