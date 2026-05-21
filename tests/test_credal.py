import numpy as np

from gap.credal import (
    FixedRadiusCredalCalibrator,
    IsotonicCredalCalibrator,
    build_kl_intervals,
    evaluate_intervals,
    label_kl_scores,
)


def test_label_kl_scores_match_binary_nll():
    p0 = np.array([0.8, 0.8])
    labels = np.array([1, 0])
    scores = label_kl_scores(p0, labels)
    assert np.allclose(scores, [-np.log(0.8), -np.log(0.2)])


def test_kl_intervals_contain_anchor_and_expand_with_radius():
    p0 = np.array([0.25, 0.75])
    small = build_kl_intervals(p0, np.array([0.01, 0.01]))
    large = build_kl_intervals(p0, np.array([0.5, 0.5]))
    assert np.all(small.lower <= p0)
    assert np.all(small.upper >= p0)
    assert np.all(large.width > small.width)


def test_fixed_radius_coverage_on_calibration_distribution():
    rng = np.random.default_rng(0)
    p_cal = rng.uniform(0.1, 0.9, size=500)
    y_cal = rng.binomial(1, p_cal)
    p_test = rng.uniform(0.1, 0.9, size=2000)
    y_test = rng.binomial(1, p_test)

    calibrator = FixedRadiusCredalCalibrator(alpha=0.1)
    calibrator.fit(p_cal, y_cal)
    metrics = evaluate_intervals(calibrator.intervals(p_test), y_test)
    assert metrics["coverage"] >= 0.87


def test_isotonic_adaptive_reduces_width_when_uncertainty_is_informative():
    rng = np.random.default_rng(1)
    n_cal, n_test = 2000, 2000
    u_cal = rng.uniform(0.0, 1.0, size=n_cal)
    u_test = rng.uniform(0.0, 1.0, size=n_test)
    base_cal = rng.binomial(1, 0.5, size=n_cal)
    base_test = rng.binomial(1, 0.5, size=n_test)
    p_cal = np.where(base_cal == 1, 0.75, 0.25)
    p_test = np.where(base_test == 1, 0.75, 0.25)

    flip_cal = u_cal > 0.86
    flip_test = u_test > 0.86
    y_cal = base_cal.copy()
    y_test = base_test.copy()
    y_cal = np.where(flip_cal, 1 - y_cal, y_cal)
    y_test = np.where(flip_test, 1 - y_test, y_test)

    fixed = FixedRadiusCredalCalibrator(alpha=0.1)
    fixed.fit(p_cal, y_cal)
    fixed_metrics = evaluate_intervals(fixed.intervals(p_test), y_test)

    adaptive = IsotonicCredalCalibrator(alpha=0.1, transform="raw")
    adaptive.fit(p_cal, y_cal, u_cal)
    adaptive_metrics = evaluate_intervals(adaptive.intervals(p_test, u_test), y_test)

    assert adaptive_metrics["coverage"] >= 0.87
    assert adaptive_metrics["mean_width"] < fixed_metrics["mean_width"] * 0.95
