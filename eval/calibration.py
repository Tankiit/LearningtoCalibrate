"""Calibration metrics and calibrators for f_pred probe outputs.

All public functions operate on NumPy arrays. Callers are responsible for
converting tensors before passing them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


@dataclass
class BinStats:
    """One bin in a reliability diagram."""

    bin_lower: float
    bin_upper: float
    mean_confidence: float
    fraction_positive: float
    count: int
    weight: float


@dataclass
class CalibrationResult:
    """Full calibration evaluation for one model/dataset/method cell."""

    model: str
    dataset: str
    method: str
    ece_15: float
    ece_adaptive: float
    brier_score: float
    log_loss: float
    bins_equal_width: list[BinStats]
    bins_adaptive: list[BinStats]
    seed: Optional[int] = None
    platt_a: Optional[float] = None
    platt_b: Optional[float] = None
    temperature: Optional[float] = None
    n_test: int = 0
    prevalence: float = 0.0
    mean_predicted: float = 0.0
    ambiguity_type: str = "unambiguous"


def fit_platt(logits_cal: np.ndarray, labels_cal: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling, p = sigmoid(a * logit + b), by minimizing NLL."""
    logits_cal = _as_1d_float(logits_cal, "logits_cal")
    labels_cal = _as_binary_labels(labels_cal, "labels_cal")
    _validate_same_length(logits_cal, labels_cal)

    def nll(params: np.ndarray) -> float:
        a, b = params
        p = np.clip(expit(a * logits_cal + b), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(labels_cal * np.log(p) + (1.0 - labels_cal) * np.log(1.0 - p)))

    result = minimize(
        nll,
        x0=np.array([1.0, 0.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.05, 20.0), (-10.0, 10.0)],
    )
    if not result.success:
        raise RuntimeError(f"Platt scaling failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def fit_temperature(logits_cal: np.ndarray, labels_cal: np.ndarray) -> float:
    """Fit temperature scaling, p = sigmoid(logit / T), by minimizing NLL."""
    logits_cal = _as_1d_float(logits_cal, "logits_cal")
    labels_cal = _as_binary_labels(labels_cal, "labels_cal")
    _validate_same_length(logits_cal, labels_cal)

    def nll(log_t: float) -> float:
        temperature = float(np.exp(log_t))
        p = np.clip(expit(logits_cal / temperature), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(labels_cal * np.log(p) + (1.0 - labels_cal) * np.log(1.0 - p)))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature scaling failed: {result.message}")
    return float(np.exp(result.x))


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply a fitted Platt scaler."""
    return expit(float(a) * _as_1d_float(logits, "logits") + float(b))


def ece_equal_width(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[BinStats]]:
    """Expected calibration error with equal-width bins."""
    probs = _as_probabilities(probs, "probs")
    labels = _as_binary_labels(labels, "labels")
    _validate_same_length(probs, labels)
    _validate_bins(n_bins)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    stats: list[BinStats] = []
    ece = 0.0

    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        if k < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        count = int(mask.sum())

        if count == 0:
            stats.append(BinStats(lo, hi, (lo + hi) / 2.0, 0.0, 0, 0.0))
            continue

        confidence = float(probs[mask].mean())
        fraction_positive = float(labels[mask].mean())
        weight = count / n
        ece += weight * abs(confidence - fraction_positive)
        stats.append(BinStats(lo, hi, confidence, fraction_positive, count, weight))

    return float(ece), stats


def ece_adaptive(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[BinStats]]:
    """Expected calibration error with equal-mass bins."""
    probs = _as_probabilities(probs, "probs")
    labels = _as_binary_labels(labels, "labels")
    _validate_same_length(probs, labels)
    _validate_bins(n_bins)

    n = len(probs)
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order]
    bin_edges = np.linspace(0, n, min(n_bins, n) + 1, dtype=int)
    stats: list[BinStats] = []
    ece = 0.0

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end <= start:
            continue
        bin_probs = sorted_probs[start:end]
        bin_labels = sorted_labels[start:end]
        count = len(bin_probs)
        confidence = float(bin_probs.mean())
        fraction_positive = float(bin_labels.mean())
        weight = count / n
        ece += weight * abs(confidence - fraction_positive)
        stats.append(
            BinStats(
                float(bin_probs[0]),
                float(bin_probs[-1]),
                confidence,
                fraction_positive,
                count,
                weight,
            )
        )

    return float(ece), stats


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for binary probabilities."""
    probs = _as_probabilities(probs, "probs")
    labels = _as_binary_labels(labels, "labels")
    _validate_same_length(probs, labels)
    return float(np.mean((probs - labels) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Binary negative log likelihood."""
    probs = _as_probabilities(probs, "probs")
    labels = _as_binary_labels(labels, "labels")
    _validate_same_length(probs, labels)
    p = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def evaluate(
    probs: np.ndarray,
    labels: np.ndarray,
    model: str,
    dataset: str,
    method: str,
    ambiguity_type: str = "unambiguous",
    n_bins: int = 15,
    seed: Optional[int] = None,
    platt_a: Optional[float] = None,
    platt_b: Optional[float] = None,
    temperature: Optional[float] = None,
) -> CalibrationResult:
    """Compute all calibration metrics for one method."""
    probs = _as_probabilities(probs, "probs")
    labels = _as_binary_labels(labels, "labels")
    _validate_same_length(probs, labels)

    ece_15, bins_15 = ece_equal_width(probs, labels, n_bins)
    ece_adp, bins_adp = ece_adaptive(probs, labels, n_bins)

    return CalibrationResult(
        model=model,
        dataset=dataset,
        method=method,
        seed=seed,
        ece_15=ece_15,
        ece_adaptive=ece_adp,
        brier_score=brier_score(probs, labels),
        log_loss=log_loss(probs, labels),
        bins_equal_width=bins_15,
        bins_adaptive=bins_adp,
        platt_a=platt_a,
        platt_b=platt_b,
        temperature=temperature,
        n_test=len(probs),
        prevalence=float(labels.mean()),
        mean_predicted=float(probs.mean()),
        ambiguity_type=ambiguity_type,
    )


def run_calibration_suite(
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    logits_test: np.ndarray,
    labels_test: np.ndarray,
    model: str,
    dataset: str,
    ambiguity_type: str = "unambiguous",
    verbalized_probs_test: Optional[np.ndarray] = None,
    n_bins: int = 15,
    seed: Optional[int] = None,
) -> dict[str, CalibrationResult]:
    """Fit calibrators on the calibration split and evaluate on test split."""
    logits_cal = _as_1d_float(logits_cal, "logits_cal")
    labels_cal = _as_binary_labels(labels_cal, "labels_cal")
    logits_test = _as_1d_float(logits_test, "logits_test")
    labels_test = _as_binary_labels(labels_test, "labels_test")
    _validate_same_length(logits_cal, labels_cal)
    _validate_same_length(logits_test, labels_test)

    a, b = fit_platt(logits_cal, labels_cal)
    temperature = fit_temperature(logits_cal, labels_cal)

    results = {
        "raw": evaluate(
            expit(logits_test),
            labels_test,
            model,
            dataset,
            "raw",
            ambiguity_type,
            n_bins,
            seed=seed,
        ),
        "platt": evaluate(
            apply_platt(logits_test, a, b),
            labels_test,
            model,
            dataset,
            "platt",
            ambiguity_type,
            n_bins,
            seed=seed,
            platt_a=a,
            platt_b=b,
        ),
        "temperature": evaluate(
            expit(logits_test / temperature),
            labels_test,
            model,
            dataset,
            "temperature",
            ambiguity_type,
            n_bins,
            seed=seed,
            temperature=temperature,
        ),
    }

    if verbalized_probs_test is not None:
        verbalized_probs_test = _as_probabilities(verbalized_probs_test, "verbalized_probs_test")
        _validate_same_length(verbalized_probs_test, labels_test)
        results["verbalized"] = evaluate(
            verbalized_probs_test,
            labels_test,
            model,
            dataset,
            "verbalized",
            ambiguity_type,
            n_bins,
            seed=seed,
        )

    return results


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    arr = _as_1d_float(values, name)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return arr


def _as_binary_labels(values: np.ndarray, name: str) -> np.ndarray:
    arr = _as_1d_float(values, name)
    if np.any((arr != 0.0) & (arr != 1.0)):
        raise ValueError(f"{name} must contain binary labels 0/1")
    return arr


def _validate_same_length(left: np.ndarray, right: np.ndarray) -> None:
    if len(left) != len(right):
        raise ValueError(f"length mismatch: {len(left)} != {len(right)}")


def _validate_bins(n_bins: int) -> None:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
