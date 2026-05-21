"""Coverage-width evaluation for conformal credal intervals."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eval.calibration import apply_platt, fit_platt
from eval.calibration_analysis import (
    AMBIGUITY_TYPE,
    DATASETS,
    MODELS,
    SEEDS,
    load_signals,
    make_cal_test_split,
    resolve_signals_path,
)
from gap.credal import (
    FixedRadiusCredalCalibrator,
    IsotonicCredalCalibrator,
    evaluate_intervals,
    label_kl_scores,
    transform_uncertainty,
)


ALPHAS = [0.20, 0.10, 0.05]


def evaluate_credal_cell(
    signals_path: Path,
    model: str,
    dataset: str,
    seed: int,
    alphas: Sequence[float] = ALPHAS,
    split_seed: int = 42,
    beta_transform: str = "abs",
    include_proxies: bool = True,
) -> list[dict]:
    """Evaluate fixed, beta-adaptive, and entropy-adaptive credal intervals."""
    signals = load_signals(signals_path)
    if signals is None:
        return []
    if "beta" in signals and "beta_valid" in signals:
        signals = _filter_beta_valid(signals)
        if signals is None:
            return []

    cal, test = make_cal_test_split(signals, split_seed=split_seed)
    a, b = fit_platt(cal["logit_pred"], cal["y_correct"])
    p0_cal = apply_platt(cal["logit_pred"], a, b)
    p0_test = apply_platt(test["logit_pred"], a, b)

    rows: list[dict] = []
    nonconformity_test = label_kl_scores(p0_test, test["y_correct"])
    for alpha in alphas:
        fixed = FixedRadiusCredalCalibrator(alpha=alpha)
        fixed.fit(p0_cal, cal["y_correct"])
        rows.append(
            _row(
                fixed.intervals(p0_test),
                test["y_correct"],
                model,
                dataset,
                seed,
                alpha,
                method="fixed",
                extra={"has_beta": "beta" in signals, "has_semantic_entropy": _entropy_key(signals) is not None},
            )
        )

        if "beta" in signals:
            beta = IsotonicCredalCalibrator(alpha=alpha, transform=beta_transform)
            beta.fit(p0_cal, cal["y_correct"], cal["beta"])
            beta_intervals = beta.intervals(p0_test, test["beta"])
            rows.append(
                _row(
                    beta_intervals,
                    test["y_correct"],
                    model,
                    dataset,
                    seed,
                    alpha,
                    method=f"beta_{beta_transform}",
                    extra={
                        "spearman_uncertainty_nonconformity": _safe_spearman(
                            transform_uncertainty(test["beta"], beta_transform),
                            nonconformity_test,
                        ),
                        "has_beta": True,
                        "has_semantic_entropy": _entropy_key(signals) is not None,
                    },
                )
            )

        if include_proxies and "gap" in signals:
            neg_gap = -cal["gap"]
            neg_gap_test = -test["gap"]
            gap_calibrator = IsotonicCredalCalibrator(alpha=alpha, transform="raw")
            gap_calibrator.fit(p0_cal, cal["y_correct"], neg_gap)
            rows.append(
                _row(
                    gap_calibrator.intervals(p0_test, neg_gap_test),
                    test["y_correct"],
                    model,
                    dataset,
                    seed,
                    alpha,
                    method="neg_gap",
                    extra={
                        "spearman_uncertainty_nonconformity": _safe_spearman(
                            neg_gap_test,
                            nonconformity_test,
                        ),
                        "has_beta": "beta" in signals,
                        "has_semantic_entropy": _entropy_key(signals) is not None,
                    },
                )
            )

        if include_proxies and "sigma_defer" in signals:
            defer_calibrator = IsotonicCredalCalibrator(alpha=alpha, transform="raw")
            defer_calibrator.fit(p0_cal, cal["y_correct"], cal["sigma_defer"])
            rows.append(
                _row(
                    defer_calibrator.intervals(p0_test, test["sigma_defer"]),
                    test["y_correct"],
                    model,
                    dataset,
                    seed,
                    alpha,
                    method="sigma_defer",
                    extra={
                        "spearman_uncertainty_nonconformity": _safe_spearman(
                            test["sigma_defer"],
                            nonconformity_test,
                        ),
                        "has_beta": "beta" in signals,
                        "has_semantic_entropy": _entropy_key(signals) is not None,
                    },
                )
            )

        entropy_key = _entropy_key(signals)
        if entropy_key is not None:
            entropy = IsotonicCredalCalibrator(alpha=alpha, transform="raw")
            entropy.fit(p0_cal, cal["y_correct"], cal[entropy_key])
            rows.append(
                _row(
                    entropy.intervals(p0_test, test[entropy_key]),
                    test["y_correct"],
                    model,
                    dataset,
                    seed,
                    alpha,
                    method="semantic_entropy",
                    extra={
                        "spearman_uncertainty_nonconformity": _safe_spearman(
                            test[entropy_key],
                            nonconformity_test,
                        ),
                        "has_beta": "beta" in signals,
                        "has_semantic_entropy": True,
                    },
                )
            )

    return rows


def run_credal_matrix(
    outputs_dir: Path,
    models: Sequence[str] = MODELS,
    datasets: Sequence[str] = DATASETS,
    seeds: Sequence[int] = SEEDS,
    alphas: Sequence[float] = ALPHAS,
    split_seed: int = 42,
    beta_transform: str = "abs",
    include_proxies: bool = True,
) -> pd.DataFrame:
    """Run credal coverage-width evaluation over the full artifact matrix."""
    rows: list[dict] = []
    skipped = 0
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                path = resolve_signals_path(outputs_dir, model, dataset, seed)
                cell_rows = evaluate_credal_cell(
                    path,
                    model,
                    dataset,
                    seed,
                    alphas=alphas,
                    split_seed=split_seed,
                    beta_transform=beta_transform,
                    include_proxies=include_proxies,
                )
                if not cell_rows:
                    skipped += 1
                    continue
                rows.extend(cell_rows)
    df = pd.DataFrame(rows)
    if skipped:
        print(f"[INFO] skipped {skipped} cells with missing/unreadable signals.")
    return df


def summarize_coverage_width(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage and width by method, alpha, and ambiguity type."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["method", "alpha", "target_coverage", "ambiguity_type"])
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            mean_width=("mean_width", "mean"),
            median_width=("median_width", "mean"),
            mean_radius=("mean_radius", "mean"),
            n_cells=("coverage", "count"),
        )
        .reset_index()
        .sort_values(["alpha", "ambiguity_type", "mean_width", "method"])
    )


def width_reduction_vs_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute adaptive width reduction relative to fixed radii per cell."""
    if df.empty or "fixed" not in set(df["method"]):
        return pd.DataFrame()
    fixed = df[df["method"] == "fixed"][
        ["model", "dataset", "seed", "alpha", "mean_width"]
    ].rename(columns={"mean_width": "fixed_mean_width"})
    adaptive = df[df["method"] != "fixed"][
        ["model", "dataset", "seed", "alpha", "method", "ambiguity_type", "mean_width"]
    ]
    merged = adaptive.merge(fixed, on=["model", "dataset", "seed", "alpha"], how="inner")
    merged["width_reduction_abs"] = merged["fixed_mean_width"] - merged["mean_width"]
    summary = (
        merged.groupby(["method", "alpha", "ambiguity_type"])
        .agg(
            width_reduction_abs_mean=("width_reduction_abs", "mean"),
            fixed_mean_width=("fixed_mean_width", "mean"),
            adaptive_mean_width=("mean_width", "mean"),
            n_cells=("width_reduction_abs", "count"),
        )
        .reset_index()
        .sort_values(["alpha", "ambiguity_type", "method"])
    )
    summary["width_reduction_rel_mean"] = (
        summary["fixed_mean_width"] - summary["adaptive_mean_width"]
    ) / summary["fixed_mean_width"].clip(1e-12)
    return summary


def adaptivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level mean width, used to show wider sets on ambiguous data."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["method", "alpha", "dataset", "ambiguity_type"])
        .agg(mean_width=("mean_width", "mean"), coverage=("coverage", "mean"), n_cells=("n", "count"))
        .reset_index()
        .sort_values(["method", "alpha", "mean_width"])
    )


def _row(
    intervals,
    labels: np.ndarray,
    model: str,
    dataset: str,
    seed: int,
    alpha: float,
    method: str,
    extra: Optional[dict] = None,
) -> dict:
    row = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "ambiguity_type": AMBIGUITY_TYPE.get(dataset, "unambiguous"),
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "method": method,
        **evaluate_intervals(intervals, labels),
    }
    if extra:
        row.update(extra)
    return row


def _entropy_key(signals: dict[str, np.ndarray]) -> Optional[str]:
    for key in ("semantic_entropy", "sem_entropy", "entropy"):
        if key in signals:
            return key
    return None


def _filter_beta_valid(signals: dict[str, np.ndarray], min_rows: int = 40) -> Optional[dict[str, np.ndarray]]:
    """Restrict beta-enabled cells to rows with real beta values."""
    valid = np.asarray(signals["beta_valid"], dtype=bool)
    if valid.sum() < min_rows:
        return None
    n = len(signals["logit_pred"])
    filtered: dict[str, np.ndarray] = {}
    for key, value in signals.items():
        if len(value) == n:
            filtered[key] = value[valid]
        else:
            filtered[key] = value
    return filtered


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
