"""Residual calibration analyses for ambiguous-dataset ECE.

These helpers turn calibration outputs into dataset-level summaries and test
whether per-item residual calibration error tracks the f_defer ambiguity probe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

from eval.calibration import apply_platt, fit_platt
from eval.calibration_analysis import (
    AMBIGUITY_TYPE,
    DATASETS,
    MODELS,
    SEEDS,
    load_signals,
    make_cal_test_split,
    resolve_signals_path,
    resolve_verbalized_path,
    load_verbalized_sidecar,
)


def per_dataset_ece(results_df: pd.DataFrame, method: str = "platt") -> pd.DataFrame:
    """Compute per-dataset ECE averaged across models and seeds."""
    subset = results_df[results_df["method"] == method].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["dataset", "ambiguity_type", "ece_mean", "ece_std", "brier_mean", "n_cells"]
        )

    return (
        subset.groupby(["dataset", "ambiguity_type"])
        .agg(
            ece_mean=("ece_15", "mean"),
            ece_std=("ece_15", "std"),
            brier_mean=("brier_score", "mean"),
            n_cells=("ece_15", "count"),
        )
        .reset_index()
        .sort_values(["ambiguity_type", "ece_mean"])
    )


def compute_per_item_residual(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-item calibration residual, |p_hat - y|."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if len(probs) != len(labels):
        raise ValueError(f"length mismatch: {len(probs)} != {len(labels)}")
    return np.abs(probs - labels)


def residual_vs_defer_correlation(
    signals_path: Path,
    model: str,
    dataset: str,
    probe_seed: int,
    split_seed: int = 42,
    verbalized_path: Optional[Path] = None,
) -> Optional[dict]:
    """Correlate Platt residuals with sigma_defer for one artifact."""
    signals = load_signals(signals_path)
    if signals is None:
        return None
    if verbalized_path is not None and "verbalized" not in signals:
        raw = torch.load(signals_path, map_location="cpu", weights_only=False)
        sidecar = load_verbalized_sidecar(signals_path, raw, verbalized_path)
        if sidecar is not None:
            signals["verbalized"] = sidecar
    if "sigma_defer" not in signals and "logit_defer" not in signals:
        return None

    cal, test = make_cal_test_split(signals, split_seed=split_seed)
    a, b = fit_platt(cal["logit_pred"], cal["y_correct"])
    probs_platt = apply_platt(test["logit_pred"], a, b)
    residuals = compute_per_item_residual(probs_platt, test["y_correct"])

    if "logit_defer" in test:
        sigma_defer = expit(test["logit_defer"])
    else:
        sigma_defer = np.asarray(test["sigma_defer"], dtype=float)

    if np.std(residuals) == 0.0 or np.std(sigma_defer) == 0.0:
        spearman_r = spearman_p = pearson_r = pearson_p = np.nan
    else:
        spearman_r, spearman_p = spearmanr(residuals, sigma_defer)
        pearson_r, pearson_p = pearsonr(residuals, sigma_defer)

    row = {
        "model": model,
        "dataset": dataset,
        "seed": probe_seed,
        "ambiguity_type": AMBIGUITY_TYPE.get(dataset, "unambiguous"),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_test": int(len(residuals)),
        "mean_residual": float(residuals.mean()),
        "mean_sigma_defer": float(sigma_defer.mean()),
        "has_sigma_defer": True,
    }

    if "verbalized" in test:
        verbalized_residuals = compute_per_item_residual(test["verbalized"], test["y_correct"])
        row["mean_verbalized_residual"] = float(verbalized_residuals.mean())
        row["mean_verbalized"] = float(np.mean(test["verbalized"]))

    return row


def run_residual_correlation_matrix(
    outputs_dir: Path,
    models: Sequence[str] = MODELS,
    datasets: Sequence[str] = DATASETS,
    seeds: Sequence[int] = SEEDS,
    split_seed: int = 42,
    verbalized_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run residual-vs-defer correlations over all model/dataset/seed cells."""
    rows = []
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                path = resolve_signals_path(outputs_dir, model, dataset, seed)
                verbalized_path = resolve_verbalized_path(verbalized_dir, model, dataset, seed)
                result = residual_vs_defer_correlation(
                    path,
                    model,
                    dataset,
                    probe_seed=seed,
                    split_seed=split_seed,
                    verbalized_path=verbalized_path,
                )
                if result is not None:
                    rows.append(result)
                else:
                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "ambiguity_type": AMBIGUITY_TYPE.get(dataset, "unambiguous"),
                            "spearman_r": np.nan,
                            "spearman_p": np.nan,
                            "pearson_r": np.nan,
                            "pearson_p": np.nan,
                            "n_test": 0,
                            "has_sigma_defer": False,
                        }
                    )

    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        print("\nResidual vs sigma_defer correlation by ambiguity type:")
        print("(positive = residual miscalibration tracks f_defer ambiguity)")
        print(summarize_residual_correlations(df).to_string(index=False))
    return df


def summarize_residual_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize residual correlation rows by ambiguity type."""
    valid = df[df["has_sigma_defer"]].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=["ambiguity_type", "spearman_r_mean", "spearman_r_std", "pearson_r_mean", "pearson_r_std", "n_cells"]
        )
    return (
        valid.groupby("ambiguity_type")
        .agg(
            spearman_r_mean=("spearman_r", "mean"),
            spearman_r_std=("spearman_r", "std"),
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            n_cells=("spearman_r", "count"),
        )
        .reset_index()
    )


def verbalized_vs_probe_comparison(results_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compare ECE of Platt-calibrated probe vs verbalized confidence."""
    if "verbalized" not in set(results_df["method"]):
        print(
            "[INFO] No verbalized baseline in results. Add sidecar scores or a "
            "'verbalized' key in signals.pt and rerun calibration."
        )
        return None

    methods = ["platt", "verbalized", "raw", "temperature"]
    comparison = (
        results_df[results_df["method"].isin(methods)]
        .groupby(["method", "ambiguity_type"])
        .agg(
            ece_mean=("ece_15", "mean"),
            ece_std=("ece_15", "std"),
            brier_mean=("brier_score", "mean"),
            n_cells=("ece_15", "count"),
        )
        .reset_index()
        .sort_values(["ambiguity_type", "ece_mean"])
    )

    platt = comparison[comparison["method"] == "platt"].set_index("ambiguity_type")["ece_mean"]
    verbalized = comparison[comparison["method"] == "verbalized"].set_index("ambiguity_type")[
        "ece_mean"
    ]
    gap = (verbalized - platt).rename("verbalized_minus_platt_ece").reset_index()
    comparison = comparison.merge(gap, on="ambiguity_type", how="left")
    return comparison


def add_verbalized_to_signals(signals_path: Path, verbalized_probs: np.ndarray) -> None:
    """Add row-aligned verbalized probabilities to an existing signals.pt file."""
    raw = torch.load(signals_path, map_location="cpu", weights_only=False)
    verbalized_probs = np.asarray(verbalized_probs, dtype=float).reshape(-1)
    n_existing = len(raw["logit_pred"])
    if len(verbalized_probs) != n_existing:
        raise ValueError(
            f"Length mismatch: signals has {n_existing}, verbalized has {len(verbalized_probs)}"
        )
    if np.any((verbalized_probs < 0.0) | (verbalized_probs > 1.0)):
        raise ValueError("verbalized probabilities must be in [0, 1]")
    raw["verbalized"] = torch.tensor(verbalized_probs, dtype=torch.float32)
    torch.save(raw, signals_path)
    print(f"Added 'verbalized' key to {signals_path}")


def format_paper_table(
    results_df: pd.DataFrame,
    methods: Sequence[str] = ("raw", "temperature", "platt", "verbalized"),
    ambiguity_types: Sequence[str] = ("unambiguous", "medical", "ambiguous"),
) -> pd.DataFrame:
    """Format ECE and Brier summaries as paper-ready strings."""
    rows = []
    for method in methods:
        if method not in set(results_df["method"]):
            continue
        subset = results_df[results_df["method"] == method]
        row = {"method": method}
        for ambiguity_type in ambiguity_types:
            cell = subset[subset["ambiguity_type"] == ambiguity_type]
            if cell.empty:
                row[f"ece_{ambiguity_type}"] = "-"
                row[f"brier_{ambiguity_type}"] = "-"
                continue
            row[f"ece_{ambiguity_type}"] = (
                f"{cell['ece_15'].mean():.3f} +/- {cell['ece_15'].std():.3f}"
            )
            row[f"brier_{ambiguity_type}"] = (
                f"{cell['brier_score'].mean():.3f} +/- {cell['brier_score'].std():.3f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)
