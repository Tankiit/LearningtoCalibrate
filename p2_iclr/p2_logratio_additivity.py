"""Exploratory log-ratio additivity check for the paired legend caches.

For each item and candidate side this fits

    log p_fwd(k) - log p_rev(k) = intercept_i + slope_i * (v_k - .5) + error.

Under f_i(v) = beta_i * v, the slope is 2*beta_i.  The intercept absorbs the
two normalizers.  Glyph and position terms cancel when they are unchanged
between the two legends, including item-dependent nuisance terms.

No pass/fail threshold is applied: the repository has no frozen
seed/paraphrase null for this diagnostic.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
DATASETS = ("truthfulqa", "pavlick_nli")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "letter11_provenance_v1"
DEFAULT_OUT = ROOT / "p2_iclr" / "cached_results" / "logratio_additivity.csv"


def additivity_check(
    p_fwd: np.ndarray,
    p_rev: np.ndarray,
    values: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Fit the requested per-item intercept/slope and return diagnostics."""
    p_fwd = np.asarray(p_fwd, dtype=float)
    p_rev = np.asarray(p_rev, dtype=float)
    values = np.asarray(values, dtype=float)
    if p_fwd.shape != p_rev.shape or p_fwd.ndim != 2:
        raise ValueError("p_fwd and p_rev must have the same [n_items, K] shape")
    if values.shape != (p_fwd.shape[1],):
        raise ValueError("values must have one entry per glyph column")
    if np.any(p_fwd < 0) or np.any(p_rev < 0):
        raise ValueError("probabilities must be nonnegative")

    L = np.log(np.maximum(p_fwd, eps)) - np.log(np.maximum(p_rev, eps))
    x = values - 0.5
    X = np.column_stack([np.ones_like(x), x])
    pinv = np.linalg.pinv(X)
    coef = L @ pinv.T
    fitted = coef @ X.T
    residual = L - fitted
    dof = max(len(x) - 2, 1)
    return {
        "intercept": coef[:, 0],
        "slope": coef[:, 1],
        "beta_hat": coef[:, 1] / 2.0,
        "residual_variance": np.sum(residual * residual, axis=1) / dof,
        "residual_rms": np.sqrt(np.mean(residual * residual, axis=1)),
        "log_ratio": L,
    }


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            cell = args.cache_root / model / dataset
            fwd, rev = _load(cell / "forward.pt"), _load(cell / "reversed.pt")
            fids = np.asarray(fwd["example_ids"]).astype(str)
            rids = np.asarray(rev["example_ids"]).astype(str)
            if not np.array_equal(fids, rids):
                raise ValueError(f"forward/reversed IDs differ: {model}/{dataset}")
            values = np.asarray(fwd["conf_values"], dtype=float) / 100.0
            if not np.allclose(
                np.asarray(rev["conf_values"], dtype=float) / 100.0,
                1.0 - values,
                atol=1e-12,
            ):
                raise ValueError(f"reversed values are not 1-values: {model}/{dataset}")

            side_results = []
            for side in ("pos", "neg"):
                result = additivity_check(
                    fwd[f"Vdist_{side}"], rev[f"Vdist_{side}"], values
                )
                side_results.append(result)
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "side": side,
                    "n_items": len(fids),
                    "mean_slope": float(np.mean(result["slope"])),
                    "sd_slope": float(np.std(result["slope"], ddof=1)),
                    "mean_beta_hat": float(np.mean(result["beta_hat"])),
                    "mean_residual_variance": float(np.mean(result["residual_variance"])),
                    "median_residual_variance": float(np.median(result["residual_variance"])),
                    "mean_residual_rms": float(np.mean(result["residual_rms"])),
                    "median_residual_rms": float(np.median(result["residual_rms"])),
                    "null_available": False,
                    "verdict": "descriptive; no frozen null",
                })

            pooled = np.concatenate([x["residual_variance"] for x in side_results])
            rows.append({
                "model": model,
                "dataset": dataset,
                "side": "pooled_pos_neg",
                "n_items": len(pooled),
                "mean_slope": "",
                "sd_slope": "",
                "mean_beta_hat": "",
                "mean_residual_variance": float(np.mean(pooled)),
                "median_residual_variance": float(np.median(pooled)),
                "mean_residual_rms": "",
                "median_residual_rms": "",
                "null_available": False,
                "verdict": "descriptive; no frozen null",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
