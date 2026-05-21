"""Diagnose whether beta can drive adaptive credal widths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.calibration import apply_platt, fit_platt  # noqa: E402
from eval.calibration_analysis import (  # noqa: E402
    AMBIGUITY_TYPE,
    DATASETS,
    MODELS,
    SEEDS,
    load_signals,
    make_cal_test_split,
    resolve_signals_path,
)
from gap.credal import label_kl_scores, transform_uncertainty  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run beta/nonconformity diagnostic.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--transform", choices=["abs", "neg", "pos", "raw"], default="neg")
    parser.add_argument("--out", type=Path, default=Path("results/credal_sets/beta_diagnostic.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                path = resolve_signals_path(args.outputs_dir, model, dataset, seed)
                signals = load_signals(path)
                if signals is None or "beta" not in signals:
                    rows.append(missing_row(model, dataset, seed))
                    continue
                if "beta_valid" in signals:
                    valid = np.asarray(signals["beta_valid"], dtype=bool)
                    if valid.sum() < 40:
                        rows.append(missing_row(model, dataset, seed))
                        continue
                    n = len(signals["logit_pred"])
                    signals = {
                        key: value[valid] if len(value) == n else value
                        for key, value in signals.items()
                    }
                cal, test = make_cal_test_split(signals, split_seed=args.split_seed)
                a, b = fit_platt(cal["logit_pred"], cal["y_correct"])
                p0 = apply_platt(test["logit_pred"], a, b)
                nonconformity = label_kl_scores(p0, test["y_correct"])
                uncertainty = transform_uncertainty(test["beta"], args.transform)
                if np.std(uncertainty) == 0.0 or np.std(nonconformity) == 0.0:
                    rho = p_value = np.nan
                else:
                    res = spearmanr(uncertainty, nonconformity)
                    rho, p_value = float(res.statistic), float(res.pvalue)
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "ambiguity_type": AMBIGUITY_TYPE.get(dataset, "unambiguous"),
                        "has_beta": True,
                        "transform": args.transform,
                        "spearman_r": rho,
                        "spearman_p": p_value,
                        "n_test": len(nonconformity),
                        "mean_beta": float(np.mean(test["beta"])),
                        "mean_uncertainty": float(np.mean(uncertainty)),
                        "mean_nonconformity": float(np.mean(nonconformity)),
                    }
                )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

    valid = df[df["has_beta"]]
    if valid.empty:
        print("No beta keys found in signals.pt.")
        return
    summary = (
        valid.groupby("ambiguity_type")
        .agg(spearman_r_mean=("spearman_r", "mean"), spearman_r_std=("spearman_r", "std"), n_cells=("spearman_r", "count"))
        .reset_index()
    )
    print(summary.to_string(index=False))


def missing_row(model: str, dataset: str, seed: int) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "ambiguity_type": AMBIGUITY_TYPE.get(dataset, "unambiguous"),
        "has_beta": False,
        "transform": "",
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "n_test": 0,
        "mean_beta": np.nan,
        "mean_uncertainty": np.nan,
        "mean_nonconformity": np.nan,
    }


if __name__ == "__main__":
    main()
