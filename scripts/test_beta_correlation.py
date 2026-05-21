"""Proxy test for beta-adaptive credal-set viability.

Before spending Modal compute on base-vs-instruct beta extraction, this script
checks whether already-available uncertainty proxies predict conformal
nonconformity scores on the same train/test split used by the credal pipeline.

Decision rule:
  PROCEED if any proxy has mean Spearman r > 0.10 on ambiguous datasets.
  WEAK    if the best ambiguous mean r is in (0, 0.10].
  STOP    if all ambiguous proxy means are <= 0 or no ambiguous cells are found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
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
from gap.credal import label_kl_scores  # noqa: E402


MODEL_ALIASES = {
    "ll": "llama3_8b",
    "llama": "llama3_8b",
    "mistral": "mistral_7b",
    "qwen": "qwen2_5_7b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test beta proxy correlations.")
    parser.add_argument("--signals-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--logprobs-dir", type=Path, default=Path("outputs/step1_extract"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--per-cell", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/credal_sets/beta_proxy_correlation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [MODEL_ALIASES.get(model, model) for model in args.models]
    rows = []

    for model in models:
        for dataset in args.datasets:
            for seed in args.seeds:
                signals_path = resolve_signals_path(args.signals_dir, model, dataset, seed)
                signals = load_signals(signals_path)
                if signals is None:
                    continue
                add_logprob_proxy(signals, args.logprobs_dir, model, dataset)
                rows.extend(evaluate_cell(signals, model, dataset, seed, args.split_seed))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

    if df.empty:
        print("STOP: no evaluable cells.")
        return

    if args.per_cell:
        print("\nPer-cell correlations:")
        print(
            df.sort_values(["dataset", "model", "seed", "proxy"])[
                ["model", "dataset", "seed", "ambiguity_type", "proxy", "spearman_r", "spearman_p", "n_test"]
            ].to_string(index=False)
        )

    summary = (
        df.groupby(["ambiguity_type", "proxy"])
        .agg(
            spearman_r_mean=("spearman_r", "mean"),
            spearman_r_std=("spearman_r", "std"),
            n_cells=("spearman_r", "count"),
        )
        .reset_index()
        .sort_values(["ambiguity_type", "spearman_r_mean"], ascending=[True, False])
    )
    print("\nSummary by ambiguity type:")
    print(summary.to_string(index=False))

    ambiguous = summary[summary["ambiguity_type"] == "ambiguous"]
    if ambiguous.empty:
        print("\nSTOP: no ambiguous cells in this run.")
        return
    best = ambiguous.sort_values("spearman_r_mean", ascending=False).iloc[0]
    best_r = float(best["spearman_r_mean"])
    best_proxy = str(best["proxy"])
    if best_r > 0.10:
        decision = "PROCEED"
    elif best_r > 0.0:
        decision = "WEAK"
    else:
        decision = "STOP"
    print(
        f"\nDecision: {decision} "
        f"(best ambiguous proxy={best_proxy}, mean Spearman r={best_r:.3f})"
    )


def evaluate_cell(
    signals: dict[str, np.ndarray],
    model: str,
    dataset: str,
    seed: int,
    split_seed: int,
) -> list[dict]:
    cal, test = make_cal_test_split(signals, split_seed=split_seed)
    a, b = fit_platt(cal["logit_pred"], cal["y_correct"])
    p0 = apply_platt(test["logit_pred"], a, b)
    nonconformity = label_kl_scores(p0, test["y_correct"])

    proxies = {
        "sigma_defer": test.get("sigma_defer"),
        "logit_defer": test.get("logit_defer"),
        "neg_gap": -test["gap"] if "gap" in test else None,
        "neg_logprob": -test["logprob"] if "logprob" in test else None,
    }

    rows = []
    for proxy_name, values in proxies.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(nonconformity) or np.std(values) == 0.0 or np.std(nonconformity) == 0.0:
            rho = p_value = np.nan
        else:
            res = spearmanr(values, nonconformity)
            rho, p_value = float(res.statistic), float(res.pvalue)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "ambiguity_type": AMBIGUITY_TYPE.get(dataset, "unambiguous"),
                "proxy": proxy_name,
                "spearman_r": rho,
                "spearman_p": p_value,
                "n_test": len(nonconformity),
                "mean_nonconformity": float(np.mean(nonconformity)),
                "mean_proxy": float(np.mean(values)),
            }
        )
    return rows


def add_logprob_proxy(
    signals: dict[str, np.ndarray],
    logprobs_dir: Path,
    model: str,
    dataset: str,
) -> None:
    path = logprobs_dir / model / dataset / "logprobs.pt"
    if not path.exists() or "logprob" in signals:
        return
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not {"example_ids", "logprob_pos", "logprob_neg"}.issubset(blob):
        return

    ids = np.asarray(blob["example_ids"], dtype=str)
    pos = np.asarray(blob["logprob_pos"], dtype=float)
    neg = np.asarray(blob["logprob_neg"], dtype=float)
    pos_by_id = dict(zip(ids, pos))
    neg_by_id = dict(zip(ids, neg))

    signal_ids = np.asarray(signals.get("example_ids", []), dtype=str)
    if len(signal_ids) == 0:
        return
    is_pos = np.asarray(signals["y_correct"], dtype=bool)
    aligned = np.empty(len(signal_ids), dtype=float)
    try:
        for i, (example_id, positive) in enumerate(zip(signal_ids, is_pos)):
            aligned[i] = pos_by_id[str(example_id)] if positive else neg_by_id[str(example_id)]
    except KeyError:
        return
    signals["logprob"] = aligned


if __name__ == "__main__":
    main()
