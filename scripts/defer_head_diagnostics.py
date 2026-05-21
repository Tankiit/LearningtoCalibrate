"""
Diagnose whether f_defer learned y_expert, and whether it is just tracking
model correctness.

For each model/dataset/seed cell, this reports:
  - accuracy of sigma(f_defer) against y_expert on the evaluation rows
  - point-biserial correlation between sigma(f_defer) and y_correct

The evaluation rows match step4_evaluate.py: held-out h+ test rows plus all h-
rows. y_expert is reconstructed using the same halueval_style_labels convention
as probe training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pointbiserialr

from data.registry import load_dataset
from gap.signal import load_signals
from utils.config import load_config
from utils.paths import conformal_path, hidden_states_path, signals_path


DEFAULT_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DEFAULT_DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "pubmedqa",
    "medqa",
    "halueval_qa",
    "popqa",
    "triviaqa",
]
DEFAULT_SEEDS = [0, 1, 2]


def _halueval_style_labels(model: str, dataset: str) -> bool:
    exp_cfg = _PROJECT_ROOT / "configs" / "experiments" / f"{model}_{dataset}.yaml"
    ds_cfg = _PROJECT_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    path = exp_cfg if exp_cfg.exists() else ds_cfg
    if not path.exists():
        return False
    cfg = load_config(path)
    return bool(cfg.get("halueval_style_labels", False))


def _expert_reliable(model: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    h_path = hidden_states_path(model, dataset)
    blob = torch.load(h_path, map_location="cpu", weights_only=False)
    example_ids = np.asarray(blob["example_ids"])

    if "expert_reliable" in blob:
        expert_reliable = np.asarray(blob["expert_reliable"], dtype=bool)
    else:
        records = load_dataset(dataset)
        er_by_id = {r.example_id: r.expert_reliable for r in records}
        expert_reliable = np.asarray([er_by_id[eid] for eid in example_ids], dtype=bool)

    if len(expert_reliable) != len(example_ids):
        raise ValueError(
            f"{model}/{dataset}: expert_reliable length mismatch "
            f"{len(expert_reliable)} vs {len(example_ids)}"
        )
    return example_ids, expert_reliable


def _y_expert_for_signal_rows(model: str, dataset: str, signal_ids: np.ndarray) -> np.ndarray:
    example_ids, expert_reliable = _expert_reliable(model, dataset)
    er_by_id = dict(zip(example_ids, expert_reliable.astype(int)))
    n = len(example_ids)
    halueval_style = _halueval_style_labels(model, dataset)

    y_pos = expert_reliable.astype(int)
    y_neg = np.ones(n, dtype=int) if halueval_style else expert_reliable.astype(int)
    y_expert = np.concatenate([y_pos, y_neg])

    expected_ids = np.concatenate([example_ids, example_ids])
    if len(signal_ids) == len(expected_ids) and np.array_equal(signal_ids, expected_ids):
        return y_expert

    # Fallback for older signal files whose row order still follows h+ then h-.
    half = len(signal_ids) // 2
    is_neg_row = np.arange(len(signal_ids)) >= half
    out = np.empty(len(signal_ids), dtype=int)
    for i, eid in enumerate(signal_ids):
        if halueval_style and is_neg_row[i]:
            out[i] = 1
        else:
            out[i] = int(er_by_id[eid])
    return out


def diagnose_cell(model: str, dataset: str, seed: int) -> dict | None:
    sp = signals_path(model, dataset, seed)
    cp = conformal_path(model, dataset, seed)
    if not sp.exists() or not cp.exists():
        return None

    sig = load_signals(sp)
    cal_blob = torch.load(cp, map_location="cpu", weights_only=False)
    test_pos_idx = np.asarray(cal_blob["test_pos_idx"])
    neg_idx = np.where(~sig.is_pos)[0]
    eval_idx = np.concatenate([test_pos_idx, neg_idx])

    y_correct = np.concatenate([
        np.ones(len(test_pos_idx), dtype=int),
        np.zeros(len(neg_idx), dtype=int),
    ])
    y_expert_all = _y_expert_for_signal_rows(model, dataset, sig.example_ids)
    y_expert_eval = y_expert_all[eval_idx].astype(int)
    p_defer_eval = sig.sigma_defer[eval_idx]

    acc_defer_on_yexpert = ((p_defer_eval > 0.5).astype(int) == y_expert_eval).mean()
    rho_defer_with_ymodel, p_defer_with_ymodel = pointbiserialr(y_correct, p_defer_eval)
    rho_error_with_yexpert, p_error_with_yexpert = pointbiserialr(1 - y_correct, y_expert_eval)

    return {
        "model_key": model,
        "dataset": dataset,
        "seed": seed,
        "acc_defer_on_yexpert": float(acc_defer_on_yexpert),
        "rho_defer_with_ycorrect": float(rho_defer_with_ymodel),
        "p_defer_with_ycorrect": float(p_defer_with_ymodel),
        "rho_error_with_yexpert": float(rho_error_with_yexpert),
        "p_error_with_yexpert": float(p_error_with_yexpert),
        "n_correct": int((y_correct == 1).sum()),
        "y_expert_rate": float(y_expert_eval.mean()),
        "p_defer_mean": float(p_defer_eval.mean()),
        "n": int(len(eval_idx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/defer_head_diagnostics.parquet"),
    )
    args = parser.parse_args()

    rows = []
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                row = diagnose_cell(model, dataset, seed)
                if row is not None:
                    rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    summary = df.groupby(["model_key", "dataset"]).agg(
        acc_defer_on_yexpert=("acc_defer_on_yexpert", "mean"),
        rho_defer_with_ycorrect=("rho_defer_with_ycorrect", "mean"),
        p_defer_with_ycorrect_max=("p_defer_with_ycorrect", "max"),
        rho_error_with_yexpert=("rho_error_with_yexpert", "mean"),
        p_error_with_yexpert_max=("p_error_with_yexpert", "max"),
        n_correct=("n_correct", "first"),
        y_expert_rate=("y_expert_rate", "mean"),
        p_defer_mean=("p_defer_mean", "mean"),
        n=("n", "first"),
    ).reset_index()
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
