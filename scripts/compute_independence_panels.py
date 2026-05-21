"""
Compute independence statistics across all available (model, dataset) cells.

Outputs under data/independence_panels/:
  - cosines_by_cell.dat       geometric independence: |cos(w_pred, w_defer)|
  - stc_by_cell.dat           supervision-target correlation: |rho(y_expert, model_error)|
  - signal_corr_by_cell.dat   per-item output correlation: |rho(p_pred, p_defer)|
  - scatter_{dataset}_{model}.dat
  - independence_summary.csv

Also writes results/aggregated_probes.parquet as a reusable intermediate with
probe weights, evaluation predictions, and aligned targets.
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
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, pointbiserialr

from gap.signal import load_signals
from scripts.defer_head_diagnostics import _y_expert_for_signal_rows
from utils.paths import conformal_path, ova_heads_path, signals_path


MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
NON_DEGENERATE = ["chaosnli", "pavlick_nli", "pubmedqa", "medqa", "popqa"]
DEGENERATE = ["ambigqa_kge2", "halueval_qa", "triviaqa"]
PER_CATEGORY = ["truthfulqa"]

DATASET_DISPLAY = {
    "chaosnli": "ChNLI",
    "pavlick_nli": "PavNLI",
    "ambigqa_kge2": "AmbigQA",
    "pubmedqa": "PubMed",
    "medqa": "MedQA",
    "popqa": "PopQA",
    "truthfulqa": "TfQA",
    "halueval_qa": "HaluEval",
    "triviaqa": "TriviaQA",
}

MODEL_DISPLAY = {
    "llama3_8b": "LLaMA",
    "mistral_7b": "Mistral",
    "qwen2_5_7b": "Qwen",
}

OUTPUT_DIR = Path("data/independence_panels")
AGG_PROBES_PATH = Path("results/aggregated_probes.parquet")


def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.var(x) == 0 or np.var(y) == 0:
        return float("nan")
    rho, _ = pearsonr(x, y)
    return float(abs(rho))


def _safe_abs_pointbiserial(binary: np.ndarray, value: np.ndarray) -> float:
    if len(binary) < 2 or np.var(binary) == 0 or np.var(value) == 0:
        return float("nan")
    rho, _ = pointbiserialr(binary, value)
    return float(abs(rho))


def _row_keys(sig, eval_idx: np.ndarray) -> list[str]:
    keys = []
    for idx in eval_idx:
        side = "pos" if sig.is_pos[idx] else "neg"
        keys.append(f"{side}:{sig.example_ids[idx]}")
    return keys


def _load_seed_row(model: str, dataset: str, seed: int) -> dict | None:
    heads_path = ova_heads_path(model, dataset, seed)
    sig_path = signals_path(model, dataset, seed)
    cal_path = conformal_path(model, dataset, seed)
    if not heads_path.exists() or not sig_path.exists() or not cal_path.exists():
        return None

    heads = torch.load(heads_path, map_location="cpu", weights_only=False)
    state = heads["state_dict"]
    w_pred = state["pred.weight"].detach().cpu().numpy().reshape(-1)
    w_defer = state["defer.weight"].detach().cpu().numpy().reshape(-1)

    sig = load_signals(sig_path)
    cal_blob = torch.load(cal_path, map_location="cpu", weights_only=False)
    test_pos_idx = np.asarray(cal_blob["test_pos_idx"])
    neg_idx = np.where(~sig.is_pos)[0]
    eval_idx = np.concatenate([test_pos_idx, neg_idx])

    y_correct = np.concatenate([
        np.ones(len(test_pos_idx), dtype=int),
        np.zeros(len(neg_idx), dtype=int),
    ])
    y_expert_all = _y_expert_for_signal_rows(model, dataset, sig.example_ids)
    y_expert = y_expert_all[eval_idx].astype(int)

    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "w_pred": w_pred,
        "w_defer": w_defer,
        "row_keys": np.asarray(_row_keys(sig, eval_idx), dtype=object),
        "p_pred_eval": sig.sigma_pred[eval_idx],
        "p_defer_eval": sig.sigma_defer[eval_idx],
        "y_correct": y_correct,
        "y_expert": y_expert,
    }


def build_aggregated_probes(models: list[str], datasets: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                row = _load_seed_row(model, dataset, seed)
                if row is not None:
                    rows.append(row)
    df = pd.DataFrame(rows)
    AGG_PROBES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AGG_PROBES_PATH, index=False)
    return df


def _average_eval_arrays(rows: pd.DataFrame) -> dict:
    buckets: dict[str, dict[str, list[float] | int]] = {}
    for row in rows.itertuples(index=False):
        for key, pp, pdv, yc, ye in zip(
            row.row_keys,
            row.p_pred_eval,
            row.p_defer_eval,
            row.y_correct,
            row.y_expert,
        ):
            bucket = buckets.setdefault(
                str(key),
                {"p_pred": [], "p_defer": [], "y_correct": int(yc), "y_expert": int(ye)},
            )
            bucket["p_pred"].append(float(pp))
            bucket["p_defer"].append(float(pdv))

    keys = sorted(buckets)
    return {
        "row_keys": np.asarray(keys, dtype=object),
        "p_pred_eval": np.asarray([np.mean(buckets[k]["p_pred"]) for k in keys]),
        "p_defer_eval": np.asarray([np.mean(buckets[k]["p_defer"]) for k in keys]),
        "y_correct": np.asarray([buckets[k]["y_correct"] for k in keys], dtype=int),
        "y_expert": np.asarray([buckets[k]["y_expert"] for k in keys], dtype=int),
    }


def load_cell(model: str, dataset: str, df: pd.DataFrame) -> dict | None:
    rows = df[(df.model == model) & (df.dataset == dataset)]
    if len(rows) == 0:
        return None

    w_pred = np.stack(rows.w_pred.values).mean(axis=0)
    w_defer = np.stack(rows.w_defer.values).mean(axis=0)
    eval_arrays = _average_eval_arrays(rows)

    return {
        "w_pred": w_pred,
        "w_defer": w_defer,
        **eval_arrays,
    }


def geometric_cosine(cell: dict) -> float:
    """|cos(w_pred, w_defer)|: geometric non-independence."""
    if np.linalg.norm(cell["w_pred"]) == 0 or np.linalg.norm(cell["w_defer"]) == 0:
        return float("nan")
    return float(abs(1 - cosine(cell["w_pred"], cell["w_defer"])))


def supervision_target_correlation(cell: dict) -> float:
    """|rho(y_expert, 1 - y_correct)|: supervision-target correlation."""
    return _safe_abs_pointbiserial(cell["y_expert"], 1 - cell["y_correct"])


def signal_output_correlation(cell: dict) -> float:
    """|rho(p_pred, p_defer)|: per-item signal-output correlation."""
    return _safe_abs_pearson(cell["p_pred_eval"], cell["p_defer_eval"])


def _format_value(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "nan"


def write_bar_chart_data(metric_fn, datasets: list[str], filename: str, df: pd.DataFrame) -> None:
    """Write pgfplots .dat file: dataset model_1_value model_2_value model_3_value."""
    lines = ["dataset " + " ".join(MODEL_DISPLAY[m] for m in MODELS)]
    for dataset in datasets:
        row = [DATASET_DISPLAY.get(dataset, dataset)]
        for model in MODELS:
            cell = load_cell(model, dataset, df)
            row.append("nan" if cell is None else _format_value(metric_fn(cell)))
        lines.append(" ".join(row))

    (OUTPUT_DIR / filename).write_text("\n".join(lines) + "\n")
    print(f"  wrote {filename}")


def write_scatter_data(model: str, dataset: str, df: pd.DataFrame, max_points: int = 2000) -> None:
    """Write p_pred/p_defer scatter coordinates for one model/dataset cell."""
    cell = load_cell(model, dataset, df)
    if cell is None:
        print(f"  WARN: no data for {model}/{dataset}")
        return

    n = len(cell["p_pred_eval"])
    rng = np.random.default_rng(0)
    idx = rng.choice(n, max_points, replace=False) if n > max_points else np.arange(n)

    lines = ["p_pred p_defer y_correct y_expert"]
    for i in idx:
        lines.append(
            f"{cell['p_pred_eval'][i]:.6f} "
            f"{cell['p_defer_eval'][i]:.6f} "
            f"{int(cell['y_correct'][i])} "
            f"{int(cell['y_expert'][i])}"
        )

    fname = f"scatter_{dataset}_{model}.dat"
    (OUTPUT_DIR / fname).write_text("\n".join(lines) + "\n")
    print(f"  wrote {fname} ({len(idx)} points)")


def write_summary(df: pd.DataFrame, datasets: list[str]) -> None:
    rows = []
    for dataset in datasets:
        for model in MODELS:
            cell = load_cell(model, dataset, df)
            if cell is None:
                continue
            rows.append({
                "model": model,
                "dataset": dataset,
                "cos_w_pred_w_defer": geometric_cosine(cell),
                "rho_ST": supervision_target_correlation(cell),
                "rho_f_pred_f_defer": signal_output_correlation(cell),
                "n_eval_rows_after_seed_union": len(cell["y_correct"]),
                "y_correct_rate": float(np.mean(cell["y_correct"])),
                "y_expert_rate": float(np.mean(cell["y_expert"])),
            })
    out = OUTPUT_DIR / "independence_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote independence_summary.csv ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=NON_DEGENERATE + PER_CATEGORY + DEGENERATE,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-points", type=int, default=1500)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating independence panel data files...")
    df = build_aggregated_probes(args.models, args.datasets, args.seeds)
    print(f"  wrote {AGG_PROBES_PATH} ({len(df)} seed rows)")

    print("\n[1/5] Geometric cosines")
    write_bar_chart_data(geometric_cosine, args.datasets, "cosines_by_cell.dat", df)

    print("\n[2/5] Supervision-target correlation")
    write_bar_chart_data(supervision_target_correlation, args.datasets, "stc_by_cell.dat", df)

    print("\n[3/5] Signal-output correlation")
    write_bar_chart_data(signal_output_correlation, args.datasets, "signal_corr_by_cell.dat", df)

    print("\n[4/5] Per-item scatter data")
    write_scatter_data("llama3_8b", "chaosnli", df, max_points=2000)
    for model in args.models:
        for dataset in NON_DEGENERATE:
            if dataset in args.datasets:
                if model == "llama3_8b" and dataset == "chaosnli":
                    continue
                write_scatter_data(model, dataset, df, max_points=args.max_points)

    print("\n[5/5] Summary table")
    write_summary(df, args.datasets)

    print("\nDone.")


if __name__ == "__main__":
    main()
