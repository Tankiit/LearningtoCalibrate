"""
Path 2: Verma OVA-L2D alternative for the credal-width diagnostic.

This script runs directly on the repo's compact extraction cache:

    outputs/step1_extract/{model}/{dataset}/hidden_states.pt

It compares two defer-head targets:
  - our target:      expert_reliable from the dataset adapter
  - Verma target:   1 - y_model on contrastive pairs

Outputs:
  - results/path2_verma_alternative_{model}_{dataset}.parquet
  - results/path2_diagnostic_summary_{model}_{dataset}.json
  - results/path2_credal_width_comparison_{model}_{dataset}.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.risk_coverage import aurc  # noqa: E402
from data.registry import load_dataset  # noqa: E402
from utils.paths import hidden_states_path  # noqa: E402


DEFAULT_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DEFAULT_DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "truthfulqa",
    "halueval_qa",
    "triviaqa",
    "popqa",
]


@dataclass
class Config:
    model: str
    dataset: str
    output_dir: Path = Path("results")
    k_bootstrap: int = 20
    probe_c: float = 1.0
    class_weight: str | None = "balanced"
    train_frac: float = 0.6
    cal_frac: float = 0.2
    seed: int = 0
    max_items: int | None = None


class ConstantProbabilityProbe:
    """Drop-in predict_proba object for one-class training targets."""

    def __init__(self, p_one: float):
        self.p_one = float(np.clip(p_one, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.p_one, dtype=float)
        return np.column_stack([1.0 - p, p])


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_probe(
    h_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c: float,
    class_weight: str | None,
    seed: int,
) -> LogisticRegression | ConstantProbabilityProbe:
    y_train = np.asarray(y_train).astype(int)
    uniq = np.unique(y_train)
    if len(uniq) < 2:
        return ConstantProbabilityProbe(float(uniq[0]))
    probe = LogisticRegression(
        C=c,
        class_weight=class_weight,
        max_iter=1000,
        random_state=seed,
    )
    probe.fit(h_train, y_train)
    return probe


def bootstrap_credal_intervals(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_eval: np.ndarray,
    *,
    k_bootstrap: int,
    probe_c: float,
    class_weight: str | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = len(h_train)
    all_probs = np.zeros((k_bootstrap, len(h_eval)), dtype=np.float32)
    y_train = np.asarray(y_train).astype(int)

    for k in range(k_bootstrap):
        idx = rng.choice(n_train, size=n_train, replace=True)
        probe = fit_probe(
            h_train[idx],
            y_train[idx],
            c=probe_c,
            class_weight=class_weight,
            seed=seed + k,
        )
        all_probs[k] = probe.predict_proba(h_eval)[:, 1]

    return all_probs.min(axis=0), all_probs.mean(axis=0), all_probs.max(axis=0)


def load_contrastive_cache(cfg: Config) -> dict[str, Any]:
    import torch

    path = hidden_states_path(cfg.model, cfg.dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden-state cache: {path}")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    h_pos = np.asarray(blob["h_pos"], dtype=np.float32)
    h_neg = np.asarray(blob["h_neg"], dtype=np.float32)
    example_ids = np.asarray(blob.get("example_ids", np.arange(len(h_pos))))

    if "expert_reliable" in blob:
        expert = np.asarray(blob["expert_reliable"]).astype(int)
    else:
        records = load_dataset(cfg.dataset)
        expert_by_id = {record.example_id: int(record.expert_reliable) for record in records}
        try:
            expert = np.asarray([expert_by_id[str(example_id)] for example_id in example_ids]).astype(int)
        except KeyError as exc:
            raise KeyError(
                f"Could not recover expert_reliable for {cfg.dataset} example_id={exc.args[0]!r}"
            ) from exc

    if cfg.max_items is not None:
        h_pos = h_pos[: cfg.max_items]
        h_neg = h_neg[: cfg.max_items]
        expert = expert[: cfg.max_items]
        example_ids = example_ids[: cfg.max_items]

    if not (len(h_pos) == len(h_neg) == len(expert)):
        raise ValueError(
            f"Cache length mismatch for {cfg.model}/{cfg.dataset}: "
            f"h_pos={len(h_pos)}, h_neg={len(h_neg)}, expert={len(expert)}"
        )

    h = np.concatenate([h_pos, h_neg], axis=0)
    y_model = np.concatenate([
        np.ones(len(h_pos), dtype=int),
        np.zeros(len(h_neg), dtype=int),
    ])
    item_id = np.concatenate([np.arange(len(h_pos)), np.arange(len(h_pos))])
    example_id = np.concatenate([example_ids, example_ids])

    return {
        "h": h,
        "y_model": y_model,
        "y_expert": np.concatenate([expert, expert]),
        "y_expert_verma": 1 - y_model,
        "item_id": item_id,
        "example_id": example_id,
        "n_items": len(h_pos),
        "cache_path": str(path),
    }


def split_by_item(data: dict[str, Any], cfg: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    item_ids = np.unique(data["item_id"])
    rng.shuffle(item_ids)

    n_train = int(cfg.train_frac * len(item_ids))
    n_cal = int(cfg.cal_frac * len(item_ids))
    train_ids = item_ids[:n_train]
    cal_ids = item_ids[n_train : n_train + n_cal]
    eval_ids = item_ids[n_train + n_cal :]

    return {
        "train": np.isin(data["item_id"], train_ids),
        "cal": np.isin(data["item_id"], cal_ids),
        "eval": np.isin(data["item_id"], eval_ids),
    }


def evaluate_construction(
    data: dict[str, Any],
    masks: dict[str, np.ndarray],
    y_defer_key: str,
    cfg: Config,
) -> dict[str, Any]:
    h_train = data["h"][masks["train"]]
    h_eval = data["h"][masks["eval"]]
    y_model_train = data["y_model"][masks["train"]]
    y_model_eval = data["y_model"][masks["eval"]].astype(bool)
    y_defer_train = data[y_defer_key][masks["train"]]

    f_pred = fit_probe(
        h_train,
        y_model_train,
        c=cfg.probe_c,
        class_weight=cfg.class_weight,
        seed=cfg.seed,
    )
    f_defer = fit_probe(
        h_train,
        y_defer_train,
        c=cfg.probe_c,
        class_weight=cfg.class_weight,
        seed=cfg.seed,
    )

    p_pred = f_pred.predict_proba(h_eval)[:, 1]
    p_defer = f_defer.predict_proba(h_eval)[:, 1]
    gap = safe_logit(p_pred) - safe_logit(p_defer)

    p_pred_lower, p_pred_mean, p_pred_upper = bootstrap_credal_intervals(
        h_train,
        y_model_train,
        h_eval,
        k_bootstrap=cfg.k_bootstrap,
        probe_c=cfg.probe_c,
        class_weight=cfg.class_weight,
        seed=cfg.seed,
    )
    p_defer_lower, p_defer_mean, p_defer_upper = bootstrap_credal_intervals(
        h_train,
        y_defer_train,
        h_eval,
        k_bootstrap=cfg.k_bootstrap,
        probe_c=cfg.probe_c,
        class_weight=cfg.class_weight,
        seed=cfg.seed + 10_000,
    )
    robust_gap = safe_logit(p_pred_lower) - safe_logit(p_defer_lower)

    pred_width = p_pred_upper - p_pred_lower
    defer_width = p_defer_upper - p_defer_lower
    if np.std(defer_width) > 1e-12 and len(np.unique(y_model_eval)) == 2:
        rho, p_value = pointbiserialr(y_model_eval.astype(int), defer_width)
    else:
        rho, p_value = np.nan, np.nan

    return {
        "target": y_defer_key,
        "metrics": {
            "aurc_gap": aurc(gap, y_model_eval),
            "aurc_robust_gap": aurc(robust_gap, y_model_eval),
            "aurc_p_pred": aurc(p_pred, y_model_eval),
            "aurc_p_pred_mean": aurc(p_pred_mean, y_model_eval),
            "aurc_p_defer": aurc(p_defer, y_model_eval),
        },
        "width_stats": {
            "pred_width_mean": float(pred_width.mean()),
            "pred_width_median": float(np.median(pred_width)),
            "pred_width_p90": float(np.quantile(pred_width, 0.9)),
            "defer_width_mean": float(defer_width.mean()),
            "defer_width_median": float(np.median(defer_width)),
            "defer_width_p90": float(np.quantile(defer_width, 0.9)),
        },
        "defer_width_correctness_corr": {"rho": float(rho), "p": float(p_value)},
        "per_item": {
            "gap": gap,
            "robust_gap": robust_gap,
            "p_pred_lower": p_pred_lower,
            "p_pred_mean": p_pred_mean,
            "p_pred_upper": p_pred_upper,
            "p_defer_lower": p_defer_lower,
            "p_defer_mean": p_defer_mean,
            "p_defer_upper": p_defer_upper,
            "y_correct": y_model_eval.astype(int),
        },
    }


def plot_widths(ours: dict[str, Any], verma: dict[str, Any], save_to: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    our_width = ours["per_item"]["p_defer_upper"] - ours["per_item"]["p_defer_lower"]
    verma_width = verma["per_item"]["p_defer_upper"] - verma["per_item"]["p_defer_lower"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, width, label, color in [
        (axes[0], our_width, "Our defer target", "C0"),
        (axes[1], verma_width, "Verma defer target", "C1"),
    ]:
        ax.hist(width, bins=30, alpha=0.75, color=color)
        ax.axvline(np.median(width), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("bootstrap credal width")
        ax.set_ylabel("count")
        ax.set_title(f"{label}\nmedian={np.median(width):.3f}")
    fig.suptitle(title)
    fig.tight_layout()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=180)
    plt.close(fig)


def write_outputs(
    cfg: Config,
    data: dict[str, Any],
    masks: dict[str, np.ndarray],
    ours: dict[str, Any],
    verma: dict[str, Any],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.model}_{cfg.dataset}"
    summary_path = cfg.output_dir / f"path2_diagnostic_summary_{stem}.json"
    parquet_path = cfg.output_dir / f"path2_verma_alternative_{stem}.parquet"
    plot_path = cfg.output_dir / f"path2_credal_width_comparison_{stem}.png"

    summary = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "cache_path": data["cache_path"],
        "n_items": data["n_items"],
        "n_contrastive_eval": int(masks["eval"].sum()),
        "ours": {k: v for k, v in ours.items() if k != "per_item"},
        "verma": {k: v for k, v in verma.items() if k != "per_item"},
        "diagnostic_check": {
            "ours_predicted_regime": (
                "showcase" if ours["width_stats"]["defer_width_median"] > 0.10 else "diagnostic"
            ),
            "verma_predicted_regime": (
                "showcase" if verma["width_stats"]["defer_width_median"] > 0.10 else "diagnostic"
            ),
            "ours_aurc_improves_over_pred": (
                ours["metrics"]["aurc_gap"] < ours["metrics"]["aurc_p_pred"]
            ),
            "verma_aurc_improves_over_pred": (
                verma["metrics"]["aurc_gap"] < verma["metrics"]["aurc_p_pred"]
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    eval_example_ids = data["example_id"][masks["eval"]]
    per_item = pd.DataFrame({
        "model": cfg.model,
        "dataset": cfg.dataset,
        "example_id": eval_example_ids,
        "y_correct": ours["per_item"]["y_correct"],
        "gap_ours": ours["per_item"]["gap"],
        "gap_verma": verma["per_item"]["gap"],
        "robust_gap_ours": ours["per_item"]["robust_gap"],
        "robust_gap_verma": verma["per_item"]["robust_gap"],
        "p_defer_lower_ours": ours["per_item"]["p_defer_lower"],
        "p_defer_mean_ours": ours["per_item"]["p_defer_mean"],
        "p_defer_upper_ours": ours["per_item"]["p_defer_upper"],
        "p_defer_lower_verma": verma["per_item"]["p_defer_lower"],
        "p_defer_mean_verma": verma["per_item"]["p_defer_mean"],
        "p_defer_upper_verma": verma["per_item"]["p_defer_upper"],
    })
    per_item.to_parquet(parquet_path, index=False)
    plot_widths(ours, verma, plot_path, f"{cfg.model}/{cfg.dataset} (K={cfg.k_bootstrap})")

    print(f"Wrote {summary_path}")
    print(f"Wrote {parquet_path}")
    print(f"Wrote {plot_path}")


def run(cfg: Config) -> dict[str, Any]:
    print(f"\n=== Path 2 Verma OVA alternative: {cfg.model}/{cfg.dataset} ===")
    data = load_contrastive_cache(cfg)
    masks = split_by_item(data, cfg)
    for name, mask in masks.items():
        print(f"  {name:5s}: {int(mask.sum())} contrastive points")

    ours = evaluate_construction(data, masks, "y_expert", cfg)
    verma = evaluate_construction(data, masks, "y_expert_verma", cfg)
    write_outputs(cfg, data, masks, ours, verma)

    print(
        "  ours:  "
        f"AURC(gap)={ours['metrics']['aurc_gap']:.4f}, "
        f"AURC(pred)={ours['metrics']['aurc_p_pred']:.4f}, "
        f"median_width={ours['width_stats']['defer_width_median']:.4f}"
    )
    print(
        "  verma: "
        f"AURC(gap)={verma['metrics']['aurc_gap']:.4f}, "
        f"AURC(pred)={verma['metrics']['aurc_p_pred']:.4f}, "
        f"median_width={verma['width_stats']['defer_width_median']:.4f}"
    )
    return {"ours": ours, "verma": verma}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=DEFAULT_MODELS)
    parser.add_argument("--dataset", required=True, choices=DEFAULT_DATASETS + ["multinli_disagreement"])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--K", type=int, default=20, dest="k_bootstrap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    run(Config(**vars(args)))


if __name__ == "__main__":
    main()
