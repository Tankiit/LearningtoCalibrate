"""
SCOD-optimal selector
=====================

Constructs the Bayes-optimal selector in the (p_pred, p_defer) plane
under the SCOD cost structure, then compares against:
  - probe-only (1D selector on p_pred)
  - gap (delta = logit(p_pred) - logit(p_defer))
  - learned 2D selector (logistic regression on (logit(p_pred), logit(p_defer)))

If the gap is well-designed, gap should approach the 2D Bayes-optimal
selector on showcase datasets. If gap is FAR from the 2D optimum, you're
leaving information on the table -- and a reviewer will say so.

The SCOD cost structure (per Franc, Paplham & Prusa ECCV 2024):

    cost(commit, y_correct)      = 1 - y_correct
    cost(defer,  y_expert_reliable) = c_defer - y_expert_reliable

where c_defer is the per-instance expert query cost (typically constant in [0,1]).
The Bayes-optimal action at each x is:

    commit iff  E[cost(commit|x)] <= E[cost(defer|x)]
    i.e.,           (1 - p_pred(x))     <=  (c_defer - p_defer(x))
    i.e.,           p_pred(x) + p_defer(x) >= 1 + c_defer

This gives a STRAIGHT-LINE decision boundary in (p_pred, p_defer) at
slope -1, intercept (1 + c_defer). The gap signal delta = logit(p_pred) - logit(p_defer)
is a DIFFERENT boundary (a curve in (p_pred, p_defer) space). They coincide
only when both are calibrated and (p_pred, p_defer) are jointly Gaussian.

Usage:
  python scripts/scod_optimal_selector.py \
      --config configs/experiments/llama3_8b_truthfulqa.yaml
  python scripts/scod_optimal_selector.py \
      --model llama3_8b --dataset truthfulqa
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.risk_coverage import aurc
from utils.config import load_config, assert_required
from utils.device import configure_torch_threads, get_torch_device
from utils.logging import get_logger
from utils.seed import set_global_seed

log = get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# =============================================================================
# The SCOD-optimal selector
# =============================================================================

def scod_optimal_score(
    p_pred: np.ndarray,
    p_defer: np.ndarray,
    c_defer: float = 0.3,
) -> np.ndarray:
    """
    Returns the commit-confidence score: higher = more confident commit.

    INTUITION: the SCOD-optimal selector commits iff
        p_pred + p_defer >= 1 + c_defer
    Rearranging: commit-confidence = p_pred + p_defer - (1 + c_defer).
    For AURC computation we just need a *ranking*, so the constant doesn't
    matter.

    But: this is optimal ONLY IF p_pred and p_defer are calibrated probabilities.
    Probe outputs typically aren't perfectly calibrated, so in practice you
    should temperature-scale both first (see scod_optimal_calibrated_score).
    """
    return p_pred + p_defer


def scod_optimal_calibrated_score(
    p_pred_train: np.ndarray,
    p_defer_train: np.ndarray,
    y_correct_train: np.ndarray,
    y_expert_reliable_train: np.ndarray,
    p_pred_test: np.ndarray,
    p_defer_test: np.ndarray,
) -> np.ndarray:
    """
    Calibrate both probes via per-head Platt scaling on a held-out split,
    then apply the SCOD-optimal additive selector.

    Calibration is non-negotiable here: raw probe outputs are over-confident
    (LRs trained on contrastive pairs tend to push toward 0 or 1), and the
    additive decision rule is calibration-sensitive in a way the gap (which
    only depends on the difference of logits) is not.
    """
    # 1D Platt scaler for each head
    pp_cal = (
        LogisticRegression()
        .fit(p_pred_train.reshape(-1, 1), y_correct_train)
        .predict_proba(p_pred_test.reshape(-1, 1))[:, 1]
    )
    pd_cal = (
        LogisticRegression()
        .fit(p_defer_train.reshape(-1, 1), y_expert_reliable_train)
        .predict_proba(p_defer_test.reshape(-1, 1))[:, 1]
    )
    return pp_cal + pd_cal


# =============================================================================
# The learned 2D selector (the empirical-Bayes upper bound)
# =============================================================================

def learned_2d_selector_score(
    p_pred_train: np.ndarray,
    p_defer_train: np.ndarray,
    y_correct_train: np.ndarray,
    p_pred_test: np.ndarray,
    p_defer_test: np.ndarray,
) -> np.ndarray:
    """
    Train a logistic regression on (logit(p_pred), logit(p_defer)) to predict
    y_correct on a held-out split, then use its score on the eval split.

    INTUITION: this is the *learned* 2D selector. If it dominates the SCOD-
    optimal additive selector, the SCOD cost structure is mismatched to the
    actual decision problem. If it doesn't dominate, you've confirmed the
    SCOD structure is the right one.

    The learned 2D selector is the empirical-Bayes UPPER BOUND on what any
    selector in (p_pred, p_defer) space can achieve. The gap should approach
    it on showcase datasets.
    """
    X_train = np.stack([_safe_logit(p_pred_train),
                        _safe_logit(p_defer_train)], axis=1)
    X_test = np.stack([_safe_logit(p_pred_test),
                       _safe_logit(p_defer_test)], axis=1)

    clf = LogisticRegression(C=1.0).fit(X_train, y_correct_train)
    return clf.predict_proba(X_test)[:, 1]


# =============================================================================
# Full comparison
# =============================================================================

def compare_selectors(
    p_pred_full: np.ndarray,
    p_defer_full: np.ndarray,
    y_correct_full: np.ndarray,
    y_expert_reliable_full: np.ndarray,
    c_defer: float = 0.3,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    K-fold CV comparison of:
      - probe_only         : 1D, p_pred
      - gap                : 1D in difference of logits
      - scod_optimal_raw   : 2D additive, uncalibrated
      - scod_optimal_cal   : 2D additive, calibrated
      - learned_2d         : 2D empirical Bayes -- the upper bound

    Returns a DataFrame with per-fold AURC and a summary.

    The CV split here is CRITICAL: scod_optimal_cal and learned_2d need
    separate training data for their calibration / fitting step, which is
    why we wrap them in K-fold. probe_only and gap don't need extra data
    (they're already trained), but we apply the same fold structure for a
    fair comparison.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (cal_idx, test_idx) in enumerate(kf.split(p_pred_full)):
        pp_cal, pp_te = p_pred_full[cal_idx], p_pred_full[test_idx]
        pd_cal, pd_te = p_defer_full[cal_idx], p_defer_full[test_idx]
        yc_cal, yc_te = y_correct_full[cal_idx], y_correct_full[test_idx]
        ye_cal = y_expert_reliable_full[cal_idx]

        scores = {
            "probe_only":       pp_te,
            "gap":              _safe_logit(pp_te) - _safe_logit(pd_te),
            "scod_optimal_raw": scod_optimal_score(pp_te, pd_te, c_defer),
            "learned_2d":       learned_2d_selector_score(
                pp_cal, pd_cal, yc_cal, pp_te, pd_te,
            ),
        }
        if len(np.unique(ye_cal)) >= 2:
            scores["scod_optimal_cal"] = scod_optimal_calibrated_score(
                pp_cal, pd_cal, yc_cal, ye_cal, pp_te, pd_te,
            )
        for name, s in scores.items():
            rows.append({
                "fold": fold,
                "method": name,
                "aurc": aurc(score=s, correct=yc_te),
                "status": "ok",
            })
        if len(np.unique(ye_cal)) < 2:
            rows.append({
                "fold": fold,
                "method": "scod_optimal_cal",
                "aurc": np.nan,
                "status": "skipped_constant_expert",
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("method")["aurc"].agg(["mean", "std"]).round(4)
    print("\n=== Selector comparison (AURC, K-fold CV) ===")
    print(summary)
    return df


# =============================================================================
# Data loading (same pattern as scod_diagnostics.py)
# =============================================================================

def load_probes_and_scores(
    model_key: str,
    dataset: str,
    seed: int = 0,
    halueval_style_labels: bool = False,
    ova_train: dict | None = None,
):
    """
    Load cached hidden states, train OVA probes, and extract
    (p_pred, p_defer) probabilities.

    Returns:
        p_pred:    (M,) sigma(f_pred) on eval rows
        p_defer:   (M,) sigma(f_defer) on eval rows
        y_correct: (M,) correctness labels
        y_expert_reliable: (M,) bool expert reliability for eval rows
    """
    import torch

    from utils.paths import hidden_states_path
    from data.registry import load_dataset
    from probes.training import prepare_training_data, train_ova
    from probes.ova import OVATrainConfig

    h_path = hidden_states_path(model_key, dataset)
    if not h_path.exists():
        raise FileNotFoundError(
            f"Run step 1 first: hidden states missing at {h_path}"
        )

    blob = torch.load(h_path, map_location="cpu", weights_only=False)
    h_pos = np.asarray(blob["h_pos"])
    h_neg = np.asarray(blob["h_neg"])
    extracted_ids = np.asarray(blob["example_ids"])
    N = len(h_pos)

    if "expert_reliable" in blob:
        expert_reliable = np.asarray(blob["expert_reliable"], dtype=bool)
        if len(expert_reliable) != len(extracted_ids):
            raise ValueError(
                f"cached expert_reliable length mismatch: "
                f"{len(expert_reliable)} vs {len(extracted_ids)}"
            )
    else:
        records = load_dataset(dataset)
        er_by_id = {r.example_id: r.expert_reliable for r in records}
        expert_reliable = np.array(
            [er_by_id[eid] for eid in extracted_ids], dtype=bool,
        )

    data = prepare_training_data(
        h_pos=h_pos, h_neg=h_neg,
        expert_reliable=expert_reliable,
        example_ids=extracted_ids,
        halueval_style=halueval_style_labels,
    )

    ova_cfg = OVATrainConfig(**(ova_train or {}))
    set_global_seed(seed)
    model = train_ova(data, cfg=ova_cfg, seed=seed)

    configure_torch_threads()
    device = get_torch_device()
    model = model.to(device).eval()
    # y_correct from judge labels if available. Otherwise fall back to the
    # contrastive evaluation used by scripts/step4_evaluate.py: h^+ rows are
    # correct commits and h^- rows are incorrect commits.
    judge_path = (
        Path("outputs") / "step4_eval" / model_key / dataset
        / f"seed{seed}" / "judge_labels.pt"
    )
    if judge_path.exists():
        judge_blob = torch.load(judge_path, map_location="cpu",
                                weights_only=False)
        y_correct = judge_blob["labels"].astype(int)
        h_eval = h_pos[:len(y_correct)]
        y_expert_eval = expert_reliable[:len(y_correct)]
    else:
        y_correct = np.concatenate([
            np.ones(N, dtype=int),
            np.zeros(N, dtype=int),
        ])
        h_eval = np.concatenate([h_pos, h_neg], axis=0)
        y_expert_eval = np.concatenate([expert_reliable, expert_reliable])
        log.warning(
            "No judge labels found; using contrastive h+/h- correctness fallback."
        )

    h_t = torch.from_numpy(h_eval).float().to(device)
    with torch.no_grad():
        lp, ld = model(h_t)
    p_pred = torch.sigmoid(lp).cpu().numpy()
    p_defer = torch.sigmoid(ld).cpu().numpy()

    return p_pred, p_defer, y_correct[:len(p_pred)], y_expert_eval


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="SCOD-optimal selector comparison",
    )
    p.add_argument("--config", default=None,
                   help="YAML config for single model+dataset")
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--c-defer", type=float, default=0.3,
                   help="Expert cost parameter for SCOD-optimal selector")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/scod_selectors.parquet")
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config)
        assert_required(cfg, ["model_key", "dataset", "seed", "halueval_style_labels"])
        model_key = cfg.model_key
        dataset = cfg.dataset
        seed = cfg.seed
        halueval_style_labels = cfg.halueval_style_labels
        ova_train = cfg.get("ova_train")
    elif args.model and args.dataset:
        model_key = args.model
        dataset = args.dataset
        seed = args.seed
        halueval_style_labels = False
        ova_train = None
    else:
        p.error("Provide either --config or both --model and --dataset")

    p_pred, p_defer, y_correct, y_expert_rel = load_probes_and_scores(
        model_key, dataset, seed=seed,
        halueval_style_labels=halueval_style_labels,
        ova_train=ova_train,
    )

    print(f"\n{model_key} / {dataset}: {len(p_pred)} examples")
    print(f"  y_correct True: {y_correct.sum()}/{len(y_correct)}")
    print(f"  expert_reliable True: {y_expert_rel.sum()}/{len(y_expert_rel)}")

    df = compare_selectors(
        p_pred, p_defer, y_correct, y_expert_rel.astype(int),
        c_defer=args.c_defer,
        n_folds=args.n_folds,
        seed=seed,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
