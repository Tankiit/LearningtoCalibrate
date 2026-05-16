"""
Step 4: evaluate AURC for gap vs all baselines on the test split.

Outputs results.parquet with columns:
  method, aurc, n, model_key, dataset, seed

Aggregating across (model, dataset, seed) gives the headline table.

Usage:
  python scripts/step4_evaluate.py --config configs/experiments/llama3_8b_truthfulqa.yaml
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from utils.config import load_config, assert_required
from utils.logging import get_logger
from utils.paths import (
    signals_path, conformal_path, results_path, logprobs_path, ensure_parent,
)
from utils.seed import set_global_seed
from gap.signal import load_signals
from eval.risk_coverage import aurc_table
from eval.baselines import (
    baseline_gap, baseline_sigma_pred, baseline_logit_pred,
    baseline_phillips_sum,
)

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force",  action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    assert_required(cfg, ["model_key", "dataset", "seed"])
    set_global_seed(cfg.seed)

    out_path = results_path(cfg.model_key, cfg.dataset, cfg.seed)
    if out_path.exists() and not args.force:
        log.info(f"already evaluated: {out_path}")
        return

    # ── Load all artifacts ──────────────────────────────────────────────
    sig = load_signals(signals_path(cfg.model_key, cfg.dataset, cfg.seed))
    cal_blob = torch.load(
        conformal_path(cfg.model_key, cfg.dataset, cfg.seed),
        map_location="cpu", weights_only=False,
    )
    test_pos_idx = np.asarray(cal_blob["test_pos_idx"])
    neg_idx = np.where(~sig.is_pos)[0]

    # Test set: h^+ test split + ALL h^- (negatives are evaluation, not cal).
    eval_idx = np.concatenate([test_pos_idx, neg_idx])
    is_correct = np.concatenate([
        np.ones(len(test_pos_idx),  dtype=bool),  # h^+ rows are 'correct' commits
        np.zeros(len(neg_idx),      dtype=bool),  # h^- rows are 'wrong' commits
    ])

    # ── Build score matrix per method ──────────────────────────────────
    # Use sig at eval_idx for each method
    sub = _subset_signals(sig, eval_idx)

    scores = {
        "gap":             baseline_gap(sub),
        "probe_pred_only": baseline_sigma_pred(sub),
        "logit_pred":      baseline_logit_pred(sub),
        "probe_amb_only":  sub.sigma_defer,
        "phillips_sum":    baseline_phillips_sum(sub),
    }

    # Optional logprob baseline (length-normalized, if extracted)
    lp_path = logprobs_path(cfg.model_key, cfg.dataset)
    if lp_path.exists():
        lp_blob = torch.load(lp_path, map_location="cpu", weights_only=False)
        lp_pos = np.asarray(lp_blob["logprob_pos"])
        lp_neg = np.asarray(lp_blob["logprob_neg"])
        # Build aligned array indexed by sig.example_ids
        lp_by_id = {**dict(zip(lp_blob["example_ids"], lp_pos))}  # h^+ rows
        lp_by_id_neg = dict(zip(lp_blob["example_ids"], lp_neg))   # h^- rows

        lp_eval = np.empty(len(eval_idx), dtype=float)
        for j, idx in enumerate(eval_idx):
            ex_id = sig.example_ids[idx]
            lp_eval[j] = lp_by_id[ex_id] if sig.is_pos[idx] else lp_by_id_neg[ex_id]
        scores["logprob"] = lp_eval

        if "seq_entropy_pos" in lp_blob and "seq_entropy_neg" in lp_blob:
            ent_pos = np.asarray(lp_blob["seq_entropy_pos"])
            ent_neg = np.asarray(lp_blob["seq_entropy_neg"])
            ent_by_id = dict(zip(lp_blob["example_ids"], ent_pos))
            ent_by_id_neg = dict(zip(lp_blob["example_ids"], ent_neg))
            ent_eval = np.empty(len(eval_idx), dtype=float)
            for j, idx in enumerate(eval_idx):
                ex_id = sig.example_ids[idx]
                ent_eval[j] = ent_by_id[ex_id] if sig.is_pos[idx] else ent_by_id_neg[ex_id]
            scores["seq_entropy"] = -ent_eval

    # ── Compute AURC for each ──────────────────────────────────────────
    table = aurc_table(scores, is_correct)
    table["model_key"] = cfg.model_key
    table["dataset"]   = cfg.dataset
    table["seed"]      = cfg.seed

    table.to_parquet(ensure_parent(out_path))
    log.info(f"saved → {out_path}")
    log.info("\n" + table.to_string(index=False))


def _subset_signals(sig, idx):
    """Slice GapSignals along the row axis."""
    from gap.signal import GapSignals
    return GapSignals(
        example_ids = sig.example_ids[idx],
        is_pos      = sig.is_pos[idx],
        logit_pred  = sig.logit_pred[idx],
        logit_defer = sig.logit_defer[idx],
        gap         = sig.gap[idx],
        sigma_pred  = sig.sigma_pred[idx],
        sigma_defer = sig.sigma_defer[idx],
    )


if __name__ == "__main__":
    main()
