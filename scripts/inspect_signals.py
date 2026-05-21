"""
inspect_signals.py — what does the trained probe actually produce?

Reports the distribution of gap, logit_pred, logit_defer for h^+ and h^- rows.
If the gap on TruthfulQA has wide variance and clearly separates h^+ from h^-,
the paper's claim works. If gap and logit_pred are essentially the same
(correlation > 0.95, similar variance), then the defer head learned nothing
useful and the gap reduces to probe-only.

Usage:
    python scripts/inspect_signals.py \\
      --signals outputs/step2_probes/llama3_8b/truthfulqa/seed0/signals.pt
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `utils`, `data`, `probes`, `gap`
# are importable without requiring `pip install -e .`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from gap.signal import load_signals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signals", required=True, type=Path)
    args = p.parse_args()

    sig = load_signals(args.signals)

    pos = sig.is_pos
    neg = ~sig.is_pos

    print(f"\n{'='*60}")
    print(f"Signal inspection: {args.signals}")
    print(f"{'='*60}")
    print(f"N total: {len(sig.gap)}  ({pos.sum()} h^+ rows, {neg.sum()} h^- rows)\n")

    print(f"{'metric':<14} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 60)
    for name, vals in [
        ("logit_pred", sig.logit_pred),
        ("logit_defer", sig.logit_defer),
        ("gap", sig.gap),
        ("logit_pred[h+]", sig.logit_pred[pos]),
        ("logit_pred[h-]", sig.logit_pred[neg]),
        ("logit_defer[h+]", sig.logit_defer[pos]),
        ("logit_defer[h-]", sig.logit_defer[neg]),
        ("gap[h+]", sig.gap[pos]),
        ("gap[h-]", sig.gap[neg]),
    ]:
        print(f"{name:<14} {vals.mean():>10.4f} {vals.std():>10.4f} "
              f"{vals.min():>10.4f} {vals.max():>10.4f}")

    print()

    # ── Key diagnostic 1: did f_defer learn anything? ────────────────────
    print("KEY DIAGNOSTICS")
    print("-" * 60)
    defer_std = sig.logit_defer.std()
    if defer_std < 0.1:
        verdict = "✗ DEAD — defer head is near-constant"
    elif defer_std < 0.5:
        verdict = "△ MARGINAL — defer head has weak variation"
    else:
        verdict = "✓ ALIVE — defer head has meaningful variation"
    print(f"  logit_defer std = {defer_std:.4f}  →  {verdict}")

    # ── Key diagnostic 2: gap vs logit_pred correlation ──────────────────
    corr = np.corrcoef(sig.gap, sig.logit_pred)[0, 1]
    if corr > 0.95:
        verdict = "✗ Gap ≈ logit_pred. Defer head adds no information."
    elif corr > 0.85:
        verdict = "△ Gap mostly tracks logit_pred. Defer head adds little."
    else:
        verdict = "✓ Gap is genuinely different from logit_pred."
    print(f"  corr(gap, logit_pred) = {corr:.4f}  →  {verdict}")

    # ── Key diagnostic 3: does gap separate h^+ from h^-? ────────────────
    gap_pos_mean = sig.gap[pos].mean()
    gap_neg_mean = sig.gap[neg].mean()
    separation = gap_pos_mean - gap_neg_mean
    if separation < 0.2:
        verdict = "✗ Gap does NOT separate h^+ from h^-"
    elif separation < 1.0:
        verdict = "△ Weak separation"
    else:
        verdict = "✓ Clean separation"
    print(f"  mean(gap[h+]) - mean(gap[h-]) = {separation:+.4f}  →  {verdict}")

    # ── Key diagnostic 4: point-biserial correlation with correctness ───────
    from scipy.stats import pointbiserialr
    y_correct = pos.astype(int)
    gap_score = sig.gap
    p_pred_eval = sig.sigma_pred
    rho_gap, p_gap = pointbiserialr(y_correct, gap_score)
    rho_pred, p_pred = pointbiserialr(y_correct, p_pred_eval)
    print(f"  corr(p_pred, y_correct) = {rho_pred:+.4f} (p={p_pred:.4f})")
    print(f"  corr(gap,    y_correct) = {rho_gap:+.4f} (p={p_gap:.4f})")

    # ── Quick AUROC for gap and logit_pred on h^+ vs h^- separation ──────
    from sklearn.metrics import roc_auc_score
    y = pos.astype(int)
    auroc_gap   = roc_auc_score(y, sig.gap)
    auroc_pred  = roc_auc_score(y, sig.logit_pred)
    auroc_defer = roc_auc_score(y, sig.logit_defer)
    print()
    print(f"  AUROC (separating h^+ from h^-):")
    print(f"    gap:         {auroc_gap:.4f}")
    print(f"    logit_pred:  {auroc_pred:.4f}")
    print(f"    logit_defer: {auroc_defer:.4f}")
    diff = auroc_gap - auroc_pred
    if abs(diff) < 0.005:
        verdict = "✗ gap and logit_pred are equivalent"
    elif diff > 0:
        verdict = f"✓ gap beats logit_pred by {diff:.4f}"
    else:
        verdict = f"△ gap is {-diff:.4f} WORSE than logit_pred"
    print(f"    →  {verdict}")


if __name__ == "__main__":
    main()
