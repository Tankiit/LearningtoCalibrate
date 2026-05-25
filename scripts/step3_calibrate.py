"""
Step 3: split-conformal calibration of the gap threshold.

Reads signals.pt from step 2; computes calibration threshold on the
correct-rep portion of the cal split; saves threshold for step 4.

Usage:
  python scripts/step3_calibrate.py --config configs/experiments/llama3_8b_truthfulqa.yaml
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from utils.config import load_config, assert_required
from utils.logging import get_logger
from utils.paths import signals_path, conformal_path, ensure_parent
from utils.seed import set_global_seed
from gap.signal import load_signals
from gap.conformal import ConformalDeferral

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force",  action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    assert_required(cfg, ["model_key", "dataset", "seed"])
    set_global_seed(cfg.seed)

    out_path = conformal_path(cfg.model_key, cfg.dataset, cfg.seed)
    if out_path.exists() and not args.force:
        log.info(f"already calibrated: {out_path}")
        return

    sig = load_signals(signals_path(cfg.model_key, cfg.dataset, cfg.seed))

    # Cal/test split on h^+ rows. Step 4 will use the same split for evaluation.
    cal_frac = (cfg.get("conformal") or {}).get("cal_frac", 0.5)
    alpha = (cfg.get("conformal") or {}).get("alpha", 0.10)

    pos_mask = sig.is_pos
    n_pos = int(pos_mask.sum())
    rng = np.random.default_rng(cfg.seed)
    pos_idx = np.where(pos_mask)[0]
    perm = rng.permutation(len(pos_idx))
    n_cal = int(cal_frac * len(pos_idx))
    cal_pos_idx  = pos_idx[perm[:n_cal]]
    test_pos_idx = pos_idx[perm[n_cal:]]

    gap_pos_cal = sig.gap[cal_pos_idx]

    cd = ConformalDeferral(alpha=alpha)
    threshold = cd.fit(gap_pos_cal)
    coverage  = cd.coverage(sig.gap[test_pos_idx])
    log.info(f"alpha={alpha} → threshold={threshold:.4f}, "
              f"empirical coverage on h^+ test = {coverage:.4f}")

    torch.save({
        "alpha":         alpha,
        "threshold":     threshold,
        "cal_pos_idx":   cal_pos_idx,
        "test_pos_idx":  test_pos_idx,
        "n_pos":         n_pos,
        "cal_frac":      cal_frac,
    }, ensure_parent(out_path))
    log.info(f"saved → {out_path}")


if __name__ == "__main__":
    main()
