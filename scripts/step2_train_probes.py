"""
Step 2: train OVA probes on cached hidden states; emit per-example signals.

Usage:
  python scripts/step2_train_probes.py --config configs/experiments/llama3_8b_truthfulqa.yaml
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

from utils.config import load_config, assert_required
from utils.logging import get_logger
from utils.paths import (
    hidden_states_path, ova_heads_path, signals_path, ensure_parent,
)
from utils.seed import set_global_seed
from data.registry import load_dataset
from probes.ova import OVAHeads, OVATrainConfig
from probes.training import prepare_training_data, train_ova
from gap.signal import compute_signals, save_signals

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force",  action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    assert_required(cfg, ["model_key", "dataset", "seed", "halueval_style_labels"])
    set_global_seed(cfg.seed)

    out_heads   = ova_heads_path(cfg.model_key, cfg.dataset, cfg.seed)
    out_signals = signals_path(cfg.model_key, cfg.dataset, cfg.seed)
    if out_heads.exists() and out_signals.exists() and not args.force:
        log.info(f"already trained: {out_heads.parent}")
        return

    # ── Load hidden states ──────────────────────────────────────────────
    h_path = hidden_states_path(cfg.model_key, cfg.dataset)
    if not h_path.exists():
        raise FileNotFoundError(
            f"Run step 1 first: hidden states missing at {h_path}"
        )
    blob = torch.load(h_path, map_location="cpu", weights_only=False)
    h_pos = np.asarray(blob["h_pos"])
    h_neg = np.asarray(blob["h_neg"])
    extracted_ids = np.asarray(blob["example_ids"])

    # ── Pull per-record metadata from the cache when available ──────────
    # Full-matrix imports carry this array so downstream runs do not need
    # to re-download HF datasets just to recover expert_reliable labels.
    if "expert_reliable" in blob:
        expert_reliable = np.asarray(blob["expert_reliable"], dtype=bool)
        if len(expert_reliable) != len(extracted_ids):
            raise ValueError(
                f"cached expert_reliable length mismatch: "
                f"{len(expert_reliable)} vs {len(extracted_ids)}"
            )
        log.info("loaded expert_reliable labels from hidden-state cache")
    else:
        log.info(f"loading dataset {cfg.dataset!r} for expert_reliable labels...")
        records = load_dataset(cfg.dataset)
        er_by_id = {r.example_id: r.expert_reliable for r in records}
        expert_reliable = np.array(
            [er_by_id[eid] for eid in extracted_ids], dtype=bool,
        )
    log.info(f"expert_reliable=True on {expert_reliable.sum()}/{len(expert_reliable)}")

    # ── Prepare stacked training data ──────────────────────────────────
    data = prepare_training_data(
        h_pos=h_pos, h_neg=h_neg,
        expert_reliable=expert_reliable,
        example_ids=extracted_ids,
        halueval_style=cfg.halueval_style_labels,
    )

    # ── Train ───────────────────────────────────────────────────────────
    train_cfg = OVATrainConfig(**(cfg.get("ova_train") or {}))
    log.info(f"training OVA heads (input_dim={data.h.shape[1]})...")
    model = train_ova(data, cfg=train_cfg, seed=cfg.seed)

    # ── Save heads ──────────────────────────────────────────────────────
    ensure_parent(out_heads)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  model.input_dim,
        "config":     dict(cfg),
    }, out_heads)
    log.info(f"saved heads → {out_heads}")

    # ── Compute and save signals ────────────────────────────────────────
    is_pos = np.concatenate([
        np.ones(len(h_pos), dtype=bool),
        np.zeros(len(h_neg), dtype=bool),
    ])
    signals = compute_signals(
        model       = model,
        h           = data.h,
        is_pos      = is_pos,
        example_ids = data.example_ids,
    )
    save_signals(signals, ensure_parent(out_signals))
    log.info(f"saved signals → {out_signals}")


if __name__ == "__main__":
    main()
