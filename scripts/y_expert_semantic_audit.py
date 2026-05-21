"""
y_expert Semantic Audit
=======================

Goal: prove that the gap mechanism's performance is determined by the
PER-EXAMPLE STRUCTURE of y_expert, not by its source (real-human vs proxy).

This single experiment neutralizes the most dangerous reviewer concern (S1):
"y_expert isn't really an expert signal."

The defense works by showing:
  - With NATIVE y_expert: gap beats probe-only on showcase datasets
  - With SHUFFLED y_expert (preserving marginal, destroying per-example signal):
    gap collapses to probe-only
  - With CONSTANT y_expert: gap collapses to scaled probe-only
  - With TOPIC-INDICATOR y_expert on TruthfulQA: reproduces the observed collapse
    (S4 defense -- shows the TruthfulQA result IS "f_defer learned topic")

If native >> shuffled on showcase datasets, the per-example variation is
load-bearing and the paper's claim is empirically grounded.
If native ~ shuffled, the paper has a deeper problem and you need Path A or B.

Usage:
  python scripts/y_expert_semantic_audit.py \
      --config configs/experiments/llama3_8b_truthfulqa.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config, assert_required
from utils.logging import get_logger
from utils.paths import hidden_states_path
from utils.seed import set_global_seed
from utils.device import configure_torch_threads, get_torch_device
from data.registry import load_dataset
from data.schema import Record
from probes.ova import OVAHeads, OVATrainConfig
from probes.training import prepare_training_data, train_ova
from gap.signal import compute_signals
from eval.risk_coverage import aurc, risk_coverage_curve, coverage_at_accuracy

log = get_logger(__name__)


# =============================================================================
# Variant builders. Each takes the original records + already-computed
# per-example y_model (model correctness, i.e. is_pos) and returns a new
# y_expert array (float in [0,1] -- expert reliability signal).
#
# The existing codebase stores expert_reliable as bool. We convert to float
# inside variant_native and let other variants operate in float space.
# =============================================================================

def variant_native(records, y_model, rng):
    """The unchanged y_expert from the adapter (expert_reliable as float)."""
    return np.array([float(r.expert_reliable) for r in records], dtype=np.float32)


def variant_shuffled_within_y_model(records, y_model, rng):
    """
    INTUITION: preserve the marginal distribution of y_expert AND its
    relationship with y_model (per-stratum mean), but destroy
    per-example structure. If gap performance survives this, the per-example
    variation wasn't doing useful work.

    Procedure: for each value of y_model in {0, 1}, permute y_expert among
    those indices.
    """
    y_exp = np.array([float(r.expert_reliable) for r in records], dtype=np.float32)
    out = y_exp.copy()
    for ym in (0, 1):
        idx = np.where(y_model == ym)[0]
        if len(idx) > 1:
            shuffled = rng.permutation(out[idx])
            out[idx] = shuffled
    return out


def variant_shuffled_unconditionally(records, y_model, rng):
    """
    INTUITION: destroy all structure including the y_expert/y_model
    relationship. This is a stronger null than 'shuffled_within' -- if the
    gap performs identically, the variable carries no information at all.
    """
    y_exp = np.array([float(r.expert_reliable) for r in records], dtype=np.float32)
    return rng.permutation(y_exp)


def variant_constant_half(records, y_model, rng):
    """
    INTUITION: collapses the second head. SCOD theory says gap should
    degenerate to scaled probe-only. This is the theoretical baseline.
    """
    return np.full(len(records), 0.5, dtype=np.float32)


def variant_inverse_y_model_oracle(records, y_model, rng):
    """
    INTUITION: upper bound. Maximally informative y_expert -- perfect
    anti-correlation with model correctness. If gap doesn't beat probe-only
    even here, something is wrong with the probe or the eval.
    """
    return 1.0 - y_model.astype(np.float32)


def variant_topic_indicator(records, y_model, rng):
    """
    INTUITION: defends S4 ('f_defer is just a topic classifier on TruthfulQA').
    Replaces y_expert with the per-category empirical y_model rate -- a
    "topic difficulty proxy". If the gap result with this variant ~ gap result
    with native y_expert, you've shown the collapse mechanism IS topic.

    For datasets without categorical topic, this returns None -- caller should
    skip.
    """
    cats = [r.category for r in records]
    if not cats or all(c == "" for c in cats):
        return None  # signal: skip this variant for this dataset

    cat_to_id = {c: i for i, c in enumerate(sorted(set(cats)))}
    # Map category to its empirical y_model rate
    rates = {}
    for c in cat_to_id:
        mask = np.array([r.category == c for r in records])
        rates[c] = float(y_model[mask].mean()) if mask.sum() > 0 else 0.5
    return np.array([rates[r.category] for r in records], dtype=np.float32)


VARIANTS: dict[str, Callable] = {
    "native":                      variant_native,
    "shuffled_within_y_model":     variant_shuffled_within_y_model,
    "shuffled_unconditionally":    variant_shuffled_unconditionally,
    "constant_half":               variant_constant_half,
    "inverse_y_model_oracle":      variant_inverse_y_model_oracle,
    "topic_indicator":             variant_topic_indicator,
}


# =============================================================================
# Probe training wrapper. Trains OVAHeads on (h_train, y_model, y_expert).
# =============================================================================

def train_variant_probes(
    h_train: np.ndarray,          # [2N, d]
    y_model_train: np.ndarray,    # [2N]
    y_expert_train: np.ndarray,   # [2N]
    seed: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    batch_size: int = 256,
    balance_classes: bool = True,
) -> OVAHeads:
    """
    Train OVAHeads with the given y_expert variant.

    y_expert_train is float in [0,1]; we binarize at 0.5 to match the
    existing TrainingData convention (int64 binary labels).
    """
    y_exp_binary = (y_expert_train >= 0.5).astype(np.int64)
    if len(np.unique(y_exp_binary)) < 2:
        # Constant variant -- f_defer has nothing to learn. We still train
        # but expect the defer head to be trivial.
        log.warning("y_expert is constant after binarization; f_defer will be trivial.")

    from probes.training import TrainingData
    data = TrainingData(
        h           = torch.from_numpy(h_train).float(),
        y_model     = torch.from_numpy(y_model_train.astype(np.int64)),
        y_expert    = torch.from_numpy(y_exp_binary),
        example_ids = np.array(["dummy"] * len(h_train)),
    )

    cfg = OVATrainConfig(
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        balance_classes=balance_classes,
    )
    model = train_ova(data, cfg=cfg, seed=seed)
    return model


def scores_from_model(
    model: OVAHeads,
    h_eval: np.ndarray,            # [N, d]
) -> dict[str, np.ndarray]:
    """Extract probe_pred_only, probe_defer_only, and gap scores."""
    configure_torch_threads()
    device = get_torch_device()
    model = model.to(device).eval()

    h_t = torch.from_numpy(h_eval).float().to(device)
    with torch.no_grad():
        lp, ld = model(h_t)
        lp = lp.cpu().numpy()
        ld = ld.cpu().numpy()

    return {
        "probe_pred_only":   lp,      # logit_pred -- higher = more confident
        "probe_defer_only":  ld,      # logit_defer
        "gap":               lp - ld, # the paper's score
    }


# =============================================================================
# Main audit loop
# =============================================================================

def run_audit(
    model_key: str,
    dataset: str,
    seed: int = 0,
    halueval_style_labels: bool = False,
    ova_train: dict | None = None,
    out_dir: Path = Path("results/y_expert_audit"),
    seeds: list[int] | None = None,
):
    """
    Run the semantic audit for one (model, dataset) pair.

    Uses the existing hidden_states.pt cache (from step 1) and trains fresh
    OVA probes per variant per seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ova_cfg = ova_train or {}

    # -- Load cached hidden states from step 1 --------------------------------
    h_path = hidden_states_path(model_key, dataset)
    if not h_path.exists():
        raise FileNotFoundError(
            f"Run step 1 first: hidden states missing at {h_path}"
        )
    blob = torch.load(h_path, map_location="cpu", weights_only=False)
    h_pos = np.asarray(blob["h_pos"])   # [N, d]
    h_neg = np.asarray(blob["h_neg"])   # [N, d]
    extracted_ids = np.asarray(blob["example_ids"])
    N = len(h_pos)

    # -- Load records for expert_reliable ------------------------------------
    records = load_dataset(dataset)
    er_by_id = {r.example_id: r.expert_reliable for r in records}
    expert_reliable = np.array(
        [er_by_id[eid] for eid in extracted_ids], dtype=bool,
    )

    # Build stacked training arrays (same as prepare_training_data)
    h_train = np.concatenate([h_pos, h_neg], axis=0)   # [2N, d]
    y_model_train = np.concatenate([
        np.ones(N, dtype=np.int64),
        np.zeros(N, dtype=np.int64),
    ])

    # -- Build y_correct for evaluation ---------------------------------------
    # Use generation-based judge labels when available. If not, fall back to the
    # same contrastive h+/h- evaluation convention as scripts/step4_evaluate.py.
    judge_path = Path(out_dir).parent / "step4_eval" / model_key / dataset / f"seed{seed}"
    judge_file = judge_path / "judge_labels.pt"
    if judge_file.exists():
        judge_blob = torch.load(judge_file, map_location="cpu", weights_only=False)
        y_correct = judge_blob["labels"].astype(int)
        h_eval = h_pos[:len(y_correct)]
        log.info(f"Loaded judge labels: {y_correct.sum()}/{len(y_correct)} correct")
    else:
        y_correct = y_model_train
        h_eval = h_train
        log.warning(
            f"No judge labels at {judge_file}. Using contrastive h+/h- "
            f"correctness fallback."
        )

    # Records for eval (aligned with h_pos, not h_train)
    records_eval = [r for r in records if r.example_id in set(extracted_ids)]

    # -- Run audit over variants and seeds ------------------------------------
    audit_seeds = seeds if seeds else [seed]

    results = []
    for variant_name, builder in VARIANTS.items():
        for s in audit_seeds:
            set_global_seed(s)
            rng = np.random.default_rng(s)

            # Build y_expert from records (N records -> duplicate for 2N)
            y_exp_records = builder(records_eval, y_correct[:len(records_eval)], rng)
            if y_exp_records is None:
                continue  # variant doesn't apply

            # Expand to 2N: same y_expert for h^+ and h^- rows
            y_exp_train_full = np.concatenate([y_exp_records, y_exp_records])
            assert len(y_exp_train_full) == len(y_model_train)

            # Train probes
            model = train_variant_probes(
                h_train, y_model_train, y_exp_train_full,
                seed=s, **ova_cfg,
            )

            # Score on the selected evaluation set.
            scores = scores_from_model(model, h_eval)

            for method, s_arr in scores.items():
                a = aurc(score=s_arr[:len(y_correct)], correct=y_correct)
                c80 = coverage_at_accuracy(
                    score=s_arr[:len(y_correct)], correct=y_correct,
                    target_accuracy=0.80,
                )
                c90 = coverage_at_accuracy(
                    score=s_arr[:len(y_correct)], correct=y_correct,
                    target_accuracy=0.90,
                )
                results.append({
                    "model":    model_key,
                    "dataset":  dataset,
                    "variant":  variant_name,
                    "method":   method,
                    "seed":     s,
                    "aurc":     a,
                    "cov_at_80": c80,
                    "cov_at_90": c90,
                })

    df = pd.DataFrame(results)
    out_file = out_dir / f"{model_key}__{dataset}.parquet"
    df.to_parquet(out_file, index=False)
    print(f"Wrote {out_file} ({len(df)} rows)")

    # -- Headline summary table ----------------------------------------------
    if len(df) > 0:
        summary = (
            df.groupby(["variant", "method"])["aurc"]
            .agg(["mean", "std"])
            .round(4)
        )
        print("\n=== AURC by variant x method (mean +/- std over seeds) ===")
        print(summary)

    # -- Pre-registered prediction check -------------------------------------
    # Showcase datasets should show: native gap < shuffled gap in AURC
    # (lower AURC = better)
    showcase = dataset in ("truthfulqa", "halueval_qa")
    if showcase and len(df) > 0:
        native_gap = df.query("variant=='native' and method=='gap'")["aurc"].mean()
        shuffled_gap = df.query(
            "variant=='shuffled_within_y_model' and method=='gap'"
        )["aurc"].mean()
        pred_only = df.query(
            "variant=='native' and method=='probe_pred_only'"
        )["aurc"].mean()

        if not np.isnan(native_gap) and not np.isnan(shuffled_gap):
            delta_native = pred_only - native_gap
            delta_shuffled = pred_only - shuffled_gap
            print(f"\nShowcase prediction check ({dataset}):")
            print(f"  native gap AURC:              {native_gap:.4f}")
            print(f"  shuffled-within gap AURC:     {shuffled_gap:.4f}")
            print(f"  probe-only AURC:              {pred_only:.4f}")
            print(
                f"  delta (probe_only - gap), native:    {delta_native:+.4f}  "
                f"{'PASS: gap wins' if delta_native > 0.01 else 'FAIL: gap does not clearly win'}"
            )
            print(
                f"  delta (probe_only - gap), shuffled:  {delta_shuffled:+.4f}  "
                f"{'PASS: collapses as expected' if abs(delta_shuffled) < 0.01 else 'FAIL: unexpected'}"
            )

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="y_expert semantic audit")
    p.add_argument("--config", required=True, help="YAML config for model+dataset")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seed list (default: use config seed)")
    p.add_argument("--out-dir", default="results/y_expert_audit")
    args = p.parse_args()

    cfg = load_config(args.config)
    assert_required(cfg, ["model_key", "dataset", "seed", "halueval_style_labels"])

    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds
        else [0, 1, 2, 3, 4]
    )

    run_audit(
        model_key           = cfg.model_key,
        dataset             = cfg.dataset,
        seed                = cfg.seed,
        halueval_style_labels = cfg.halueval_style_labels,
        ova_train           = cfg.get("ova_train"),
        out_dir             = Path(args.out_dir),
        seeds               = seeds,
    )


if __name__ == "__main__":
    main()
