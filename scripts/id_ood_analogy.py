"""
ID/OOD analogy
==============

This script is CAREFUL. The framing doc section 2.3 says: "No claim that
f_defer is a formal OOD detector (analogy only)." So this script doesn't
claim f_defer IS an OOD detector. It tests whether f_defer BEHAVES LIKE
one in specific operational senses.

Three operational tests:

  1. Three-rank correlation: does f_defer's threshold-induced ranking
     correlate with an actual OOD score (e.g., Mahalanobis distance to
     training-data hidden states)? If yes, the analogy is empirically
     supported; if no, the framing should be removed.

  2. Coverage-conditional risk: bucket test items by f_defer level. If
     low-f_defer buckets have systematically higher model error, f_defer
     is acting OOD-like (filtering hard items). This is exactly what
     selective-prediction risk-coverage curves measure.

  3. Cross-dataset transfer: train f_defer on one dataset, evaluate on
     another. If f_defer assigns systematically lower scores to the
     out-of-distribution dataset, it has captured something that
     transfers -- which an OOD detector would.

The script outputs a single section the paper can use: "we empirically
characterize f_defer's behavior in three OOD-style senses."

Usage:
  python scripts/id_ood_analogy.py \
      --config configs/experiments/llama3_8b_truthfulqa.yaml \
      --eval-datasets truthfulqa halueval_qa triviaqa
  python scripts/id_ood_analogy.py \
      --model llama3_8b --train-dataset truthfulqa \
      --eval-datasets truthfulqa halueval_qa triviaqa popqa bioasq
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config, assert_required
from utils.logging import get_logger
from utils.paths import hidden_states_path
from utils.seed import set_global_seed
from data.registry import load_dataset
from probes.ova import OVAHeads, OVATrainConfig
from probes.training import prepare_training_data, train_ova

log = get_logger(__name__)


# =============================================================================
# Data loading (same pattern as scod_diagnostics.py)
# =============================================================================

def load_probes_and_eval(
    model_key: str,
    dataset: str,
    seed: int = 0,
    halueval_style_labels: bool = False,
    ova_train: dict | None = None,
):
    """
    Load cached hidden states, train OVA probes, return probe model and
    all data needed for evaluation.

    Returns:
        model:            trained OVAHeads
        h_pos:            (N, d) h^+ hidden states
        h_neg:            (N, d) h^- hidden states
        y_model_train:    (2N,) binary labels
        y_correct:        (N,) correctness labels for eval
        records_eval:     list[Record] aligned with h_pos
    """
    import torch

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

    records = load_dataset(dataset)
    er_by_id = {r.example_id: r.expert_reliable for r in records}
    expert_reliable = np.array(
        [er_by_id[eid] for eid in extracted_ids], dtype=bool,
    )

    h_train = np.concatenate([h_pos, h_neg], axis=0)
    y_model_train = np.concatenate([
        np.ones(N, dtype=np.int64),
        np.zeros(N, dtype=np.int64),
    ])

    data = prepare_training_data(
        h_pos=h_pos, h_neg=h_neg,
        expert_reliable=expert_reliable,
        example_ids=extracted_ids,
        halueval_style=halueval_style_labels,
    )

    ova_cfg = OVATrainConfig(**(ova_train or {}))
    set_global_seed(seed)
    model = train_ova(data, cfg=ova_cfg, seed=seed)

    # y_correct from judge labels if available
    judge_path = (
        Path("outputs") / "step4_eval" / model_key / dataset
        / f"seed{seed}" / "judge_labels.pt"
    )
    if judge_path.exists():
        judge_blob = torch.load(judge_path, map_location="cpu",
                                weights_only=False)
        y_correct = judge_blob["labels"].astype(int)
    else:
        y_correct = np.ones(N, dtype=int)
        log.warning("No judge labels found; using all-ones placeholder.")

    records_eval = [r for r in records if r.example_id in set(extracted_ids)]
    return model, h_pos, h_neg, y_model_train, y_correct[:N], records_eval


def extract_p_defer(model: OVAHeads, h_eval: np.ndarray) -> np.ndarray:
    """Extract sigma(f_defer) probabilities from a trained model."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    h_t = torch.from_numpy(h_eval).float().to(device)
    with torch.no_grad():
        _, ld = model(h_t)
    return torch.sigmoid(ld).cpu().numpy()


# =============================================================================
# Test 1: agreement with Mahalanobis distance
# =============================================================================

def mahalanobis_score(
    h_eval: np.ndarray,
    h_train_correct: np.ndarray,
) -> np.ndarray:
    """
    Classic OOD-detection score: distance from the *correctly-predicted*
    training distribution. Lower = more ID = should commit.

    INTUITION: if f_defer is OOD-like, items far from the correct-training
    cluster (high Mahalanobis) should get LOW f_defer scores (don't trust
    the expert here either, because we're far from where we've seen the
    expert be reliable).

    The signed correlation between f_defer and Mahalanobis tells us whether
    f_defer's geometry matches an OOD detector's.
    """
    lw = LedoitWolf().fit(h_train_correct)
    mu = h_train_correct.mean(axis=0)
    diff = h_eval - mu
    prec = lw.precision_
    md = np.einsum("ni,ij,nj->n", diff, prec, diff)
    return np.sqrt(np.clip(md, 0, None))


def test_1_mahalanobis_agreement(
    h_train_correct: np.ndarray,
    h_eval: np.ndarray,
    p_defer: np.ndarray,
) -> dict:
    """Test whether f_defer correlates (negatively) with Mahalanobis distance."""
    md = mahalanobis_score(h_eval, h_train_correct)
    rho, p = spearmanr(md, p_defer)
    # Expect negative correlation: high Mahalanobis (OOD) <-> low f_defer (don't trust)
    return {
        "spearman_rho": float(rho),
        "p_value":      float(p),
        "behaves_ood_like": bool((rho < -0.2) and (p < 0.05)),
    }


# =============================================================================
# Test 2: coverage-conditional risk
# =============================================================================

def test_2_coverage_conditional_risk(
    p_defer: np.ndarray,
    y_correct: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Bucket items by f_defer level. Report mean error per bucket.

    INTUITION: an OOD-like score should produce monotonically decreasing
    error as f_defer increases. f_defer = 0 should look like "OOD items"
    where the model performs poorly. f_defer = 1 should look like "well
    within distribution" where the model performs well.

    If the curve is non-monotonic or flat, f_defer is doing something
    OTHER than OOD detection.
    """
    edges = np.quantile(p_defer, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-9
    buckets = np.digitize(p_defer, edges) - 1
    buckets = np.clip(buckets, 0, n_buckets - 1)

    rows = []
    for b in range(n_buckets):
        mask = buckets == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket":     b,
            "n":          int(mask.sum()),
            "mean_defer": float(p_defer[mask].mean()),
            "mean_error": float(1 - y_correct[mask].mean()),
        })
    df = pd.DataFrame(rows)

    # Monotonicity test: Spearman between bucket mean_defer and mean_error
    if len(df) >= 3:
        rho, p = spearmanr(df["mean_defer"], df["mean_error"])
        df.attrs["monotonicity_rho"] = float(rho)
        df.attrs["monotonicity_p"] = float(p)
        df.attrs["behaves_ood_like"] = bool((rho < -0.5) and (p < 0.05))
    else:
        df.attrs["monotonicity_rho"] = float("nan")
        df.attrs["monotonicity_p"] = float("nan")
        df.attrs["behaves_ood_like"] = False

    return df


# =============================================================================
# Test 3: cross-dataset transfer
# =============================================================================

def test_3_cross_dataset_transfer(
    probes_trained_on: str,
    evaluated_on: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    f_defer trained on dataset A is applied to hidden states from datasets
    A, B, C, ...

    INTUITION: if f_defer captures "expert reliability" as a property of
    hidden-state geometry that transfers, items from a different distribution
    should get systematically lower f_defer scores (treated as OOD relative
    to the training distribution).

    If f_defer scores are identical across datasets, the head learned a
    dataset-specific marker that doesn't transfer -- undermining the
    OOD-analogy framing.

    `evaluated_on` is {dataset_name: f_defer_scores_on_that_dataset}.
    """
    rows = []
    for dname, scores in evaluated_on.items():
        rows.append({
            "trained_on":   probes_trained_on,
            "evaluated_on": dname,
            "n":            len(scores),
            "mean_p_defer": float(np.mean(scores)),
            "std_p_defer":  float(np.std(scores)),
        })
    df = pd.DataFrame(rows)

    own = df.query("evaluated_on == @probes_trained_on")["mean_p_defer"].iloc[0]
    df["delta_vs_own"] = df["mean_p_defer"] - own
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="ID/OOD analogy tests for f_defer",
    )
    p.add_argument("--config", default=None,
                   help="YAML config (model + train-dataset)")
    p.add_argument("--model", default=None)
    p.add_argument("--train-dataset", default=None,
                   help="Dataset on which f_defer is trained")
    p.add_argument("--eval-datasets", nargs="+", default=None,
                   help="Datasets to evaluate on (incl. train-dataset for self-comparison)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/id_ood_analogy")
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config)
        assert_required(cfg, ["model_key", "dataset", "seed", "halueval_style_labels"])
        model_key = cfg.model_key
        train_dataset = cfg.dataset
        seed = cfg.seed
        halueval_style_labels = cfg.halueval_style_labels
        ova_train = cfg.get("ova_train")
    elif args.model and args.train_dataset:
        model_key = args.model
        train_dataset = args.train_dataset
        seed = args.seed
        halueval_style_labels = False
        ova_train = None
    else:
        p.error("Provide either --config or both --model and --train-dataset")

    eval_datasets = args.eval_datasets or [train_dataset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Train probes on the training dataset ---------------------------------
    print(f"\nTraining probes on {model_key}/{train_dataset}")
    model, h_pos, h_neg, y_model_train, y_correct, records_eval = \
        load_probes_and_eval(
            model_key, train_dataset, seed=seed,
            halueval_style_labels=halueval_style_labels,
            ova_train=ova_train,
        )

    # h_train_correct: h^+ rows where model would be correct
    h_train_correct = h_pos  # h^+ corresponds to correct-answer representations

    # -- Test 1: Mahalanobis agreement on own dataset -------------------------
    p_defer_own = extract_p_defer(model, h_pos)
    t1 = test_1_mahalanobis_agreement(h_train_correct, h_pos, p_defer_own)
    print("\nTest 1 -- Mahalanobis vs f_defer (Spearman rho):")
    print(f"  rho={t1['spearman_rho']:+.3f}  p={t1['p_value']:.4f}  "
          f"OOD-like behavior: {t1['behaves_ood_like']}")

    # -- Test 2: coverage-conditional risk on own dataset ---------------------
    t2 = test_2_coverage_conditional_risk(p_defer_own, y_correct)
    print("\nTest 2 -- Coverage-conditional risk (bucketed by f_defer):")
    print(t2.to_string(index=False))
    print(f"  Monotonicity rho={t2.attrs['monotonicity_rho']:+.3f}  "
          f"OOD-like: {t2.attrs['behaves_ood_like']}")

    # -- Test 3: cross-dataset transfer ---------------------------------------
    evaluated_on = {train_dataset: p_defer_own}
    for d in eval_datasets:
        if d == train_dataset:
            continue
        # Load hidden states for the other dataset (probe model stays the same)
        try:
            other_h_path = hidden_states_path(model_key, d)
            if not other_h_path.exists():
                print(f"  [skip] {d}: no hidden states at {other_h_path}")
                continue
            other_blob = __import__("torch").load(
                other_h_path, map_location="cpu", weights_only=False,
            )
            h_pos_other = np.asarray(other_blob["h_pos"])
            evaluated_on[d] = extract_p_defer(model, h_pos_other)
        except Exception as e:
            print(f"  [error] {d}: {e}")
            continue

    t3 = test_3_cross_dataset_transfer(train_dataset, evaluated_on)
    print("\nTest 3 -- Cross-dataset transfer of f_defer:")
    print(t3.to_string(index=False))

    # -- Save results ---------------------------------------------------------
    t1_df = pd.DataFrame([t1])
    t1_df.to_csv(out_dir / f"test1_mahalanobis_{model_key}__{train_dataset}.csv",
                 index=False)

    t2.to_csv(out_dir / f"test2_coverage_risk_{model_key}__{train_dataset}.csv",
              index=False)

    t3.to_csv(out_dir / f"test3_cross_dataset_{model_key}__{train_dataset}.csv",
              index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results saved to {out_dir}/")
    print(f"  Test 1 (Mahalanobis): OOD-like = {t1['behaves_ood_like']}")
    print(f"  Test 2 (Coverage):    OOD-like = {t2.attrs['behaves_ood_like']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
