"""
SCOD diagnostics
================

For each (model, dataset), measure how 2D the (p_pred, p_defer) joint
distribution really is. If it's effectively 1D, SCOD theory predicts the
optimal selector collapses to probe-only -- which is what we observe on
TruthfulQA/HaluEval and (predicted-not) on ChaosNLI/AmbigQA/Pavlick.

Three measurements:
  1. Effective dimensionality of (p_pred, p_defer) -- PCA on the 2D point cloud.
  2. Mutual information I(p_pred ; p_defer) -- how much the second coordinate
     adds beyond the first.
  3. Conditional variance Var(p_defer | p_pred) -- does p_defer still
     vary at fixed p_pred? If not, no information beyond probe-only.

The output is a single table that, for each dataset, says:
  - "predicted to collapse" (high coupling, low conditional variance) -> diagnostic
  - "predicted to separate" (genuine 2D structure) -> showcase

This is the empirical defense of the SCOD framing in section 2.4 of the paper.

Usage:
  python scripts/scod_diagnostics.py \
      --config configs/experiments/llama3_8b_truthfulqa.yaml
  python scripts/scod_diagnostics.py \
      --model llama3_8b --datasets truthfulqa halueval_qa triviaqa
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

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
# Core diagnostic functions
# =============================================================================

@dataclass
class SCODDiagnostic:
    dataset: str
    model: str
    n: int

    # 2D structure metrics
    pca_var_ratio_1: float        # variance captured by 1st PC. >=0.95 means ~1D.
    pca_var_ratio_2: float        # variance captured by 2nd PC
    effective_dim: float          # 1/sum(p_i^2) -- Shannon-style soft dim count

    # Coupling between coordinates
    pearson_r: float
    spearman_r: float
    mutual_info_bits: float       # I(p_pred ; p_defer) in bits

    # Conditional structure (the SCOD-relevant quantity)
    cond_var_defer_given_pred: float       # mean of Var(p_defer | bin(p_pred))
    cond_var_pred_given_defer: float       # symmetric

    # Direct prediction
    predicted_regime: str        # "collapse" or "separate"


def compute_scod_diagnostics(
    p_pred: np.ndarray,
    p_defer: np.ndarray,
    dataset: str,
    model: str,
    n_bins: int = 10,
) -> SCODDiagnostic:
    """
    INTUITION: each diagnostic answers a slightly different question about
    whether the 2D space (p_pred, p_defer) genuinely needs both axes.

    Why three diagnostics, not one:
      - PCA tells us if the cloud lies on a line in (p_pred, p_defer) space.
        But a curved 1D manifold would still register as 2D under PCA.
      - Mutual information catches nonlinear coupling that PCA misses.
      - Conditional variance is the *operational* quantity: at a fixed value
        of p_pred, how much does p_defer vary? If it doesn't, the SCOD-optimal
        selector cannot use p_defer to make finer-grained decisions.

    A dataset is predicted to "collapse" iff:
      (effective_dim < 1.15) OR (cond_var_defer < 0.005)
    The threshold values are eyeballed; tune on TruthfulQA where we know
    the answer empirically. Lock these BEFORE running showcase datasets.
    """
    pp = np.asarray(p_pred, dtype=np.float64)
    pd_ = np.asarray(p_defer, dtype=np.float64)
    assert pp.shape == pd_.shape and pp.ndim == 1

    X = np.stack([pp, pd_], axis=1)

    # -- PCA on the 2D point cloud -------------------------------------------
    Xc = X - X.mean(axis=0)
    pca = PCA(n_components=2).fit(Xc)
    var_ratio = pca.explained_variance_ratio_
    # Soft effective dimension via participation ratio of the eigenvalues
    eigs = pca.explained_variance_
    eff_dim = (eigs.sum() ** 2) / (eigs ** 2).sum()

    # -- Correlations --------------------------------------------------------
    pearson_r, _ = pearsonr(pp, pd_)
    spearman_r, _ = spearmanr(pp, pd_)

    # -- Mutual information --------------------------------------------------
    # mutual_info_regression treats one variable as continuous target.
    mi_nats = mutual_info_regression(
        pp.reshape(-1, 1), pd_, n_neighbors=5, random_state=0,
    )[0]
    mi_bits = mi_nats / np.log(2)

    # -- Conditional variance ------------------------------------------------
    # Bin one axis, compute variance of the other within each bin, average.
    cond_var_defer = _binned_conditional_variance(pp, pd_, n_bins)
    cond_var_pred = _binned_conditional_variance(pd_, pp, n_bins)

    # -- Regime prediction ---------------------------------------------------
    regime = (
        "collapse"
        if (eff_dim < 1.15) or (cond_var_defer < 0.005)
        else "separate"
    )

    return SCODDiagnostic(
        dataset=dataset,
        model=model,
        n=len(pp),
        pca_var_ratio_1=float(var_ratio[0]),
        pca_var_ratio_2=float(var_ratio[1]),
        effective_dim=float(eff_dim),
        pearson_r=float(pearson_r),
        spearman_r=float(spearman_r),
        mutual_info_bits=float(mi_bits),
        cond_var_defer_given_pred=float(cond_var_defer),
        cond_var_pred_given_defer=float(cond_var_pred),
        predicted_regime=regime,
    )


def _binned_conditional_variance(
    x: np.ndarray, y: np.ndarray, n_bins: int,
) -> float:
    """
    Var(y | bin(x)), averaged over bins weighted by bin count.

    INTUITION: at each level of x, how much does y move? This is what the
    SCOD-optimal selector cares about -- it can only exploit information
    in y that varies within the strata of x already used by f_pred.
    """
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = np.digitize(x, edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    total_n = len(x)
    weighted_var = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() < 2:
            continue
        weighted_var += (mask.sum() / total_n) * y[mask].var()
    return weighted_var


# =============================================================================
# Data loading: reuse the same loading pattern as step2 / y_expert_semantic_audit
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
    (p_pred, p_defer) probabilities on the evaluation set.

    Returns:
        p_pred:    (N,) sigma(f_pred) on h^+
        p_defer:   (N,) sigma(f_defer) on h^+
        y_correct: (N,) correctness labels (from judge if available)
        records:   list[Record] aligned with p_pred/p_defer
    """
    import torch

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

    # Load records for expert_reliable
    records = load_dataset(dataset)
    er_by_id = {r.example_id: r.expert_reliable for r in records}
    expert_reliable = np.array(
        [er_by_id[eid] for eid in extracted_ids], dtype=bool,
    )

    # Build stacked training arrays
    h_train = np.concatenate([h_pos, h_neg], axis=0)
    y_model_train = np.concatenate([
        np.ones(N, dtype=np.int64),
        np.zeros(N, dtype=np.int64),
    ])

    # Prepare training data
    data = prepare_training_data(
        h_pos=h_pos, h_neg=h_neg,
        expert_reliable=expert_reliable,
        example_ids=extracted_ids,
        halueval_style=halueval_style_labels,
    )

    # Train probes
    ova_cfg = OVATrainConfig(**(ova_train or {}))
    set_global_seed(seed)
    model = train_ova(data, cfg=ova_cfg, seed=seed)

    # Extract (logit_pred, logit_defer) on h_pos, then sigmoid
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    h_t = torch.from_numpy(h_pos).float().to(device)
    with torch.no_grad():
        lp, ld = model(h_t)
    lp = torch.sigmoid(lp).cpu().numpy()   # sigma(f_pred)
    ld = torch.sigmoid(ld).cpu().numpy()   # sigma(f_defer)

    # y_correct: try to load judge labels, fall back to all-ones placeholder
    judge_path = (
        Path("outputs") / "step4_eval" / model_key / dataset
        / f"seed{seed}" / "judge_labels.pt"
    )
    if judge_path.exists():
        judge_blob = torch.load(judge_path, map_location="cpu",
                                weights_only=False)
        y_correct = judge_blob["labels"].astype(int)
        log.info(f"Loaded judge labels: {y_correct.sum()}/{len(y_correct)} correct")
    else:
        y_correct = np.ones(N, dtype=int)
        log.warning("No judge labels found; using all-ones placeholder.")

    records_eval = [r for r in records if r.example_id in set(extracted_ids)]
    return lp, ld, y_correct[:len(lp)], records_eval


# =============================================================================
# Per-dataset entry point
# =============================================================================

def run_for_dataset(
    model_key: str,
    dataset: str,
    seed: int = 0,
    halueval_style_labels: bool = False,
    ova_train: dict | None = None,
) -> SCODDiagnostic:
    """Train probes, extract (p_pred, p_defer), run diagnostics."""
    p_pred, p_defer, y_correct, records = load_probes_and_scores(
        model_key, dataset, seed=seed,
        halueval_style_labels=halueval_style_labels,
        ova_train=ova_train,
    )
    return compute_scod_diagnostics(p_pred, p_defer, dataset, model_key)


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="SCOD diagnostics: measure 2D structure of (p_pred, p_defer)")
    p.add_argument("--config", default=None,
                   help="YAML config for single model+dataset")
    p.add_argument("--model", default=None,
                   help="Model key (alternative to --config)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Datasets to run (alternative to --config)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/scod_diagnostics.parquet")
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config)
        assert_required(cfg, ["model_key", "dataset", "seed", "halueval_style_labels"])
        model_key = cfg.model_key
        datasets = [cfg.dataset]
        seed = cfg.seed
        halueval_style_labels = cfg.halueval_style_labels
        ova_train = cfg.get("ova_train")
    elif args.model and args.datasets:
        model_key = args.model
        datasets = args.datasets
        seed = args.seed
        halueval_style_labels = False
        ova_train = None
    else:
        p.error("Provide either --config or both --model and --datasets")

    rows = []
    for d in datasets:
        diag = run_for_dataset(
            model_key, d, seed=seed,
            halueval_style_labels=halueval_style_labels,
            ova_train=ova_train,
        )
        rows.append(diag.__dict__)
        print(f"{d:15s}  eff_dim={diag.effective_dim:.3f}  "
              f"MI={diag.mutual_info_bits:.3f}b  "
              f"cond_var={diag.cond_var_defer_given_pred:.4f}  "
              f"-> {diag.predicted_regime}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
