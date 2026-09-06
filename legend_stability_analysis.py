"""Compare question-level report selection under two independently elicited legends.

No forward pass is run.  The canonical letter11-v1 cache is compared with the
independently elicited reversed-legend cache.  Scores are always computed in
each cache's denoted-value space, and the existing mean-logprob preference
target is retained as a question-level diagnostic rather than called answer
correctness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ladder import _split
from scripts.step_token_contribution import aurc
from utils.paths import logprobs_path

MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
DATASETS = ("truthfulqa", "pavlick_nli")
SEEDS = (0, 1, 2)
TOP_FRAC = 0.20
TIE_TOL = 1e-12
BOOT = 2000
BOOT_SEED = 20260905
PERM_ROOT = Path(__file__).resolve().parent / "letter11_permuted_remote"
OUT = Path(__file__).resolve().parent / "cached_results" / "legend_stability"


def _cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _readouts(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"Vdist_pos", "Vdist_neg", "Vmass_pos", "Vmass_neg",
                "conf_values", "example_ids"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"cache is missing {sorted(missing)}")
    pos = np.asarray(data["Vdist_pos"], dtype=float)
    neg = np.asarray(data["Vdist_neg"], dtype=float)
    values = np.asarray(data["conf_values"], dtype=float) / 100.0
    if pos.shape != neg.shape or pos.ndim != 2 or pos.shape[1] != len(values):
        raise ValueError("incompatible Vdist/conf_values shapes")
    if (not np.isfinite(pos).all() or not np.isfinite(neg).all()
            or not np.isfinite(values).all() or (pos < 0).any() or (neg < 0).any()
            or (pos.sum(axis=1) <= 0).any() or (neg.sum(axis=1) <= 0).any()):
        raise ValueError("invalid confidence distribution")
    if len(data["example_ids"]) != len(pos):
        raise ValueError("item IDs and distributions differ in length")
    # Match distribution_readouts(): conditional on the available alphabet,
    # while retaining the original alphabet mass as a separate diagnostic.
    pos_cond = pos / pos.sum(axis=1, keepdims=True)
    neg_cond = neg / neg.sum(axis=1, keepdims=True)
    vp = pos_cond @ values
    vn = neg_cond @ values
    ids = np.asarray(data["example_ids"], dtype=object)
    meta = data.get("meta") or {}
    # The isolated permuted runner softmaxes over candidate letters only and
    # writes Vmass=1 as a placeholder.  That is conditional mass, not the
    # full-vocabulary availability diagnostic, so expose it as unavailable.
    has_true_mass = (meta.get("conf_scheme") != "letter11-permuted"
                     or meta.get("mass_semantics") == "full_vocabulary_probability_of_candidate_tokens")
    if has_true_mass:
        mass_p = np.asarray(data["Vmass_pos"], dtype=float)
        mass_n = np.asarray(data["Vmass_neg"], dtype=float)
    else:
        mass_p = np.full(len(ids), np.nan)
        mass_n = np.full(len(ids), np.nan)
    return vp, vn, ids, mass_p, mass_n


def _align(ids_a, ids_b) -> np.ndarray:
    if len(set(map(str, ids_a))) != len(ids_a) or len(set(map(str, ids_b))) != len(ids_b):
        raise ValueError("duplicate item IDs")
    index = {str(x): i for i, x in enumerate(ids_b)}
    if {str(x) for x in ids_a} != set(index):
        raise ValueError("forward and reverse item IDs do not match")
    return np.asarray([index[str(x)] for x in ids_a], dtype=int)


def _top_mask(scores: np.ndarray, ids: np.ndarray, frac: float) -> tuple[np.ndarray, bool]:
    n = len(scores)
    if n == 0 or not 0 < frac <= 1 or not np.isfinite(scores).all():
        raise ValueError("invalid selection inputs")
    k = max(1, int(np.ceil(frac * n)))
    # Descending score, then stable item ID.  The item-ID tie break is shared
    # across legends and does not inspect the target.
    order = np.lexsort((np.asarray([str(x) for x in ids]), -scores))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    threshold = scores[order[k - 1]]
    boundary_tie = bool(k < n and scores[order[k]] == threshold)
    return mask, boundary_tie


def _stable_aurc(scores, risk, ids):
    order = np.lexsort((np.asarray(list(map(str, ids))), -scores))
    return float(np.mean(np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)))


def _joint_bootstrap(scores_f, scores_r, risk, ids, rng):
    n = len(risk)
    draws = rng.integers(0, n, size=(BOOT, n))
    out = []
    for ix in draws:
        sf, sr, rr = scores_f[ix], scores_r[ix], risk[ix]
        mf, _ = _top_mask(sf, ids[ix], TOP_FRAC)
        mr, _ = _top_mask(sr, ids[ix], TOP_FRAC)
        out.append((rr[mf].mean(), rr[mr].mean(), _stable_aurc(sf, rr, ids[ix]), _stable_aurc(sr, rr, ids[ix]), mf[mr].sum() / mf.sum()))
    return np.asarray(out)


def main():
    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for model in MODELS:
        for dataset in DATASETS:
            cell = ROOT / "outputs" / "step1_extract" / model / dataset
            forward_data = _cache(cell / "logprobs.pt")
            reverse_data = _cache(PERM_ROOT / f"{model}_{dataset}.pt")
            f_vp, f_vn, f_ids, f_mass_p, f_mass_n = _readouts(forward_data)
            r_vp, r_vn, r_ids, r_mass_p, r_mass_n = _readouts(reverse_data)
            reverse_order = _align(f_ids, r_ids)
            r_vp, r_vn = r_vp[reverse_order], r_vn[reverse_order]
            r_mass_p, r_mass_n = r_mass_p[reverse_order], r_mass_n[reverse_order]
            y = (np.asarray(forward_data["logprob_pos"], dtype=float)
                 > np.asarray(forward_data["logprob_neg"], dtype=float)).astype(int)
            risk = 1.0 - y
            gap_f, gap_r = f_vp - f_vn, r_vp - r_vn
            for seed in SEEDS:
                _, test = _split(len(y), seed)
                ids = f_ids[test]
                sf, sr, rr = gap_f[test], gap_r[test], risk[test]
                mf, tie_f = _top_mask(sf, ids, TOP_FRAC)
                mr, tie_r = _top_mask(sr, ids, TOP_FRAC)
                sign_f = np.where(sf > TIE_TOL, 1, np.where(sf < -TIE_TOL, -1, 0))
                sign_r = np.where(sr > TIE_TOL, 1, np.where(sr < -TIE_TOL, -1, 0))
                boot = _joint_bootstrap(sf, sr, rr, ids, rng)
                ci = np.percentile(boot[:, :4], [2.5, 97.5], axis=0)
                overlap = float(np.logical_and(mf, mr).sum() / mf.sum())
                union = np.logical_or(mf, mr).sum()
                rows.append({
                    "model": model, "dataset": dataset, "seed": seed,
                    "n_test": len(sf), "top_k": int(mf.sum()),
                    "target_definition": "1[logprob_pos_mean > logprob_neg_mean]",
                    "forward_score": "conditional V+ - V-; letter11-v1",
                    "reverse_score": "conditional V+ - V-; letter11-permuted",
                    "retained_overlap_fraction": overlap,
                    "overlap_ci_lo": float(np.percentile(boot[:, 4], 2.5)),
                    "overlap_ci_hi": float(np.percentile(boot[:, 4], 97.5)),
                    "retained_risk_reverse_minus_forward": float(rr[mr].mean() - rr[mf].mean()),
                    "risk_difference_ci_lo": float(np.percentile(boot[:, 1] - boot[:, 0], 2.5)),
                    "risk_difference_ci_hi": float(np.percentile(boot[:, 1] - boot[:, 0], 97.5)),
                    "sign_flip_fraction": float(np.mean(sign_f * sign_r == -1)),
                    "retained_jaccard": float(np.logical_and(mf, mr).sum() / union),
                    "forward_retained_risk": float(rr[mf].mean()),
                    "reverse_retained_risk": float(rr[mr].mean()),
                    "forward_aurc": float(_stable_aurc(sf, rr, ids)),
                    "reverse_aurc": float(_stable_aurc(sr, rr, ids)),
                    "forward_boundary_tie": tie_f, "reverse_boundary_tie": tie_r,
                    "boundary_tie_rate": float((tie_f + tie_r) / 2),
                    "positive_to_negative": int(np.sum((sign_f == 1) & (sign_r == -1))),
                    "negative_to_positive": int(np.sum((sign_f == -1) & (sign_r == 1))),
                    "forward_ties": int(np.sum(sign_f == 0)), "reverse_ties": int(np.sum(sign_r == 0)),
                    "forward_mass_pos_mean": float(f_mass_p[test].mean()),
                    "forward_mass_neg_mean": float(f_mass_n[test].mean()),
                    "reverse_mass_pos_mean": float(r_mass_p[test].mean()),
                    "reverse_mass_neg_mean": float(r_mass_n[test].mean()),
                    "risk_boot_lo_forward": float(ci[0, 0]), "risk_boot_hi_forward": float(ci[1, 0]),
                    "risk_boot_lo_reverse": float(ci[0, 1]), "risk_boot_hi_reverse": float(ci[1, 1]),
                    "aurc_boot_lo_forward": float(ci[0, 2]), "aurc_boot_hi_forward": float(ci[1, 2]),
                    "aurc_boot_lo_reverse": float(ci[0, 3]), "aurc_boot_hi_reverse": float(ci[1, 3]),
                })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_seed.csv", index=False)
    summary = (frame.groupby(["model", "dataset"], as_index=False)
               .agg({"retained_overlap_fraction": "mean", "retained_jaccard": "mean",
                     "forward_retained_risk": "mean", "reverse_retained_risk": "mean",
                     "forward_aurc": "mean", "reverse_aurc": "mean",
                     "positive_to_negative": "mean", "negative_to_positive": "mean",
                     "sign_flip_fraction": "mean", "retained_risk_reverse_minus_forward": "mean",
                     "forward_mass_pos_mean": "mean", "forward_mass_neg_mean": "mean",
                     "reverse_mass_pos_mean": "mean", "reverse_mass_neg_mean": "mean"}))
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "forward": "canonical letter11-v1 logprobs.pt",
        "reverse": "independently elicited letter11-permuted cache",
        "score": "conditional V+ - V- in each arm's denoted-value space",
        "mass": "forward Vmass reported; reverse full-vocabulary mass unavailable because legacy candidate-conditional cache was used",
        "target": "mean-logprob preference; not candidate correctness",
        "top_fraction": TOP_FRAC, "tie_tolerance": TIE_TOL,
        "bootstrap": {"unit": "test question", "draws": BOOT, "seed": BOOT_SEED},
        "status": "cached analysis; no forward pass; legacy letter11-permuted cache explicitly requested",
        "summary_counts": "mean per split; overlapping test sets are not independent replications",
        "selection_ties": "exact score ties broken by item ID; boundary tie means tied across cutoff",
    }, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
