"""Cache-backed robustness checks for the report-channel comparison.

All checks use the canonical three seeds and held-out splits.  No model forward
pass is needed: ``probe_readout`` fits a three-feature logistic regression to
cached V values, and every other check is a deterministic rescore.

Positive ``delta`` always favors the report channel: for AURC/Brier it is
metric(s_q) - metric(report), and for AUROC it is metric(report) - metric(s_q).

Examples
--------
    python verbalized_conformal_confidence/report_channel_robustness.py
    python verbalized_conformal_confidence/report_channel_robustness.py \
        --checks probe_readout alternative_readouts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ladder import _split  # noqa: E402
from scripts.step_token_contribution import (  # noqa: E402
    aurc,
    build_targets,
    fit_predict,
    load_cell,
)
from utils.paths import logprobs_path  # noqa: E402


CELLS = tuple(
    (m, d)
    for m in ("llama3_8b", "mistral_7b", "qwen2_5_7b")
    for d in ("truthfulqa", "pavlick_nli")
)
SEEDS = (0, 1, 2)
HIGH_LEVEL = 0.70
COMPLIANCE_STRATA = (
    ("low_[0,.70)", 0.00, 0.70),
    ("mid_[.70,.90)", 0.70, 0.90),
    ("high_[.90,1]", 0.90, np.inf),
)
OUT = Path(__file__).resolve().parent / "cached_results" / "report_robustness"


CONCESSIONS = """1. ONE a- PER QUESTION. The cache stores h_neg as [N, d], not
[N, K, d]; a distractor sweep requires a full second extraction. The result is
conditional on one draw of a-. Two qualitatively different constructions
(curated falsehood on TruthfulQA and complementary label on Pavlick NLI) give
the same direction, which is weaker than robustness to the construction.

2. a- IS DATASET-SUPPLIED, NOT MODEL-GENERATED. This is bounded by the deployed
object: V+ on the model's own answer performs worse than the paired report
(-0.060 versus -0.011 over measurable cells). The objection therefore points
away from the paper's conclusion, but it remains a construction limitation.

3. TEACHER-FORCING IS OFF-POLICY. Forcing a- can put the model in a state that
sampling would not reach. This is the cost of exact item-level difficulty
control. Alphabet mass (0.61--0.94 on Llama and Mistral) records how far the
elicitation goes off-format and is reported rather than treated as harmless.

4. SINGLE FIXED PARTITION. The experiment has no independent confirmatory
partition; rerunning random splits cannot recreate one.
"""


BASELINE_DEFENCE = (
    "The question-only baseline is not artificially strong: on identical "
    "TruthfulQA items, the paired state probe improves its AURC from 0.481 to "
    "0.335 for Llama, 0.402 to 0.298 for Mistral, and 0.414 to 0.260 for Qwen."
)

ORDERING_CLARIFICATION = (
    "Pair order is fixed by gold identity before the target is computed, and "
    "both channels receive both ordered branches. This differs from the "
    "withdrawn h_a construction, where the target selected the input branch."
)


def _cell_name(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def _load(cell: tuple[str, str]) -> tuple[dict, dict, np.ndarray]:
    d, err = load_cell(*cell)
    if err:
        raise FileNotFoundError(err["error"])
    lp = torch.load(logprobs_path(*cell), map_location="cpu", weights_only=False)
    y = build_targets(d)["mean"][0]
    return d, lp, y


def _normalise_rows(dist) -> np.ndarray:
    p = np.clip(np.asarray(dist, dtype=np.float64), 0.0, None)
    total = p.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Vdist contains a row with no probability mass")
    return p / total


def distribution_readouts(lp: dict, high_level: float = HIGH_LEVEL) -> dict[str, np.ndarray]:
    """Return paired report scores under four fixed distribution functionals."""
    values = np.asarray(lp["conf_values"], dtype=np.float64) / 100.0
    out = {}
    side = {}
    for name in ("pos", "neg"):
        p = _normalise_rows(lp[f"Vdist_{name}"])
        cdf = np.cumsum(p, axis=1)
        side[name] = {
            "mean": p @ values,
            "argmax": values[np.argmax(p, axis=1)],
            "median": values[np.argmax(cdf >= 0.5, axis=1)],
            "mass_hi": p[:, values >= high_level].sum(axis=1),
        }
    for readout in side["pos"]:
        out[readout] = side["pos"][readout] - side["neg"][readout]
    return out


def _adequacy(score: np.ndarray, width: float = 0.05) -> dict[str, float | bool]:
    """Outcome-free ranking gate fixed in paper/MEASUREMENT_ADEQUACY.md."""
    score = np.asarray(score, dtype=float)
    score = score[np.isfinite(score)]
    bins = np.floor(score / width + 0.5) * width
    _, counts = np.unique(bins, return_counts=True)
    mass = counts / counts.sum()
    effective = float(np.exp(-np.sum(mass * np.log(mass))))
    return {"effective_bins": effective, "max_bin_mass": float(mass.max()),
            "measurable": bool(effective >= 2.0)}


def _question_scores(d: dict, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr, te = _split(len(y), seed)
    sq = fit_predict(d["h_q"][tr], y[tr], d["h_q"][te], seed)
    return tr, te, sq


def _fit_small_probe(xtr, ytr, xte, seed: int) -> np.ndarray:
    """Three-feature parity probe: standardized logistic regression, no PCA."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
    ).fit(xtr, ytr)
    return model.predict_proba(xte)[:, 1]


def probe_readout(cells: Sequence = CELLS) -> pd.DataFrame:
    """Read [V+, V-, V+-V-] with the same supervised estimator family as h."""
    rows = []
    for cell in cells:
        d, lp, y = _load(tuple(cell))
        vp = np.asarray(lp["V_pos"], dtype=np.float64)
        vn = np.asarray(lp["V_neg"], dtype=np.float64)
        x = np.column_stack((vp, vn, vp - vn))
        for seed in SEEDS:
            tr, te, sq = _question_scores(d, y, seed)
            risk = 1.0 - y[te]
            a_q = aurc(sq, risk)
            scores = {
                "fixed_mean_gap": vp[te] - vn[te],
                "probe_[V+,V-,gap]": _fit_small_probe(x[tr], y[tr], x[te], seed),
            }
            for readout, score in scores.items():
                a_report = aurc(score, risk)
                rows.append({
                    "model": cell[0], "dataset": cell[1], "seed": seed,
                    "readout": readout, "n_test": int(te.sum()),
                    "AURC_sq": a_q, "AURC_report": a_report,
                    "delta": a_q - a_report,
                })
    return pd.DataFrame(rows)


def alternative_readouts(cells: Sequence = CELLS, high_level: float = HIGH_LEVEL) -> pd.DataFrame:
    """Rescore Vdist using mean, modal level, median, and mass >= 0.70."""
    rows = []
    for cell in cells:
        d, lp, y = _load(tuple(cell))
        scores = distribution_readouts(lp, high_level)
        adequacy = {name: _adequacy(score) for name, score in scores.items()}
        for seed in SEEDS:
            _, te, sq = _question_scores(d, y, seed)
            risk, a_q = 1.0 - y[te], aurc(sq, 1.0 - y[te])
            for readout, score in scores.items():
                a_report = aurc(score[te], risk)
                rows.append({
                    "model": cell[0], "dataset": cell[1], "seed": seed,
                    "readout": readout, "high_level": high_level,
                    "n_test": int(te.sum()), "AURC_sq": a_q,
                    "AURC_report": a_report, "delta": a_q - a_report,
                    **adequacy[readout],
                })
    return pd.DataFrame(rows)


def compliance_stratified(cells: Sequence = CELLS) -> pd.DataFrame:
    """Evaluate the mean-gap score in strata of min(Vmass+, Vmass-)."""
    rows = []
    for cell in cells:
        d, lp, y = _load(tuple(cell))
        score = np.asarray(lp["V_pos"], dtype=float) - np.asarray(lp["V_neg"], dtype=float)
        compliance = np.minimum(np.asarray(lp["Vmass_pos"], dtype=float),
                                np.asarray(lp["Vmass_neg"], dtype=float))
        stratum_adequacy = {}
        for label, lo, hi in COMPLIANCE_STRATA:
            member = (compliance >= lo) & (compliance < hi)
            stratum_adequacy[label] = (_adequacy(score[member]) if member.any() else
                                       {"effective_bins": np.nan, "max_bin_mass": np.nan,
                                        "measurable": False})
        for seed in SEEDS:
            _, te, sq = _question_scores(d, y, seed)
            test_ix = np.flatnonzero(te)
            for label, lo, hi in COMPLIANCE_STRATA:
                keep = (compliance[test_ix] >= lo) & (compliance[test_ix] < hi)
                if not keep.any():
                    rows.append({
                        "model": cell[0], "dataset": cell[1], "seed": seed,
                        "stratum": label, "mass_lo": lo, "mass_hi": hi,
                        "n_test": 0, "positive_rate": np.nan,
                        "AURC_sq": np.nan, "AURC_report": np.nan, "delta": np.nan,
                        **stratum_adequacy[label],
                    })
                    continue
                ix = test_ix[keep]
                risk = 1.0 - y[ix]
                aq, ar = aurc(sq[keep], risk), aurc(score[ix], risk)
                rows.append({
                    "model": cell[0], "dataset": cell[1], "seed": seed,
                    "stratum": label, "mass_lo": lo, "mass_hi": hi,
                    "n_test": int(keep.sum()), "positive_rate": float(y[ix].mean()),
                    "AURC_sq": aq, "AURC_report": ar, "delta": aq - ar,
                    **stratum_adequacy[label],
                })
    return pd.DataFrame(rows)


def _entropy_norm(dist) -> np.ndarray:
    p = _normalise_rows(dist)
    h = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
    return h / np.log(p.shape[1])


def alphabet_robustness(model: str = "llama3_8b") -> pd.DataFrame:
    """Compare canonical letter11 with Llama's clean digits101 cache."""
    if model != "llama3_8b":
        raise ValueError("only llama3_8b has an uncontaminated cached numeric run")
    rows = []
    for dataset in ("truthfulqa", "pavlick_nli"):
        cell = (model, dataset)
        d, letter, y = _load(cell)
        fine_path = logprobs_path(*cell).with_name("fine_conf.pt")
        if not fine_path.exists():
            raise FileNotFoundError(f"missing numeric-alphabet cache: {fine_path}")
        numeric = torch.load(fine_path, map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(letter["example_ids"]), np.asarray(numeric["example_ids"])):
            raise ValueError(f"item IDs do not align for {_cell_name(cell)}")
        for alphabet, blob in (("letter11", letter), ("digits101", numeric)):
            vp = np.asarray(blob["V_pos"], dtype=float)
            vn = np.asarray(blob["V_neg"], dtype=float)
            score = vp - vn
            adequacy = _adequacy(score)
            hnorm = np.concatenate((_entropy_norm(blob["Vdist_pos"]),
                                    _entropy_norm(blob["Vdist_neg"])))
            for seed in SEEDS:
                _, te, sq = _question_scores(d, y, seed)
                risk, aq = 1.0 - y[te], aurc(sq, 1.0 - y[te])
                ar = aurc(score[te], risk)
                rows.append({
                    "model": model, "dataset": dataset, "seed": seed,
                    "alphabet": alphabet, "K": len(blob["conf_values"]),
                    "n_test": int(te.sum()), "V_pos_mean": float(vp.mean()),
                    "V_neg_mean": float(vn.mean()), "H_norm_both_mean": float(hnorm.mean()),
                    "AURC_sq": aq, "AURC_report": ar, "delta": aq - ar,
                    **adequacy,
                })
    return pd.DataFrame(rows)


def target_convention(cells: Sequence = CELLS) -> pd.DataFrame:
    """Re-run the report comparison against mean- and sum-logprob targets."""
    rows = []
    for cell in cells:
        d, lp, _ = _load(tuple(cell))
        targets = build_targets(d)
        if "sum" not in targets:
            raise ValueError(f"sum-logprob target unavailable for {_cell_name(tuple(cell))}")
        ym, ys = targets["mean"][0], targets["sum"][0]
        score = np.asarray(lp["V_pos"], dtype=float) - np.asarray(lp["V_neg"], dtype=float)
        adequacy = _adequacy(score)
        for target_name, (y, provenance) in targets.items():
            for seed in SEEDS:
                _, te, sq = _question_scores(d, y, seed)
                risk, aq = 1.0 - y[te], aurc(sq, 1.0 - y[te])
                ar = aurc(score[te], risk)
                rows.append({
                    "model": cell[0], "dataset": cell[1], "seed": seed,
                    "target": target_name, "target_provenance": provenance,
                    "label_agreement": float((ym == ys).mean()),
                    "n_test": int(te.sum()), "AURC_sq": aq,
                    "AURC_report": ar, "delta": aq - ar,
                    **adequacy,
                })
    return pd.DataFrame(rows)


def _brier(prob, y) -> float:
    p = np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def metric_robustness(cells: Sequence = CELLS) -> pd.DataFrame:
    """Compare the same held-out rankings under AURC, AUROC, and Brier.

    For Brier only, the signed report gap is mapped affinely from [-1, 1] to
    [0, 1].  This is a declared sensitivity, not a calibrated probability; its
    comparison with a fitted s_q probability therefore mixes discrimination
    and calibration.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for cell in cells:
        d, lp, y = _load(tuple(cell))
        gap = np.asarray(lp["V_pos"], dtype=float) - np.asarray(lp["V_neg"], dtype=float)
        adequacy = _adequacy(gap)
        for seed in SEEDS:
            _, te, sq = _question_scores(d, y, seed)
            yte, gte = y[te], gap[te]
            metrics: tuple[tuple[str, Callable, np.ndarray, np.ndarray, str], ...] = (
                ("AURC", lambda s, yy: aurc(s, 1.0 - yy), sq, gte, "lower"),
                ("AUROC", lambda s, yy: float(roc_auc_score(yy, s)), sq, gte, "higher"),
                ("Brier", _brier, sq, np.clip((gte + 1.0) / 2.0, 0.0, 1.0), "lower"),
            )
            for metric, fn, qscore, rscore, direction in metrics:
                mq, mr = fn(qscore, yte), fn(rscore, yte)
                delta = (mr - mq) if direction == "higher" else (mq - mr)
                rows.append({
                    "model": cell[0], "dataset": cell[1], "seed": seed,
                    "metric": metric, "better": direction, "n_test": int(te.sum()),
                    "metric_sq": mq, "metric_report": mr, "delta": delta,
                    "report_transform": ("(gap+1)/2" if metric == "Brier" else "none"),
                    **adequacy,
                })
    return pd.DataFrame(rows)


CHECKS = {
    "probe_readout": probe_readout,
    "alternative_readouts": alternative_readouts,
    "compliance_stratified": compliance_stratified,
    "alphabet_robustness": alphabet_robustness,
    "target_convention": target_convention,
    "metric_robustness": metric_robustness,
}


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("model", "dataset", "readout", "stratum", "alphabet", "target", "metric")
            if c in frame.columns]
    aggregations = {"n_test_mean": ("n_test", "mean"),
                    "delta_mean": ("delta", "mean"),
                    "delta_sd": ("delta", "std")}
    for col in ("AURC_sq", "AURC_report", "metric_sq", "metric_report"):
        if col in frame:
            aggregations[f"{col}_mean"] = (col, "mean")
    for col in ("effective_bins", "max_bin_mass", "measurable"):
        if col in frame:
            aggregations[col] = (col, "first")
    out = frame.groupby(keys, as_index=False, dropna=False).agg(**aggregations)
    if "measurable" in out:
        out["outcome"] = np.where(~out["measurable"], "not_measurable",
                                  np.where(out["delta_mean"] > 0, "improves",
                                           "does_not_improve"))
    return out


def _probe_decision(frame: pd.DataFrame) -> dict:
    wide = (_summary(frame).pivot(index=["model", "dataset"], columns="readout",
                                  values="delta_mean").reset_index())
    wide["probe_minus_fixed"] = wide["probe_[V+,V-,gap]"] - wide["fixed_mean_gap"]
    n_better = int((wide["probe_minus_fixed"] >= 0.01).sum())
    all_close = bool((wide["probe_minus_fixed"].abs() <= 0.01).all())
    verdict = ("functional_not_binding" if all_close else
               "headline_requires_probe_parity" if n_better >= 2 else
               "mixed_review_required")
    return {"threshold": 0.01, "n_cells_probe_materially_better": n_better,
            "all_cells_within_0.01": all_close, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=tuple(CHECKS), default=tuple(CHECKS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cells": [_cell_name(c) for c in CELLS], "seeds": list(SEEDS),
        "high_level": HIGH_LEVEL,
        "compliance_strata": [[x, lo, None if np.isinf(hi) else hi]
                               for x, lo, hi in COMPLIANCE_STRATA],
        "delta_convention": "positive favors report channel",
        "baseline_defence": BASELINE_DEFENCE,
        "ordering_clarification": ORDERING_CLARIFICATION,
        "concessions": CONCESSIONS,
        "checks": {},
    }
    for name in args.checks:
        frame = CHECKS[name]()
        summary = _summary(frame)
        frame.to_csv(args.out / f"{name}_per_seed.csv", index=False)
        summary.to_csv(args.out / f"{name}_summary.csv", index=False)
        manifest["checks"][name] = {
            "n_rows": len(frame),
            "per_seed": f"{name}_per_seed.csv",
            "summary": f"{name}_summary.csv",
        }
        if name == "probe_readout":
            manifest["checks"][name]["decision"] = _probe_decision(frame)
        print(f"\n{name}\n{summary.to_string(index=False)}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "paper_text.txt").write_text(
        BASELINE_DEFENCE + "\n\n" + ORDERING_CLARIFICATION + "\n\n" +
        CONCESSIONS.strip() + "\n")
    print(f"\nWrote robustness outputs to {args.out}")


if __name__ == "__main__":
    main()
