"""Post-lock analysis B (reports/postlock_analyses_note_2026-10-05.md): correctness-probe baselines.

  python -m analyses.probe_baseline [--seeds 100]   -> runs_postlock/probe_baseline.json

Same splits and budgets as Granularity. All arms use only the n_lab labelled questions. P-their follows
third_party/correctness-model-internals (StandardScaler + LogisticRegression, lbfgs, balanced weights) on the
full candidate-conditioned state. Reads cached states and reports only; imports locked code read-only.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.partition import ExemplarPartition
from experiments.e_llm import logit
from experiments.e5_partition import load_tqa, onehot
from experiments.e5v2 import tuned, nb_interval, EPS
from experiments.e9_popqa import load_popqa_tqa_style

OUT = Path("runs_postlock")


def metrics(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return [float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan]


def p_their(H, y, lab, te):
    if len(np.unique(y[lab])) < 2:
        return np.full(len(te), y[lab].mean())
    sc = StandardScaler().fit(H[lab])
    m = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000, class_weight="balanced").fit(sc.transform(H[lab]), y[lab])
    return m.predict_proba(sc.transform(H[te]))[:, 1]


def crossfit(Z, y, lab, seed):
    """Out-of-fold P16 scores on the labelled rows (5-fold), for stacking."""
    oof = np.empty(len(lab))
    k = min(5, int(min(np.bincount(y[lab].astype(int), minlength=2))))
    if k < 2:
        return np.full(len(lab), y[lab].mean())
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(Z[lab], y[lab]):
        oof[va] = tuned(Z[lab][tr], y[lab][tr].astype(float), seed)[0].predict(Z[lab][va])
    return oof


def one(H, y, V, seed, salt, budgets):
    n = len(y)
    f, c, te = np.array_split(np.random.default_rng(np.random.SeedSequence([seed, salt])).permutation(n), 3)
    Z = PCA(16, random_state=0).fit(H[f]).transform(H)
    g4 = ExemplarPartition.fit(Z[f], 4, "voronoi", seed)(Z); O = onehot(g4, 4)
    rows = []
    for nl in budgets:
        lab = c[:nl]; yl = y[lab].astype(float)
        P16 = tuned(Z[lab], yl, seed)[0]
        p16 = P16.predict(Z[te]); oof = crossfit(Z, y, lab, seed)
        pth = p_their(H, y, lab, te)
        arms = {k: [] for k in ("R", "P_their", "P16", "V+P", "G4", "D")}
        for p, v in V.items():
            lv = logit(v).reshape(-1, 1)
            arms["R"].append(metrics(tuned(lv[lab], yl, seed)[0].predict(lv[te]), y[te]))
            arms["P_their"].append(metrics(pth, y[te])); arms["P16"].append(metrics(p16, y[te]))
            Xs = np.column_stack([lv[lab, 0], logit(np.clip(oof, 1e-4, 1 - 1e-4))])
            Xt = np.column_stack([lv[te, 0], logit(np.clip(p16, 1e-4, 1 - 1e-4))])
            arms["V+P"].append(metrics(tuned(Xs, yl, seed)[0].predict(Xt), y[te]))
            XG = np.hstack([lv, O]); arms["G4"].append(metrics(tuned(XG[lab], yl, seed)[0].predict(XG[te]), y[te]))
            XD = np.hstack([lv, Z]); arms["D"].append(metrics(tuned(XD[lab], yl, seed)[0].predict(XD[te]), y[te]))
        rows.append({"seed": seed, "n_lab": nl, **{k: np.nanmean(v, 0).tolist() for k, v in arms.items()}})
    return rows


def summarise(rows, budgets, n_test):
    S = {}
    for nl in budgets:
        R = [r for r in rows if r["n_lab"] == nl]
        S[str(nl)] = {"mean": {k: np.nanmean([r[k] for r in R], 0).tolist() for k in ("R", "P_their", "P16", "V+P", "G4", "D")},
                      "brier_vs_R": {k: nb_interval([r[k][0] - r["R"][0] for r in R], n_test, nl)
                                     for k in ("P_their", "P16", "V+P", "G4", "D")},
                      "brier_vs_D": {k: nb_interval([r[k][0] - r["D"][0] for r in R], n_test, nl) for k in ("P16", "V+P", "G4")}}
    return S


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--seeds", type=int, default=100)
    a = pa.parse_args()
    OUT.mkdir(exist_ok=True)
    res = {"note": "reports/postlock_analyses_note_2026-10-05.md#b", "fields": ["brier", "logloss", "auroc"]}
    jobs = max(1, (os.cpu_count() or 2) - 1)
    for dname, loader, salt, budgets in (("truthfulqa", load_tqa, 2525, (60, 120, 180, 272)),
                                         ("popqa_elicit", load_popqa_tqa_style, 2527, (100, 300, 1000, 2000, 2396))):
        for m in ("llama3_8b", "mistral_7b", "qwen2_5_7b"):
            H, y, V = loader(m)
            per = Parallel(n_jobs=jobs)(delayed(one)(H, y, V, s, salt, budgets) for s in range(a.seeds))
            rows = [r for p in per for r in p]
            res[f"{dname}|{m}"] = summarise(rows, budgets, len(np.array_split(np.arange(len(y)), 3)[2]))
            print("done", dname, m, flush=True)
    (OUT / "probe_baseline.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
