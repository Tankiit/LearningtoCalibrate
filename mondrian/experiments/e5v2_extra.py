"""Addendum F (post hoc, round-2 review): shared-offset TOST and one-cell-threshold conformal for G4."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from mondrian.conformal import per_cell_quantile
from .e_llm import logit, label_scores, FLOOR
from .e5_partition import load_tqa, onehot
from .e5v2 import tuned
from .e2_frozen import MODELS


def nb(d, n_test, n_train, level):
    d = np.asarray(d); J = len(d); se = d.std(ddof=1) * np.sqrt(1 / J + n_test / n_train)
    t = stats.t.ppf(1 - (1 - level) / 2, J - 1)
    return {"mean": float(d.mean()), f"ci{int(level*100)}": [float(d.mean() - t * se), float(d.mean() + t * se)]}


def main(out="runs/p1_e5v2_extra", seeds=range(200), budgets=(60, 120, 180, 272)):
    res = {"criterion": "reports/estimand_note_E5_2026-10-02.md#addendum-f", "source_sha256": source_hash()}
    for model in MODELS:
        H, y, V = load_tqa(model); procs = list(V); n = len(y)
        dGS = {b: [] for b in budgets}; conf = []
        for seed in seeds:
            perm = np.random.default_rng(np.random.SeedSequence([seed, 2525])).permutation(n)
            f, pool, te = np.array_split(perm, 3)
            Z = PCA(16, random_state=0).fit(H[f]).transform(H)
            cells = ExemplarPartition.fit(Z[f], 4, "voronoi", seed)(Z); O = onehot(cells, 4)
            for b in budgets:
                lab = pool[:b]; yl = y[lab].astype(float)
                br_g, rows = [], []
                for p in procs:
                    X = np.hstack([logit(V[p]), O]); br_g.append(np.mean((tuned(X[lab], yl, seed)[0].predict(X[te]) - y[te]) ** 2))
                P = len(procs); blocks_l, blocks_t = [], []
                for i, p in enumerate(procs):
                    lv = logit(V[p])[:, 0]; pv = np.zeros((n, P)); pv[:, i] = lv
                    pd = np.zeros((n, P - 1))
                    if i > 0: pd[:, i - 1] = 1
                    Xs = np.hstack([pv, pd, O]); blocks_l.append(Xs[lab]); blocks_t.append(Xs[te])
                # report columns (first P) up-weighted as in the Addendum E amendment
                from .e5v2 import CS
                Xl = np.vstack(blocks_l); yy = np.tile(yl, P)
                scores = [cv_score_multi(Xl, yy, C, P, seed) for C in CS]
                C = CS[int(np.argmin(scores))]
                gs = fit_multi(Xl, yy, C, P)
                br_s = [np.mean((gs(bt) - y[te]) ** 2) for bt in blocks_t]
                dGS[b].append(float(np.mean(br_s) - np.mean(br_g)))
            # F2: G4 forecaster with one-cell thresholds vs R one-cell, largest budget
            lab = pool[:max(budgets)]; rng = np.random.default_rng(np.random.SeedSequence([seed, 55])); h = rng.permutation(len(lab))
            a, bb = lab[h[:len(lab) // 2]], lab[h[len(lab) // 2:]]
            rec = {}
            for p in procs:
                lv = logit(V[p]); XG = np.hstack([lv, O])
                fR = tuned(lv[a], y[a].astype(float), seed)[0]; fG = tuned(XG[a], y[a].astype(float), seed)[0]
                for alpha in (.1, .2, .3):
                    for name, (pb, pt) in {"R_1cell": (fR.predict(lv[bb]), fR.predict(lv[te])),
                                           "G4_1cell": (fG.predict(XG[bb]), fG.predict(XG[te]))}.items():
                        q, _ = per_cell_quantile(np.zeros(len(bb), int), label_scores(pb, y[bb]), 1, alpha, FLOOR)
                        thr = q[0]; hit = label_scores(pt, y[te]) <= thr
                        size = (label_scores(pt, 1) <= thr).astype(int) + (label_scores(pt, 0) <= thr)
                        rec.setdefault(f"{name}|a{alpha}", []).append((hit.mean(), size.mean(), np.mean(size == 1)))
            conf.append({k: [float(np.mean([x[i] for x in v])) for i in range(3)] for k, v in rec.items()})
        R = {"tost_GS4_minus_G4": {}, "conformal": {}}
        for b in budgets:
            e = nb(dGS[b], 272, b, .90); e["equivalent_pm0.002"] = bool(-0.002 <= e["ci90"][0] and e["ci90"][1] <= 0.002)
            R["tost_GS4_minus_G4"][b] = e
        for k in conf[0]:
            R["conformal"][k] = dict(zip(["coverage", "set_size", "singleton_rate"], [float(np.mean([c[k][i] for c in conf])) for i in range(3)]))
        for alpha in (.1, .2, .3):
            R["conformal"][f"size_G4_1cell-R_1cell|a{alpha}"] = nb([c[f"G4_1cell|a{alpha}"][1] - c[f"R_1cell|a{alpha}"][1] for c in conf], 272, 136, .95)
        res[model] = R
        print(model, {b: (round(v["mean"], 5), [round(x, 4) for x in v["ci90"]], v["equivalent_pm0.002"]) for b, v in R["tost_GS4_minus_G4"].items()},
              {k: [round(x, 3) for x in v.values()] if "size_" not in k else (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in R["conformal"].items() if "a0.1" in k}, flush=True)
    write_json(Path(out) / "e5v2_extra.json", res)


def _std(X, P):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1; sd = sd.copy(); sd[:P] /= 10.0
    return mu, sd


def fit_multi(X, y, C, P):
    from sklearn.linear_model import LogisticRegression
    mu, sd = _std(X, P); m = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, y)
    return lambda Xn: np.clip(m.predict_proba((Xn - mu) / sd)[:, 1], 1e-4, 1 - 1e-4)


def cv_score_multi(X, y, C, P, seed, folds=5):
    from sklearn.model_selection import StratifiedKFold
    tot = 0.0
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        p = fit_multi(X[tr], y[tr], C, P)(X[va]); tot -= np.sum(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
    return tot / len(y)


if __name__ == "__main__":
    main()
