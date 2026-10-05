"""E7' synthetic reliability model (Addendum E, B9): when do learned groups beat
recalibration of the report, and when do they beat a direct predictor?"""
import argparse
import itertools
import json
import os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash

EPS = 1e-4
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
MS = (1, 2, 4, 8, 16)
PROCS = {"fwd": 0.0, "rev": 1.0, "alt": -0.5}
DIMS, CLUSTERS = 8, 8      # v1; v2 (amendment): --dims 4 --clusters 32


def sig(z):
    return 1 / (1 + np.exp(-z))


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def sample(n, tau, kappa, structure, rng, centres, w, beta_z):
    z = rng.integers(len(centres), size=n)
    phi = centres[z] + rng.normal(size=(n, centres.shape[1]))
    u = rng.normal(size=n)
    if structure == "cluster":
        beta = beta_z[z]
    else:
        s = phi @ w; beta = tau * s / np.sqrt(9 * (w @ w) / len(w) + w @ w)   # roughly unit-scale w.phi
    y = (rng.uniform(size=n) < sig(beta + u)).astype(float)
    eps = rng.normal(0, .5, size=n)
    V = {p: sig(kappa * u + eps + d) for p, d in PROCS.items()}
    return phi, y, V


def fit_tuned(X, y, folds=3):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    if X.shape[1] > 1:
        sd = sd.copy(); sd[0] /= 10.0      # amendment: report column (first) penalised 100x less
    Xs = (X - mu) / sd
    if min(np.bincount(y.astype(int), minlength=2)) < folds:
        c = float(np.clip(y.mean(), EPS, 1 - EPS)); return (lambda Xn: np.full(len(Xn), c)), np.inf
    best = (np.inf, 1.0)
    for C in CS:
        ll = 0.0
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=0).split(Xs, y):
            p = np.clip(LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr]).predict_proba(Xs[va])[:, 1], EPS, 1 - EPS)
            ll -= np.sum(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
        if ll < best[0]:
            best = (ll, C)
    m = LogisticRegression(C=best[1], max_iter=2000).fit(Xs, y)
    return (lambda Xn: np.clip(m.predict_proba((Xn - mu) / sd)[:, 1], EPS, 1 - EPS)), best[0] / len(y)


def one(tau, kappa, structure, n_lab, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 77, int(tau * 10), int(kappa * 10), n_lab, structure == "cluster"]))
    d, G = DIMS, CLUSTERS; c = rng.normal(size=(G, d)); centres = 3 * c / np.linalg.norm(c, axis=1, keepdims=True)
    w = rng.normal(size=d); beta_z = rng.normal(0, tau, size=G) if tau > 0 else np.zeros(G)
    phi_f, _, _ = sample(2000, tau, kappa, structure, rng, centres, w, beta_z)
    phi_l, y_l, V_l = sample(n_lab, tau, kappa, structure, rng, centres, w, beta_z)
    phi_t, y_t, V_t = sample(2000, tau, kappa, structure, rng, centres, w, beta_z)
    cells = {M: ExemplarPartition.fit(phi_f, M, "voronoi", seed) for M in MS[1:]}
    out = {"R": [], "G": [], "D": [], "M": []}
    for p in PROCS:
        lvl, lvt = lg(V_l[p])[:, None], lg(V_t[p])[:, None]
        fR, scR = fit_tuned(lvl, y_l); out["R"].append(np.mean((fR(lvt) - y_t) ** 2))
        best = (scR, 1, fR(lvt))
        for M in MS[1:]:
            Ol, Ot = np.eye(M)[cells[M](phi_l)], np.eye(M)[cells[M](phi_t)]
            f, sc = fit_tuned(np.hstack([lvl, Ol]), y_l)
            if sc < best[0]:
                best = (sc, M, f(np.hstack([lvt, Ot])))
        out["G"].append(np.mean((best[2] - y_t) ** 2)); out["M"].append(best[1])
        fD, _ = fit_tuned(np.hstack([lvl, phi_l]), y_l); out["D"].append(np.mean((fD(np.hstack([lvt, phi_t])) - y_t) ** 2))
    return {k: float(np.mean(v)) for k, v in out.items()}


def one_dims(tau, kappa, structure, n_lab, seed, dims, clusters):
    global DIMS, CLUSTERS
    DIMS, CLUSTERS = dims, clusters
    return one(tau, kappa, structure, n_lab, seed)


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--out", default="runs/p1_synth")
    pa.add_argument("--seeds", type=int, default=30)
    pa.add_argument("--dims", type=int, default=8)
    pa.add_argument("--clusters", type=int, default=8)
    a = pa.parse_args()
    global DIMS, CLUSTERS
    DIMS, CLUSTERS = a.dims, a.clusters
    from joblib import Parallel, delayed
    grid = list(itertools.product((0.0, .5, 1.0, 2.0), (0.0, .5, 1.0, 2.0), ("cluster", "linear"), (60, 120, 240, 480)))
    jobs = [(g, s) for g in grid for s in range(a.seeds)]
    res = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1), verbose=0)(delayed(one_dims)(*g, s, DIMS, CLUSTERS) for g, s in jobs)
    table = {}
    for (g, s), r in zip(jobs, res):
        table.setdefault(g, []).append(r)
    summary = []
    for (tau, kappa, st, n), rs in table.items():
        dGR = np.array([r["G"] - r["R"] for r in rs]); dGD = np.array([r["G"] - r["D"] for r in rs])
        ci = lambda x: [float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))), float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x)))]
        summary.append({"tau": tau, "kappa": kappa, "structure": st, "n_lab": n,
                        "R": float(np.mean([r["R"] for r in rs])), "G": float(np.mean([r["G"] for r in rs])),
                        "D": float(np.mean([r["D"] for r in rs])), "G-R": float(dGR.mean()), "G-R_ci": ci(dGR),
                        "G-D": float(dGD.mean()), "G-D_ci": ci(dGD), "mean_M": float(np.mean([r["M"] for r in rs]))})
    write_json(Path(a.out) / "synth_summary.json", {"criterion": "reports/estimand_note_E5_2026-10-02.md#e7", "dims": DIMS, "clusters": CLUSTERS,
                                                   "rows": summary, "source_sha256": source_hash()})
    print("done", len(summary))


if __name__ == "__main__":
    main()
