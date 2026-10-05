"""E11 (reports/estimand_note_E9_popqa_2026-10-03.md): semi-synthetic reliability on real PopQA
representations. Known eta lets us measure the pointwise error E(f_hat - eta)^2 directly."""
import itertools
import json
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from mondrian.calibration import metrics as calib
from .e_llm import logit
from .e5_partition import onehot, unit
from .e5v2 import tuned

PROCS = {"fwd": 0.0, "rev": 1.0, "alt": -0.5}
MS = (1, 2, 4, 8, 16)
HS = "/Users/tanmoy/research/OOD_detection/OVA_ARR/outputs/step1_extract_variants/token_mode_mean/llama3_8b/popqa/hidden_states.pt"


def features():
    import hashlib, torch
    h = torch.load(HS, map_location="cpu", weights_only=False)
    ids = np.asarray(h["example_ids"]).astype(str)
    pick = np.array([hashlib.sha256(("p1-e2f|" + q).encode()).digest()[0] & 1 == 0 for q in ids])
    H = unit(np.where(pick[:, None], h["h_pos"].double().numpy(), h["h_neg"].double().numpy()))
    Z64 = PCA(64, random_state=0).fit_transform(H)
    truth = KMeans(32, n_init=10, random_state=0).fit_predict(Z64)
    return H, truth


def one(H, truth, tau, kappa, structure, n_lab, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1111, int(tau * 10), int(kappa * 10), n_lab, structure == "cluster"]))
    n = len(H); perm = rng.permutation(n); f, rest = perm[:2000], perm[2000:]
    lab, te = rest[:n_lab], rest[-2500:]
    Z = PCA(16, random_state=0).fit(H[f]).transform(H)
    if structure == "cluster":
        beta = rng.normal(0, tau, size=truth.max() + 1)[truth] if tau > 0 else np.zeros(n)
    else:
        w = rng.normal(size=16); s = Z @ w; beta = tau * (s - s.mean()) / s.std()
    u = rng.normal(size=n); eta = 1 / (1 + np.exp(-(beta + u)))
    y = (rng.uniform(size=n) < eta).astype(float)
    cells = {M: ExemplarPartition.fit(Z[f], M, "voronoi", seed)(Z) for M in MS[1:]}
    out = {"R": [], "G": [], "D": [], "M": []}
    for p, d in PROCS.items():
        V = 1 / (1 + np.exp(-(kappa * u + rng.normal(0, .5, size=n) + d))); lv = logit(V)
        mR, _, scR = tuned(lv[lab], y[lab], seed); pR = mR.predict(lv[te])
        best = (scR, 1, pR)
        for M in MS[1:]:
            X = np.hstack([lv, onehot(cells[M], M)]); m, _, sc = tuned(X[lab], y[lab], seed)
            if sc < best[0]:
                best = (sc, M, m.predict(X[te]))
        XD = np.hstack([lv, Z]); pD = tuned(XD[lab], y[lab], seed)[0].predict(XD[te])
        for k, pr in (("R", pR), ("G", best[2]), ("D", pD)):
            out[k].append((float(np.mean((pr - eta[te]) ** 2)), float(np.mean((pr - y[te]) ** 2)), calib(pr, y[te])["sce"]))
        out["M"].append(best[1])
    return {k: ([float(np.mean([t[i] for t in v])) for i in range(3)] if k != "M" else float(np.mean(v)))
            for k, v in out.items()}


def main(out="runs/p1_e11", seeds=20):
    from joblib import Parallel, delayed
    H, truth = features()
    grid = list(itertools.product((0.0, .5, 1.0, 2.0), (0.0, 1.0, 2.0), ("cluster", "linear"), (100, 300, 1000, 2000)))
    jobs = [(g, s) for g in grid for s in range(seeds)]
    res = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1))(delayed(one)(H, truth, *g, s) for g, s in jobs)
    table = {}
    for (g, s), r in zip(jobs, res):
        table.setdefault(g, []).append(r)
    rows = []
    for (tau, kappa, st, n), rs in table.items():
        def ci(x):
            x = np.asarray(x); return [float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))), float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x)))]
        dGR = [r["G"][0] - r["R"][0] for r in rs]; dGD = [r["G"][0] - r["D"][0] for r in rs]
        rows.append({"tau": tau, "kappa": kappa, "structure": st, "n_lab": n,
                     "pointwise": {k: float(np.mean([r[k][0] for r in rs])) for k in ("R", "G", "D")},
                     "brier": {k: float(np.mean([r[k][1] for r in rs])) for k in ("R", "G", "D")},
                     "sce": {k: float(np.mean([r[k][2] for r in rs])) for k in ("R", "G", "D")},
                     "G-R": float(np.mean(dGR)), "G-R_ci": ci(dGR), "G-D": float(np.mean(dGD)), "G-D_ci": ci(dGD),
                     "mean_M": float(np.mean([r["M"] for r in rs]))})
    write_json(Path(out) / "e11_result.json", {"criterion": "reports/estimand_note_E9_popqa_2026-10-03.md#e11",
                                              "rows": rows, "source_sha256": source_hash()})
    print("done", len(rows))


if __name__ == "__main__":
    main()
