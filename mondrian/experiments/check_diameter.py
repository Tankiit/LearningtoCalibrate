"""Numeric check for Lemma lem:diam (cell diameters of nearest-exemplar cells).

X ~ Uniform[0,1]^d, d in {1,2}; exemplars from mondrian.exemplars.select_exemplars
on D_fit (n_fit i.i.d. points) with method "voronoi" (Gonzalez farthest-first)
and "kmeans" (k-means centres snapped one-to-one to D_fit, as in X1).

Population Voronoi cells are clipped to the support [0,1]^d and computed EXACTLY:
d=1 -> intervals between bisector midpoints; d=2 -> convex polygons obtained by
clipping the unit square with the M-1 bisector half-planes (Sutherland-Hodgman).
A convex polygon's diameter is the max vertex-vertex distance, and the support
covering radius is the max over cells of the max vertex-to-own-exemplar distance.
A Monte-Carlo cross-check (200k uniform points, convex hull per cell) is run on
seed 0 to validate the exact computation.

Usage (repo root): python -m experiments.check_diameter [--out runs/p1_diameter]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from mondrian.cells import assign_voronoi
from mondrian.exemplars import select_exemplars

M_GRID = [1, 2, 3, 4, 5, 6, 8, 11, 14, 18, 23, 30, 39, 51, 67, 87, 112, 146, 190]
METHODS = ["voronoi", "kmeans"]


def cells_1d(E):
    """Exact cells on [0,1]: returns per-exemplar (diameter, max dist to exemplar)."""
    e = E[:, 0]
    order = np.argsort(e, kind="stable")
    s = e[order]
    cuts = np.concatenate([[0.0], (s[1:] + s[:-1]) / 2, [1.0]])
    diam = np.empty(len(e)); rad = np.empty(len(e))
    diam[order] = cuts[1:] - cuts[:-1]
    rad[order] = np.maximum(s - cuts[:-1], cuts[1:] - s)
    return diam, rad


def _clip(poly, a, b):
    """Clip convex polygon (list of 2-vectors) to half-plane a.x <= b."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        fp, fq = a @ P - b, a @ Q - b
        if fp <= 0:
            out.append(P)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append(P + t * (Q - P))
    return out


def cells_2d(E):
    square = [np.array(v, float) for v in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    M = len(E)
    diam = np.zeros(M); rad = np.zeros(M)
    for i in range(M):
        poly = list(square)
        for j in range(M):
            if j == i:
                continue
            # ||x-e_i||^2 <= ||x-e_j||^2  <=>  2(e_j-e_i).x <= |e_j|^2-|e_i|^2
            a = 2 * (E[j] - E[i]); b = E[j] @ E[j] - E[i] @ E[i]
            poly = _clip(poly, a, b)
            if not poly:
                break
        if len(poly) == 0:
            continue
        V = np.array(poly)
        diam[i] = pdist(V).max() if len(V) > 1 else 0.0
        rad[i] = np.sqrt(((V - E[i]) ** 2).sum(1)).max()
    return diam, rad


def mc_check(E, d, rng, n=200_000):
    U = rng.random((n, d))
    lab = assign_voronoi(U, E)
    best = 0.0
    for j in np.unique(lab):
        P = U[lab == j]
        if d == 1:
            best = max(best, P.max() - P.min()); continue
        if len(P) < 3:
            continue
        H = P[ConvexHull(P).vertices]
        best = max(best, pdist(H).max())
    return best


def theory_K(d):
    from math import gamma, pi, sqrt
    a = 2.0 ** (-d) * pi ** (d / 2) / gamma(d / 2 + 1)
    C1 = max(2.0, sqrt(d))
    return 2 * C1 * a ** (-1 / d), 2 * (C1 + 1) * a ** (-1 / d)


def fit_slope(M, y, m_min=1):
    M = np.asarray(M, float); y = np.asarray(y, float)
    k = M >= m_min
    b, a = np.polyfit(np.log(M[k]), np.log(y[k]), 1)
    return float(b), float(np.exp(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/p1_diameter")
    p.add_argument("--n-fit", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    t0 = time.time()
    for d in (1, 2):
        for seed in range(args.seeds):
            rng = np.random.default_rng(10_000 * d + seed)
            X = rng.random((args.n_fit, d))
            for method in METHODS:
                for M in M_GRID:
                    idx, _ = select_exemplars(X, M, method, np.random.default_rng([d, seed, M]))
                    E = X[idx]
                    diam, rad = cells_1d(E) if d == 1 else cells_2d(E)
                    row = dict(d=d, seed=seed, method=method, M=M,
                               max_diam=float(diam.max()), mean_diam=float(diam.mean()),
                               cover_radius=float(rad.max()),
                               K_hat=float(diam.max() * M ** (1 / d)),
                               sample_cover=float(np.sqrt(((X[:, None, :] - E[None]) ** 2).sum(-1)).min(1).max()))
                    if seed == 0 and M in (4, 30, 190):
                        row["mc_max_diam"] = mc_check(E, d, np.random.default_rng(seed))
                    rows.append(row)
            print(f"d={d} seed={seed} done ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(args.out, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    summary = {"n_fit": args.n_fit, "n_seeds": args.seeds, "M_grid": M_GRID,
               "note": "max_diam exact over population Voronoi cells clipped to [0,1]^d", "fits": {}}
    for d in (1, 2):
        for method in METHODS:
            R = [r for r in rows if r["d"] == d and r["method"] == method]
            md = np.array([[r["max_diam"] for r in R if r["M"] == M] for M in M_GRID])
            cr = np.array([[r["cover_radius"] for r in R if r["M"] == M] for M in M_GRID])
            mean_md = md.mean(1)
            # per-seed slopes for a spread estimate
            per_seed = [fit_slope(M_GRID, md[:, s], 4)[0] for s in range(md.shape[1])]
            K_all = md.max(1) * np.asarray(M_GRID, float) ** (1 / d)
            key = f"d{d}_{method}"
            summary["fits"][key] = {
                "slope_all_M": fit_slope(M_GRID, mean_md)[0],
                "slope_M>=4": fit_slope(M_GRID, mean_md, 4)[0],
                "slope_M>=4_seed_sd": float(np.std(per_seed, ddof=1)),
                "slope_cover_radius_M>=4": fit_slope(M_GRID, cr.mean(1), 4)[0],
                "intercept_M>=4": fit_slope(M_GRID, mean_md, 4)[1],
                "K_hat_max_over_seeds_and_M": float(K_all.max()),
                "K_hat_max_over_seeds_M>=4": float(K_all[np.asarray(M_GRID) >= 4].max()),
                "K_hat_by_M_max_over_seeds": dict(zip(map(str, M_GRID), map(float, K_all))),
                # Lemma constants for [0,1]^d: a = 2^-d omega_d, r0 = 1, C1 = max(2, sqrt d).
                # K_inf = 2 C1 a^{-1/d} (M << n/log n); K_lemma = 2 (C1+1) a^{-1/d} (M <= n/(2^d s log n)).
                "theory_K_inf": theory_K(d)[0],
                "theory_K_lemma": theory_K(d)[1],
                "mc_vs_exact": [(r["M"], r["max_diam"], r["mc_max_diam"]) for r in R if "mc_max_diam" in r],
            }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for k, v in summary["fits"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                  if kk not in ("K_hat_by_M_max_over_seeds",)})


if __name__ == "__main__":
    main()
