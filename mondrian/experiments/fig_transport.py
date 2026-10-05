"""Addendum D: legend-alignment transport plans (letter space, value space) on Run 1."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from .e2_frozen import RUN1, MODELS

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LETTERS = list("ABCDEFGHJKL")
NAMES = {"llama3_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "qwen2_5_7b": "Qwen2.5-7B"}


def load(model, rel):
    import torch
    d = torch.load(f"{RUN1}/{model}/{rel}.pt", map_location="cpu", weights_only=False)
    P = np.vstack([np.asarray(d["Vdist_pos"], float), np.asarray(d["Vdist_neg"], float)])
    return P / P.sum(1, keepdims=True), np.asarray(d["conf_values"], float)


def value_plan(PL, vals_a, vals_b):
    grid = np.arange(0, 101, 10); V = np.zeros((11, 11))
    ia = {v: i for i, v in enumerate(grid)}
    for a in range(11):
        for b in range(11):
            V[ia[vals_a[a]], ia[vals_b[b]]] += PL[a, b]
    return V


def monotone_plan(a, b):
    """Optimal (north-west corner on sorted support) coupling of two marginals on the same grid."""
    a, b = a.copy(), b.copy(); P = np.zeros((len(a), len(b))); i = j = 0
    while i < len(a) and j < len(b):
        m = min(a[i], b[j]); P[i, j] += m; a[i] -= m; b[j] -= m
        if a[i] <= 1e-15: i += 1
        if j < len(b) and b[j] <= 1e-15: j += 1
    return P


def cost(V):
    g = np.arange(0, 101, 10)
    return float(np.sum(V * np.abs(g[:, None] - g[None, :])))


def analyse(model):
    Pf, vf = load(model, "main/fwd"); Pr, vr = load(model, "main/rev"); Pa, va = load(model, "alt/fwd")
    out = {}
    for tag, Q, vq in (("reversed", Pr, vr), ("reworded", Pa, va)):
        PL = Pf.T @ Q / len(Pf)
        r, c = linear_sum_assignment(-PL)
        sigma_raw = c[np.argsort(r)]
        lift = PL / np.maximum(np.outer(PL.sum(1), PL.sum(0)), 1e-12)
        r, c = linear_sum_assignment(-np.log(np.maximum(lift, 1e-12)))
        sigma = c[np.argsort(r)]
        V = value_plan(PL, vf, vq)
        self_ = value_plan(Pf.T @ Pf / len(Pf), vf, vf)
        shuf = value_plan(np.outer(Pf.mean(0), Q.mean(0)), vf, vq)
        g = np.arange(0, 101, 10)
        w1 = wasserstein_distance(g, g, V.sum(1), V.sum(0))
        creal, cself, cshuf = cost(V), cost(self_), cost(shuf)
        out[tag] = {"plan_letters": PL.tolist(), "plan_values": V.tolist(), "hungarian": sigma.tolist(),
                    "lift": lift.tolist(), "hungarian_raw": sigma_raw.tolist(),
                    "mirror_matches_raw": int(np.sum(sigma_raw == 10 - np.arange(11))),
                    "identity_matches_raw": int(np.sum(sigma_raw == np.arange(11))),
                    "mirror_matches": int(np.sum(sigma == 10 - np.arange(11))),
                    "identity_matches": int(np.sum(sigma == np.arange(11))),
                    "mass_antidiag_band": float(sum(PL[a, b] for a in range(11) for b in range(11) if abs(b - (10 - a)) <= 1)),
                    "mass_diag_band": float(sum(PL[a, b] for a in range(11) for b in range(11) if abs(b - a) <= 1)),
                    "C_real": creal, "C_self": cself, "C_shuf": cshuf, "W1": float(w1),
                    # not interpretable when shuffled and redraw costs differ by under one point (see text)
                    "instability": (creal - cself) / (cshuf - cself) if cshuf - cself >= 1.0 else None}
    return out


def figure(res, path="paper/figs/fig_transport"):
    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.4), constrained_layout=True)
    g = np.arange(0, 101, 10)
    for j, m in enumerate(MODELS):
        R = res[m]["reversed"]
        L = np.log2(np.maximum(np.array(R["lift"]), 1e-3))
        ax = axes[0, j]
        im0 = ax.imshow(L, cmap="RdBu_r", vmin=-2, vmax=2, origin="upper")
        ax.plot(np.arange(11), 10 - np.arange(11), ls="--", lw=.8, c="black")
        ax.plot(np.arange(11), np.arange(11), ls=":", lw=.8, c="grey")
        ax.scatter(R["hungarian"], np.arange(11), s=16, facecolors="none", edgecolors="black", lw=.9)
        ax.set_xticks(range(11)); ax.set_xticklabels(LETTERS); ax.set_yticks(range(11)); ax.set_yticklabels(LETTERS)
        ax.set_title(f"{NAMES[m]}\nmax |log2 lift| {np.abs(L).max():.2f}\nmirror {R['mirror_matches']}/11, same {R['identity_matches']}/11",
                     fontsize=6.8)
        if j == 0: ax.set_ylabel("letter, forward legend")
        ax.set_xlabel("letter, reversed legend")
        V = np.array(R["plan_values"]); ax = axes[1, j]
        im1 = ax.imshow(V / V.max(), cmap="YlGnBu", origin="lower", vmin=0, vmax=1, extent=(-5, 105, -5, 105))
        ax.plot([0, 100], [0, 100], ls="--", lw=.8, c="black")
        P = monotone_plan(V.sum(1), V.sum(0)); ii, jj = np.nonzero(P > 1e-4)
        ax.scatter(g[jj], g[ii], s=200 * P[ii, jj] / P.max(), c="#d95f02", alpha=.85, lw=0)
        ax.set_xlabel("value under reversed legend (%)")
        if j == 0: ax.set_ylabel("value under forward legend (%)")
        rw = res[m]["reworded"]
        fmt = lambda v: "n/i" if v is None else f"{v:.2f}"
        ax.set_title(f"cost {R['C_real']:.0f} | redraw {R['C_self']:.0f} | shuffled {R['C_shuf']:.0f}\n"
                     f"W1 {R['W1']:.0f}; instability {fmt(R['instability'])}\n(reworded prompt: {fmt(rw['instability'])})", fontsize=6.8)
    fig.colorbar(im0, ax=axes[0, :], shrink=.8, label="log2 lift")
    fig.colorbar(im1, ax=axes[1, :], shrink=.8, label="plan mass / max")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)


def main():
    res = {m: analyse(m) for m in MODELS}
    Path("paper/figs").mkdir(parents=True, exist_ok=True)
    Path("paper/figs/fig_transport_numbers.json").write_text(json.dumps(res, indent=1))
    figure(res)
    for m in MODELS:
        for tag in ("reversed", "reworded"):
            r = res[m][tag]
            print(f"{m:11s} {tag:9s} lift-mirror {r['mirror_matches']:2d} lift-ident {r['identity_matches']:2d} raw-mirror {r['mirror_matches_raw']:2d} "
                  f"antidiag {r['mass_antidiag_band']:.2f} diag {r['mass_diag_band']:.2f} "
                  f"C_real {r['C_real']:.1f} C_self {r['C_self']:.1f} C_shuf {r['C_shuf']:.1f} W1 {r['W1']:.1f} iota {r['instability']}")


if __name__ == "__main__":
    main()
