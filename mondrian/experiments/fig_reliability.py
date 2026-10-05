"""Reliability diagrams for Label-Free-Transfer (E8, and E9c on PopQA): the forecast R_A applied to procedure-B reports,
raw and after each label-free map, against the same forecast under A. Calibration errors are from
calibration-toolbox (mondrian/calibration.py). Test predictions pooled over 50 E8 splits.

  python -m experiments.fig_reliability   -> paper/figs/fig_reliability.{pdf,png}, fig_reliability_numbers.json
"""
import json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mondrian.calibration import metrics, N_BINS
from .e8_transport import platt, map_quantile, map_paired
from .e5_partition import load_tqa
from data.tinker import load_tinker, DEFAULT as TINKER

plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ARMS = [("naive", "no map", "#d95f02"), ("ot_quantile", "quantile map", "#1b9e77"), ("paired_iso", "paired isotonic", "#7570b3")]


def tqa_case(model, B, seeds=range(50), loader=None):
    _, y, V = (loader or load_tqa)(model); n = len(y); P = {k: [] for k in ["A->A"] + [a for a, _, _ in ARMS]}; Y = []
    for seed in seeds:
        f, c, te = np.array_split(np.random.default_rng(np.random.SeedSequence([seed, 8080])).permutation(n), 3)
        RA = platt(V["fwd"][c[:len(c) // 2]], y[c[:len(c) // 2]]); Tq = map_quantile(V[B][f], V["fwd"][f]); Tp, _ = map_paired(V[B][f], V["fwd"][f])
        for k, p in (("A->A", RA(V["fwd"][te])), ("naive", RA(V[B][te])), ("ot_quantile", RA(Tq(V[B][te]))), ("paired_iso", RA(Tp(V[B][te])))):
            P[k].append(p)
        Y.append(y[te])
    return {k: np.concatenate(v) for k, v in P.items()}, np.concatenate(Y)


def tinker_case(arm="q"):
    d = load_tinker(TINKER, cache="inputs/tinker_triviaqa/aligned.npz"); sp, y = d["split"], d["correct"].astype(int)
    f, c, te = sp == "train", sp == "cal", sp == "test"
    VA, VB = d[f"p_{arm}_trained_normal"], d[f"p_{arm}_trained_reversed"]
    ci = np.flatnonzero(c)[:c.sum() // 2]
    RA = platt(VA[ci], y[ci]); Tq = map_quantile(VB[f], VA[f]); Tp, _ = map_paired(VB[f], VA[f])
    return {"A->A": RA(VA[te]), "naive": RA(VB[te]), "ot_quantile": RA(Tq(VB[te])), "paired_iso": RA(Tp(VB[te]))}, y[te]


def popqa_loader():
    from .e9_popqa import load_popqa_tqa_style
    return load_popqa_tqa_style


def curve(p, y):
    e = np.linspace(0, 1, N_BINS + 1); b = np.clip(np.digitize(p, e[1:-1]), 0, N_BINS - 1)
    k = [j for j in range(N_BINS) if np.sum(b == j) >= 20]
    return [float(p[b == j].mean()) for j in k], [float(y[b == j].mean()) for j in k], [int(np.sum(b == j)) for j in k]


def main(path="paper/figs/fig_reliability"):
    cases = [("Mistral, reversed legend\n(TruthfulQA)", tqa_case("mistral_7b", "rev")),
             ("Mistral, reworded prompt\n(TruthfulQA)", tqa_case("mistral_7b", "alt")),
             ("trained $q$ reporter, reversed\n(TriviaQA)", tinker_case("q")),
             ("Mistral, reversed legend\n(PopQA-Elicit)", tqa_case("mistral_7b", "rev", loader=popqa_loader()))]
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.9), constrained_layout=True)
    numbers = {}
    for ax, (title, (P, y)) in zip(axes, cases):
        numbers[title] = {}
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=.6)
        x, f, _ = curve(P["A->A"], y); m = metrics(P["A->A"], y)
        ax.plot(x, f, "-", color="black", lw=.9, label=f"same procedure (SCE {m['sce']:.3f})")
        numbers[title]["A->A"] = m | {"curve": [x, f]}
        for k, lab, col in ARMS:
            x, f, cnt = curve(P[k], y); m = metrics(P[k], y)
            ax.plot(x, f, "o-", ms=2.2, lw=.8, color=col, label=f"{lab} (SCE {m['sce']:.3f})")
            numbers[title][k] = m | {"curve": [x, f], "counts": cnt}
        base = np.full(len(y), y.mean()); mb = metrics(base, y); numbers[title]["base_rate"] = mb
        ax.plot([], [], " ", label=f"base rate (SCE {mb['sce']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("forecast probability correct"); ax.legend(fontsize=5.0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axes[0].set_ylabel("observed frequency correct")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    Path(f"{path}_numbers.json").write_text(json.dumps({"bins": N_BINS, "min_bin_count_plotted": 20, "cases": numbers}, indent=1))
    for t, v in numbers.items():
        print(t.replace(chr(10), ' '), {k: round(m["sce"], 4) for k, m in v.items()})


if __name__ == "__main__":
    main()
