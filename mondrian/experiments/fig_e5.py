"""Figures for the rewrite: E5 corrected contrasts by budget, and the synthetic phase map."""
import json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"tqa_llama3_8b": "Llama-3.1-8B", "tqa_mistral_7b": "Mistral-7B", "tqa_qwen2_5_7b": "Qwen2.5-7B", "popqa": "PopQA-Probe"}
ELICIT = {"llama3_8b": "PopQA-Elicit, Llama-3.1-8B", "mistral_7b": "PopQA-Elicit, Mistral-7B", "qwen2_5_7b": "PopQA-Elicit, Qwen2.5-7B"}
ARMS = [("G4-R", "groups ($M{=}4$)", "#1b9e77"), ("Gcv-R", "groups, $M$ by CV", "#66a61e"),
        ("Rand4-R", "random cells", "#999999"), ("MC-R", "multicalibration", "#e7298a"), ("D-R", "direct predictor", "#d95f02")]


def fig_e5(path="paper/figs/fig_e5"):
    r = json.load(open("runs/p1_e5v2/e5v2_result.json"))
    r |= {f"elicit_{m}": json.load(open(f"runs/p1_e9/e9b_e5_{m}.json"))[m] for m in ELICIT}
    names = NAMES | {f"elicit_{m}": t for m, t in ELICIT.items()}
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 4.6), constrained_layout=True, sharey=False)
    axes[1, 3].axis("off")
    for ax, key in zip(list(axes[0]) + list(axes[1, :3]), names):
        S = r[key]; ns = sorted(map(int, S["contrasts"]))
        for j, (c, lab, col) in enumerate(ARMS):
            x = np.arange(len(ns)) + (j - 2) * 0.13
            m = [S["contrasts"][str(n)][c]["mean"] for n in ns]
            lo = [S["contrasts"][str(n)][c]["ci95"][0] for n in ns]; hi = [S["contrasts"][str(n)][c]["ci95"][1] for n in ns]
            ax.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o", ms=2.5, lw=.8, color=col, label=lab)
        ax.axhline(0, color="black", lw=.6)
        ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns, fontsize=6.5, rotation=0 if len(ns) < 5 else 35)
        ax.set_xlabel("labelled questions")
        ax.set_title(names[key], fontsize=7.2)
        if key.startswith("elicit_"):   # zoom: the group effect is ~2e-3; small-budget intervals run off the axis
            ax.set_ylim(-0.035, 0.015)
    for a in axes[:, 0]:
        a.set_ylabel("Brier difference vs. recalibration\n(below 0 = better)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=6.5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)


def fig_synth(path="paper/figs/fig_synth"):
    """Rows: structure; columns: between-group spread tau; x: labels; lines: report informativeness kappa.
    Solid: groups minus recalibration; dashed: groups minus direct predictor. Error bars: per-configuration
    95% intervals over 30 independent seeds."""
    rows = json.load(open("runs/p1_synth_v2/synth_summary.json"))["rows"]
    taus, kaps, ns = (0.0, 1.0, 2.0), (0.0, .5, 1.0, 2.0), (60, 120, 240, 480)
    cols = {0.0: "#bdbdbd", .5: "#74a9cf", 1.0: "#2b8cbe", 2.0: "#045a8d"}
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.6), constrained_layout=True, sharex=True)
    for i, st in enumerate(("cluster", "linear")):
        for j, tau in enumerate(taus):
            ax = axes[i, j]; ax.axhline(0, color="black", lw=.6)
            for kap in kaps:
                sel = sorted([q for q in rows if q["structure"] == st and q["tau"] == tau and q["kappa"] == kap], key=lambda q: q["n_lab"])
                x = np.log2([q["n_lab"] for q in sel]) + (kap - 1) * 0.04
                for key, ls in (("G-R", "-"), ("G-D", "--")):
                    m = np.array([q[key] for q in sel]) * 1e3
                    lo = np.array([q[key + "_ci"][0] for q in sel]) * 1e3; hi = np.array([q[key + "_ci"][1] for q in sel]) * 1e3
                    ax.errorbar(x, m, yerr=[m - lo, hi - m], ls=ls, marker="o", ms=2, lw=.8, color=cols[kap],
                                label=(f"$\\kappa={kap:g}$" if key == "G-R" else None))
            ax.set_xticks(np.log2(ns)); ax.set_xticklabels(ns, fontsize=6.5)
            ax.set_title(f"{st} reliability, $\\tau={tau:g}$", fontsize=7.2)
            if j == 0: ax.set_ylabel("Brier diff. $\\times10^3$")
            if i == 1: ax.set_xlabel("labelled examples")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
               title="report informativeness (solid: groups $-$ recalibration; dashed: groups $-$ direct predictor)", title_fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)




def fig_main(path="paper/figs/fig_main"):
    """Overview: (a) elicitation change breaks coverage; (b) groups vs recalibration vs direct predictor;
    (c) label-free transport repairs coverage."""
    nulls = json.load(open("runs/p1_nulls/nulls.json"))["e2_frozen"]
    e5 = json.load(open("runs/p1_e5v2/e5v2_result.json")); e8 = json.load(open("runs/p1_e8/e8_result.json"))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.75), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.05, 1.2]})
    models = [("llama3_8b", "Llama"), ("mistral_7b", "Mistral"), ("qwen2_5_7b", "Qwen")]
    # (a)
    ax = axes[0]; w = .26
    for k, (arm, lab, col) in enumerate([("fwd->fwd", "same", "#4d4d4d"), ("fwd->rev", "legend reversed", "#d95f02"),
                                          ("fwd->alt", "prompt reworded", "#7570b3")]):
        m = [nulls[mk][arm]["mean"] for mk, _ in models]
        lo = [nulls[mk][arm]["q10_q90"][0] for mk, _ in models]; hi = [nulls[mk][arm]["q10_q90"][1] for mk, _ in models]
        x = np.arange(3) + (k - 1) * w
        ax.bar(x, m, w, color=col, label=lab)
        ax.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="none", ecolor="black", lw=.6)
    ax.axhline(.9, color="black", ls=":", lw=.8); ax.set_ylim(.4, 1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in models]); ax.set_ylabel("coverage (target 0.90)")
    ax.set_title("(a) calibrated under forward legend", fontsize=7.2)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    # (b)
    ax = axes[1]
    arms = [("G4-R", "groups", "#1b9e77"), ("Rand4-R", "random groups", "#999999"), ("MC-R", "multicalib.", "#e7298a"), ("D-R", "direct pred.", "#d95f02")]
    src = e5 | {f"elicit_{m}": json.load(open(f"runs/p1_e9/e9b_e5_{m}.json"))[m] for m in ("llama3_8b", "mistral_7b", "qwen2_5_7b")}
    keys = [("tqa_llama3_8b", "272", "Llama"), ("tqa_mistral_7b", "272", "Mistral"), ("tqa_qwen2_5_7b", "272", "Qwen"),
            ("elicit_llama3_8b", "2396", "Llama"), ("elicit_mistral_7b", "2396", "Mistral"), ("elicit_qwen2_5_7b", "2396", "Qwen")]
    for k, (c, lab, col) in enumerate(arms):
        x = np.arange(6) + (k - 1.5) * .17
        m = [src[key]["contrasts"][n][c]["mean"] * 1e3 for key, n, _ in keys]
        lo = [src[key]["contrasts"][n][c]["ci95"][0] * 1e3 for key, n, _ in keys]; hi = [src[key]["contrasts"][n][c]["ci95"][1] * 1e3 for key, n, _ in keys]
        ax.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o", ms=3, lw=.8, color=col, label=lab)
    ax.axhline(0, color="black", lw=.6); ax.axvline(2.5, color="#bbbbbb", lw=.6)
    ax.set_xticks(range(6)); ax.set_xticklabels([n for *_, n in keys], fontsize=6, rotation=30)
    ax.text(1, 1.0, "TruthfulQA\n272 labels", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8)
    ax.text(4, 1.0, "PopQA-Elicit\n2,396 labels", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8)
    ax.set_ylabel("Brier difference vs. recalibration\n($\\times10^3$; below 0 = better)"); ax.set_title("(b) against recalibration\n", fontsize=7.2)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    # (c)
    ax = axes[2]
    cases = [(e8["frozen"]["mistral_7b"]["rev"], "Mistral\nreversed"), (e8["frozen"]["mistral_7b"]["alt"], "Mistral\nreworded"),
             (e8["tinker"]["q"], "trained q\nreversed"), (e8["tinker"]["qa"], "trained qa\nreversed")]
    arms = [("naive", "no adjustment", "#d95f02"), ("ot_quantile", "quantile map", "#66a61e"), ("paired_iso", "paired isotonic", "#1b9e77"),
            ("recal_B_k120", "60+60 labels", "#4d4d4d")]
    for k, (a, lab, col) in enumerate(arms):
        x = np.arange(4) + (k - 1.5) * .2
        cov = [c[a]["coverage"] for c, _ in cases]; size = [c[a]["set_size"] for c, _ in cases]
        ax.scatter(x, cov, s=[8 + 22 * (sz - 1) for sz in size], color=col, label=lab, zorder=3)
    ax.axhline(.9, color="black", ls=":", lw=.8); ax.set_ylim(.15, 1.02)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in cases], fontsize=6)
    ax.set_ylabel("coverage"); ax.set_title("(c) label-free transfer", fontsize=7.2)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)


if __name__ == "__main__":
    fig_e5(); fig_synth(); fig_main()
