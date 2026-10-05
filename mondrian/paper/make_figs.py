"""Paper-side figures drawn from the locked run outputs (no experiment code, no reruns).

  python paper/make_figs.py      -> paper/figs/fig_overview.{pdf,png}, fig_overview_numbers.json

Reads runs/ only (experiments locked 4 Oct, reports/LOCK_2026-10-04.md) and writes new filenames,
so the manifest-tracked figures and runs/run_all_manifest.json stay valid.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 7.2, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
MODELS = [("llama3_8b", "Llama"), ("mistral_7b", "Mistral"), ("qwen2_5_7b", "Qwen")]
INK, REV, ALT = "#4d4d4d", "#d95f02", "#7570b3"
ARMCOL = {"naive": "#d95f02", "ot_quantile": "#66a61e", "paired_iso": "#1b9e77", "recal_B_k120": "#4d4d4d"}
ARMLAB = {"naive": "no adjustment", "ot_quantile": "quantile map", "paired_iso": "paired isotonic", "recal_B_k120": "120 labels"}


def J(p):
    return json.loads((ROOT / p).read_text())


def main(path=ROOT / "paper/figs/fig_overview"):
    tqa = J("runs/p1_e2_frozen/e2_frozen_result.json"); pop = J("runs/p1_e9/e9a_e2frozen.json")
    e5 = J("runs/p1_e5v2/e5v2_result.json"); e8 = J("runs/p1_e8/e8_result.json")
    e9 = J("runs/p1_e9/e9c_e8_llama3_8b_mistral_7b_qwen2_5_7b.json")
    el = {m: J(f"runs/p1_e9/e9b_e5_{m}.json")[m] for m, _ in MODELS}
    nums = {}
    fig, axs = plt.subplots(2, 2, figsize=(6.75, 4.7), constrained_layout=True)
    axes = axs.ravel()

    # (a) coverage when the procedure changes, TruthfulQA and PopQA
    ax = axes[0]; w = .26; groups = [(tqa, "TQA"), (pop, "PopQA")]
    x0 = 0; ticks = []; nums["a"] = {}
    for src, dname in groups:
        for m, short in MODELS:
            for k, (arm, col) in enumerate((("fwd->fwd", INK), ("fwd->rev", REV), ("fwd->alt", ALT))):
                v = src[m][arm]["coverage_mean"]; nums["a"][f"{dname} {short} {arm}"] = v
                ax.bar(x0 + (k - 1) * w, v, w, color=col, label=None)
            ticks.append((x0, short)); x0 += 1
        x0 += .4
    ax.axhline(.9, color="black", ls=":", lw=.8); ax.set_ylim(.4, 1.0)
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([n for _, n in ticks], fontsize=6.4)
    ax.text(1, 1.005, "TruthfulQA", ha="center", va="bottom", fontsize=6.4, color="#555555")
    ax.text(4.4, 1.005, "PopQA", ha="center", va="bottom", fontsize=6.4, color="#555555")
    for lab, col in (("same procedure", INK), ("legend reversed", REV), ("prompt reworded", ALT)):
        ax.bar([np.nan], [np.nan], color=col, label=lab)
    ax.set_ylabel("conformal coverage (target 0.90)")
    ax.set_title("(a) coverage after calibrating under the forward legend", fontsize=7.2, pad=12)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    # (b) Brier difference against recalibration at the largest budget
    ax = axes[1]; nums["b"] = {}
    arms = [("G4-R", "groups", "#1b9e77"), ("Rand4-R", "random groups", "#999999"), ("MC-R", "multicalib.", "#e7298a"),
            ("D-R", "direct pred.", "#d95f02")]
    keys = [(e5[f"tqa_{m}"], "272", s) for m, s in MODELS] + [(el[m], "2396", s) for m, s in MODELS]
    for k, (c, lab, col) in enumerate(arms):
        x = np.arange(6) + (k - 1.5) * .17
        mm = [S["contrasts"][n][c]["mean"] * 1e3 for S, n, _ in keys]
        lo = [S["contrasts"][n][c]["ci95"][0] * 1e3 for S, n, _ in keys]; hi = [S["contrasts"][n][c]["ci95"][1] * 1e3 for S, n, _ in keys]
        nums["b"][c] = mm
        ax.errorbar(x, mm, yerr=[np.subtract(mm, lo), np.subtract(hi, mm)], fmt="o", ms=2.6, lw=.7, color=col, label=lab)
    ax.axhline(0, color="black", lw=.6); ax.axvline(2.5, color="#bbbbbb", lw=.6)
    ax.set_xticks(range(6)); ax.set_xticklabels([s for *_, s in keys], fontsize=6.4)
    ax.text(1, 1.0, "TruthfulQA, 272 labels", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.4, color="#555555")
    ax.text(4, 1.0, "PopQA, 2,396 labels", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.4, color="#555555")
    ax.set_ylabel("Brier difference vs. recalibration\n($\\times10^3$; below 0 = better)")
    ax.set_title("(b) groups and alternatives against recalibration", fontsize=7.2, pad=12)
    ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)

    # (c) transfer: coverage with set size printed
    ax = axes[2]; nums["c"] = {}
    cases = [(e8["frozen"]["mistral_7b"]["rev"], "Mistral, TQA"), (e9["mistral_7b"]["rev"], "Mistral, PopQA"),
             (e9["qwen2_5_7b"]["rev"], "Qwen, PopQA"), (e8["tinker"]["q"], "trained q, TriviaQA")]
    for k, a in enumerate(ARMCOL):
        x = np.arange(4) + (k - 1.5) * .2
        cov = [c[a]["coverage"] for c, _ in cases]; size = [c[a]["set_size"] for c, _ in cases]
        nums["c"][a] = {"coverage": cov, "set_size": size}
        full = np.array(size) >= 1.9
        ax.scatter(x[~full], np.array(cov)[~full], s=12, color=ARMCOL[a], label=ARMLAB[a], zorder=3)
        ax.scatter(x[full], np.array(cov)[full], s=12, facecolors="none", edgecolors=ARMCOL[a], lw=.9, zorder=3)
    ax.axhline(.9, color="black", ls=":", lw=.8); ax.set_ylim(.15, 1.12)
    ax.set_xticks(range(4)); ax.set_xticklabels([n.replace(", ", "\n") for _, n in cases], fontsize=6.4)
    ax.set_ylabel("coverage (hollow: near-full sets)")
    ax.set_title("(c) carrying a calibration to the reversed legend", fontsize=7.2)
    ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)

    # (d) worst frozen group, net of the no-shift baseline
    ax = axes[3]; nums["d"] = {}
    gcases = [(e8["frozen"]["mistral_7b"]["rev"], "Mistral, TQA"), (e9["mistral_7b"]["rev"], "Mistral, PopQA"),
              (e9["qwen2_5_7b"]["rev"], "Qwen, PopQA")]
    for k, a in enumerate(("ot_quantile", "paired_iso", "recal_B_k120")):
        x = np.arange(3) + (k - 1) * .22
        v = [c[a]["cell_gap_min"] - c["A->A"]["cell_gap_min"] for c, _ in gcases]
        nums["d"][a] = v
        ax.bar(x, v, .2, color=ARMCOL[a], label=ARMLAB[a])
    ax.axhline(0, color="black", lw=.6); ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.set_xticks(range(3)); ax.set_xticklabels([n.replace(", ", "\n") for _, n in gcases], fontsize=6.4)
    ax.set_ylabel("worst-group coverage gap\n(net of no-shift baseline)")
    ax.set_title("(d) worst of the four frozen groups (reversed legend)", fontsize=7.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    Path(f"{path}_numbers.json").write_text(json.dumps(nums, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
