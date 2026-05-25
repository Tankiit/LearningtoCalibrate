import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MODEL_COLORS = {"LLaMA": "#4C78A8", "Mistral": "#F58518", "Qwen": "#54A24B"}
REGIME_COLOR = {"tight": "#0072B2", "borderline": "#E69F00", "wide": "#D55E00"}
MODEL_MARKER = {"LLaMA": "o", "Mistral": "s", "Qwen": "^"}
DATASET_COLOR = {
    "ChaosNLI": "#0072B2", "Pavlick-NLI": "#56B4E9", "PubMedQA": "#E69F00",
    "MedQA": "#D55E00", "PopQA": "#CC79A7", "TruthfulQA": "#009E73",
}

def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_aurc_slope():
    df = pd.read_csv(DATA / "aurc_results.csv")
    sub = df[df.method.isin(["probe_pred_only", "gap"])]
    mean = sub.groupby(["model_key", "dataset", "method"], as_index=False).aurc.mean()
    wide = mean.pivot(index=["model_key", "dataset"], columns="method", values="aurc").reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for _, r in wide.iterrows():
        ax.plot([0, 1], [r.probe_pred_only, r.gap], color="0.35", alpha=0.55, marker="o", markersize=3)
    ax.set_xticks([0, 1], [r"$f_{\mathrm{pred}}$", r"$\Delta$ (gap)"])
    ax.set_ylabel("AURC (lower is better)")
    ax.set_title("Selective-prediction AURC by scoring rule", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "aurc_slope")

def plot_independence():
    cosines = pd.read_csv(DATA / "cosines_by_cell.csv")
    signal = pd.read_csv(DATA / "signal_corr_by_cell.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), constrained_layout=True)
    for ax, df, title, ylabel in [
        (axes[0], cosines, "Geometric non-independence", r"$|\cos(w_{pred},w_{defer})|$"),
        (axes[1], signal, "Signal-output correlation", r"$|\rho(p_{pred},p_{defer})|$"),
    ]:
        x = np.arange(len(df))
        width = 0.24
        for i, model in enumerate(["LLaMA", "Mistral", "Qwen"]):
            ax.bar(x + (i - 1) * width, df[model], width=width, color=MODEL_COLORS[model], label=model)
        ax.set_xticks(x, df.dataset, rotation=35, ha="right")
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    save(fig, "independence_bars")

def plot_diagnostic_composite():
    rows = pd.read_csv(DATA / "diagnostic_probe_cells.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for _, r in rows.iterrows():
        axes[0].scatter(r.fdefer_acc, r.gap_penalty, color=REGIME_COLOR[r.regime],
                        marker=MODEL_MARKER[r.model], edgecolor="black", s=40)
        axes[1].scatter(r.bdd_median, r.gap_penalty, color=DATASET_COLOR[r.dataset],
                        marker=MODEL_MARKER[r.model], edgecolor="black", s=40)
        axes[2].scatter(r.fpred_auroc, r.fdefer_auroc, color=DATASET_COLOR[r.dataset],
                        marker=MODEL_MARKER[r.model], edgecolor="black", s=40)
    axes[0].set_xlabel(r"$f_{\mathrm{defer}}$ accuracy")
    axes[0].set_ylabel("Gap penalty")
    axes[0].set_title("(a) Accuracy vs. penalty", loc="left")
    axes[0].plot(*np.polyfit(rows.fdefer_acc, rows.gap_penalty, 1)[::-1], alpha=0)
    x = np.linspace(rows.fdefer_acc.min(), rows.fdefer_acc.max(), 100)
    m, b = np.polyfit(rows.fdefer_acc, rows.gap_penalty, 1)
    axes[0].plot(x, m*x + b, color="0.55", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("BDD median width")
    axes[1].set_ylabel("Gap penalty")
    axes[1].set_title("(b) Width vs. penalty", loc="left")
    axes[2].plot([0.5, 1.0], [0.5, 1.0], color="0.6", ls=":")
    axes[2].set_xlabel(r"$f_{\mathrm{pred}}$ AUROC")
    axes[2].set_ylabel(r"$f_{\mathrm{defer}}$ AUROC")
    axes[2].set_title("(c) Native-target AUROC", loc="left")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "diagnostic_composite")

def plot_stc_vs_alpha():
    df = pd.read_csv(DATA / "stc_vs_alpha.csv")
    summary = df.groupby("alpha", as_index=False).agg(stc_mean=("stc_mean", "mean"))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for _, cell in df.groupby(["model", "dataset"]):
        ax.plot(cell.alpha, cell.stc_p50, color="0.75", lw=0.6, alpha=0.6)
    ax.plot(summary.alpha, summary.stc_mean, color="#0072B2", lw=1.8, label="mean")
    ax.axhline(0.10, color="#D55E00", ls="--", lw=0.9, label="STC = 0.10")
    ax.set_xlabel(r"Injected alignment $\alpha$")
    ax.set_ylabel(r"STC")
    ax.set_title("STC calibration", loc="left")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "stc_vs_alpha")

if __name__ == "__main__":
    plot_aurc_slope()
    plot_independence()
    plot_diagnostic_composite()
    plot_stc_vs_alpha()
    print(f"Wrote figures to {FIG}")
