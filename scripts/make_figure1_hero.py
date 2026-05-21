"""
Figure 1: Hero figure for the paper.

Two panels:
  (a) Per-item scatter of (p_pred, p_defer) on MedQA / LLaMA, colored by
      model correctness. This is loaded from the real independence-panel data.
  (b) Slope chart of mean AURC across seeds: f_pred vs. gap composite, one line
      per (model, dataset) cell. This is loaded from the real aggregate results.

Outputs:
  figures/figure1_hero.pdf
  figures/figure1_hero.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"

MEDQA_LLAMA_PATH = (
    PROJECT_ROOT / "data" / "independence_panels" / "scatter_medqa_llama3_8b.dat"
)
AGGREGATED_RESULTS_PATH = PROJECT_ROOT / "outputs" / "aggregated_results.parquet"
PATH2_DIR = PROJECT_ROOT / "results" / "path2_matrix"

MODEL_ORDER = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DATASET_ORDER = ["chaosnli", "pavlick_nli", "pubmedqa", "medqa", "popqa", "truthfulqa"]

MODEL_DISPLAY = {
    "llama3_8b": "LLaMA",
    "mistral_7b": "Mistral",
    "qwen2_5_7b": "Qwen",
}
DATASET_DISPLAY = {
    "chaosnli": "ChaosNLI",
    "pavlick_nli": "Pavlick-NLI",
    "pubmedqa": "PubMedQA",
    "medqa": "MedQA",
    "popqa": "PopQA",
    "truthfulqa": "TruthfulQA",
}

REGIME_COLORS = {
    "wide": "#E15759",
    "borderline": "#F1A340",
    "tight": "#4E79A7",
}
SCATTER_BLUE = "#4E79A7"
SCATTER_RED = "#E15759"


def load_medqa_llama_scatter() -> pd.DataFrame:
    """Load real per-item probe outputs for MedQA / LLaMA."""
    if not MEDQA_LLAMA_PATH.exists():
        raise FileNotFoundError(
            f"Missing scatter data: {MEDQA_LLAMA_PATH}. "
            "Run scripts/compute_independence_panels.py first."
        )
    df = pd.read_csv(MEDQA_LLAMA_PATH, sep=r"\s+")
    required = {"p_pred", "p_defer", "y_correct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{MEDQA_LLAMA_PATH} missing columns: {sorted(missing)}")
    return df.rename(columns={"y_correct": "y_model"})


def _mean_bdd_median(dataset: str) -> float:
    medians = []
    for model in MODEL_ORDER:
        path = PATH2_DIR / f"path2_diagnostic_summary_{model}_{dataset}.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        medians.append(float(obj["ours"]["width_stats"]["defer_width_median"]))
    if not medians:
        raise FileNotFoundError(f"No Path2 BDD summaries found for {dataset}")
    return sum(medians) / len(medians)


def _bdd_regime(dataset: str) -> str:
    """Classify BDD regime from real mean median defer-width."""
    median = _mean_bdd_median(dataset)
    if median >= 0.15:
        return "wide"
    if median >= 0.075:
        return "borderline"
    return "tight"


def load_aurc_table() -> pd.DataFrame:
    """Load real mean AURC across seeds for f_pred and gap."""
    if not AGGREGATED_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing aggregate results: {AGGREGATED_RESULTS_PATH}")
    df = pd.read_parquet(AGGREGATED_RESULTS_PATH)
    required = {"model_key", "dataset", "method", "aurc", "seed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{AGGREGATED_RESULTS_PATH} missing columns: {sorted(missing)}")

    sub = df[
        df["model_key"].isin(MODEL_ORDER)
        & df["dataset"].isin(DATASET_ORDER)
        & df["method"].isin(["probe_pred_only", "gap"])
    ]
    means = (
        sub.groupby(["model_key", "dataset", "method"], as_index=False)["aurc"]
        .mean()
        .pivot(index=["model_key", "dataset"], columns="method", values="aurc")
        .reset_index()
    )
    means = means.rename(columns={"probe_pred_only": "pred_only"})

    rows = []
    missing_cells = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            row = means[(means["model_key"] == model) & (means["dataset"] == dataset)]
            if row.empty:
                missing_cells.append(f"{model}/{dataset}")
                continue
            item = row.iloc[0]
            rows.append({
                "model": MODEL_DISPLAY[model],
                "dataset": DATASET_DISPLAY[dataset],
                "model_key": model,
                "dataset_key": dataset,
                "pred_only": float(item["pred_only"]),
                "gap": float(item["gap"]),
                "bdd_regime": _bdd_regime(dataset),
            })
    if missing_cells:
        raise ValueError(f"Missing AURC cells in {AGGREGATED_RESULTS_PATH}: {missing_cells}")
    return pd.DataFrame(rows)


def plot_panel_a(ax: plt.Axes, scatter_df: pd.DataFrame) -> None:
    correct_mask = scatter_df["y_model"].to_numpy(dtype=int) == 1
    ax.scatter(
        scatter_df.loc[correct_mask, "p_pred"],
        scatter_df.loc[correct_mask, "p_defer"],
        c=SCATTER_BLUE,
        alpha=0.45,
        s=10,
        edgecolors="none",
        label="Model correct",
    )
    ax.scatter(
        scatter_df.loc[~correct_mask, "p_pred"],
        scatter_df.loc[~correct_mask, "p_defer"],
        c=SCATTER_RED,
        alpha=0.45,
        s=10,
        edgecolors="none",
        label="Model wrong",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_{\mathrm{pred}}$ (model-correctness probe)")
    ax.set_ylabel(r"$p_{\mathrm{defer}}$ (expert-reliability probe)")
    ax.set_title("(a) Independence per-item: MedQA / LLaMA", loc="left", pad=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.92,
        markerscale=2.5,
        handletextpad=0.3,
    )
    legend.get_frame().set_edgecolor("#CCCCCC")

    ax.text(
        0.03,
        0.96,
        "$p_{\\mathrm{pred}}$ separates blue (correct) from red (wrong).\n"
        "$p_{\\mathrm{defer}}$ does not separate them.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.92,
        ),
    )


def plot_panel_b(ax: plt.Axes, aurc_df: pd.DataFrame) -> None:
    for _, row in aurc_df.iterrows():
        color = REGIME_COLORS[row["bdd_regime"]]
        ax.plot(
            [0, 1],
            [row["pred_only"], row["gap"]],
            color=color,
            alpha=0.75,
            linewidth=1.4,
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$f_{\mathrm{pred}}$", r"$\Delta$ (gap)"])
    ax.set_xlim(-0.25, 1.25)

    y_min = float(aurc_df[["pred_only", "gap"]].min().min()) - 0.02
    y_max = float(aurc_df[["pred_only", "gap"]].max().max()) + 0.02
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("AURC (lower is better)")
    ax.set_title("(b) Universal underperformance: 18/18 cells", loc="left", pad=10)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=REGIME_COLORS["wide"], lw=2.2, label="Wide BDD"),
        Line2D([0], [0], color=REGIME_COLORS["borderline"], lw=2.2, label="Borderline"),
        Line2D([0], [0], color=REGIME_COLORS["tight"], lw=2.2, label="Tight BDD"),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.92,
        title="BDD regime",
        title_fontsize=9.5,
        handletextpad=0.4,
        borderpad=0.4,
    )
    legend.get_frame().set_edgecolor("#CCCCCC")

    ax.text(
        0.5,
        0.02,
        "All 18 lines slope up.\nSteepest: tight BDD. Shallowest: borderline.",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.92,
        ),
    )


def make_figure() -> None:
    scatter_df = load_medqa_llama_scatter()
    aurc_df = load_aurc_table()

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.1, 0.85],
        wspace=0.28,
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.13,
    )
    plot_panel_a(fig.add_subplot(gs[0, 0]), scatter_df)
    plot_panel_b(fig.add_subplot(gs[0, 1]), aurc_df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / "figure1_hero.pdf"
    out_png = FIGURES_DIR / "figure1_hero.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"Panel A data: {MEDQA_LLAMA_PATH} ({len(scatter_df)} points)")
    print(f"Panel B data: {AGGREGATED_RESULTS_PATH} ({len(aurc_df)} cells)")


if __name__ == "__main__":
    make_figure()
