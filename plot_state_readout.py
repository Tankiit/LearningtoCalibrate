"""Plot the shared-state/readout test: rank agreement and cross-fitted R2."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "cached_results" / "state_readout"


def main():
    pred = pd.read_csv(OUT / "cross_fitted_predictions.csv")
    r2 = pd.read_csv(OUT / "per_seed.csv")
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    for row, dataset in enumerate(("truthfulqa", "pavlick_nli")):
        cell = ROOT / "outputs" / "step1_extract" / "llama3_8b" / dataset
        letter = torch.load(cell / "logprobs.pt", map_location="cpu", weights_only=False)
        digit = torch.load(cell / "fine_conf.pt", map_location="cpu", weights_only=False)
        vl, vd = np.asarray(letter["V_pos"], float), np.asarray(digit["V_pos"], float)
        x, y = rankdata(vl) / (len(vl) + 1), rankdata(vd) / (len(vd) + 1)
        rho = spearmanr(vl, vd).statistic
        ax[row, 0].scatter(x, y, s=5, alpha=.28, color="#333333", linewidths=0)
        ax[row, 0].plot([0, 1], [0, 1], color="#b2182b", lw=1)
        ax[row, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="letter rank", ylabel="digit rank",
                        title=f"ordering  ρ = {rho:.3f}")
        for col, fmt in ((1, "letter11"), (2, "digits101")):
            sub = pred[(pred.dataset == dataset) & (pred.format == fmt)]
            ax[row, col].scatter(sub.observed, sub.predicted, s=5, alpha=.20,
                                 color="#2166ac" if fmt == "letter11" else "#4d9221",
                                 linewidths=0)
            ax[row, col].plot([0, 1], [0, 1], color="#888888", lw=.8)
            vals = r2[(r2.dataset == dataset) & (r2.format == fmt)].r2
            ax[row, col].set(xlim=(0, 1), ylim=(0, 1), xlabel="observed V⁺",
                              ylabel="cross-fitted prediction",
                              title=f"h → {fmt}  R² = {vals.mean():.3f} ± {vals.std(ddof=1):.3f}")
        ax[row, 0].text(.03, .96, dataset, transform=ax[row, 0].transAxes,
                        va="top", fontweight="bold")
    fig.suptitle("Same pre-confidence answer state, different report readouts", fontsize=13)
    fig.savefig(OUT / "state_readout_2x3.png", dpi=220)
    fig.savefig(OUT / "state_readout_2x3.pdf")
    print(f"saved {OUT / 'state_readout_2x3.png'}")


if __name__ == "__main__":
    main()
