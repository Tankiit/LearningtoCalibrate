"""Analyze the frozen ``letter11-permuted`` arm without changing its rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from figures_method import load_V
from report_channel_robustness import _adequacy

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
DATASETS = ("truthfulqa", "pavlick_nli")
PERM_ROOT = Path(__file__).resolve().parent / "letter11_permuted_remote"
OUT = Path(__file__).resolve().parent / "cached_results" / "letter11_permuted"
BOOT = 2000
BOOT_SEED = 20260905


def _rho(x, y):
    return float(spearmanr(x, y).statistic)


def _ci_rho(x, y, rng):
    n = len(x)
    draws = rng.integers(0, n, size=(BOOT, n))
    values = np.asarray([_rho(x[ix], y[ix]) for ix in draws])
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _ci_mean(x, rng):
    n = len(x)
    draws = rng.integers(0, n, size=(BOOT, n))
    values = np.asarray(x)[draws].mean(axis=1)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _alt_reliability(path, primary, ids):
    data = torch.load(path, map_location="cpu", weights_only=False)
    alt_index = np.asarray(data["alt_index"], dtype=int)
    alt = np.asarray(data["Valt_pos"], dtype=float)
    alt_ids = np.asarray(ids, dtype=object)[alt_index]
    return _rho(np.asarray(primary)[alt_index], alt), int(len(alt_ids))


def _load_permuted(model, dataset):
    path = PERM_ROOT / f"{model}_{dataset}.pt"
    return load_V(path, "letter11")


def main():
    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    controls = []
    for model in MODELS:
        for dataset in DATASETS:
            cell = ROOT / "outputs" / "step1_extract" / model / dataset
            letter_path = cell / "logprobs.pt"
            letter_pos, letter_neg, letter_ids = load_V(letter_path, "letter11")
            perm_pos, perm_neg, perm_ids = _load_permuted(model, dataset)
            perm_order = {str(item_id): i for i, item_id in enumerate(perm_ids)}
            if set(map(str, letter_ids)) != set(perm_order):
                raise ValueError(f"item IDs do not match for {model}/{dataset}")
            order = np.asarray([perm_order[str(item_id)] for item_id in letter_ids])
            perm_pos, perm_neg = perm_pos[order], perm_neg[order]

            rel_L, n_rel = _alt_reliability(letter_path, letter_pos, letter_ids)
            rho = _rho(letter_pos, perm_pos)
            rho_ci = _ci_rho(letter_pos, perm_pos, rng)
            gap_delta = (perm_pos - perm_neg) - (letter_pos - letter_neg)
            gap_ci = _ci_mean(gap_delta, rng)
            gate_pos = _adequacy(perm_pos)
            gate_gap = _adequacy(perm_pos - perm_neg)
            if not gate_pos["measurable"] or not np.isfinite(rel_L) or rel_L <= 0:
                verdict = "not measurable"
            elif rho_ci[0] >= 0.75 * rel_L:
                verdict = "value-tracking"
            elif rho_ci[1] < 0.50 * rel_L:
                # The statistic is computed after mapping each arm's token
                # distribution into its denoted-value space.  A low
                # cross-arm ordering correlation establishes non-invariance
                # to the legend; this arm cannot identify glyph versus list
                # position because both are held fixed by the reversal.
                verdict = "legend-noninvariant"
            else:
                verdict = "intermediate"
            rows.append({
                "model": model, "dataset": dataset, "n_items": len(letter_pos),
                "candidate_injective": True,
                "Vpos_effective_bins_perm": gate_pos["effective_bins"],
                "gap_effective_bins_perm": gate_gap["effective_bins"],
                "Vpos_measurable_perm": gate_pos["measurable"],
                "gap_measurable_perm": gate_gap["measurable"],
                "rho_Vpos_v1_perm": rho, "rho_ci_lo": rho_ci[0],
                "rho_ci_hi": rho_ci[1], "rel_v1_spearman": rel_L,
                "n_reliability": n_rel, "value_tracking_threshold": 0.75 * rel_L,
                "well_below_threshold": 0.50 * rel_L, "verdict": verdict,
                "gap_delta_mean": float(gap_delta.mean()),
                "gap_delta_ci_lo": gap_ci[0], "gap_delta_ci_hi": gap_ci[1],
            })

            if model == "llama3_8b":
                numeric_path = cell / "fine_conf.pt"
                numeric_pos, _, numeric_ids = load_V(numeric_path, "digits101")
                numeric_order = {str(item_id): i for i, item_id in enumerate(numeric_ids)}
                no = np.asarray([numeric_order[str(item_id)] for item_id in letter_ids])
                numeric_pos = numeric_pos[no]
                controls.append({
                    "dataset": dataset,
                    "rho_v1_numeric101": _rho(letter_pos, numeric_pos),
                    "rho_perm_numeric101": _rho(perm_pos, numeric_pos),
                })

    frame = pd.DataFrame(rows)
    control_frame = pd.DataFrame(controls)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_cell.csv", index=False)
    control_frame.to_csv(OUT / "numeric101_controls.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "arm": "letter11-permuted", "permutation": "reversal",
        "models": list(MODELS), "datasets": list(DATASETS),
        "bootstrap": {"unit": "item", "draws": BOOT, "seed": BOOT_SEED},
        "ratio_threshold": 0.50,
        "decision_basis": "V⁺ perm correlation against recomputed letter11-v1 alternate-form reliability",
        "status": "exploratory; fixed reversal, paired by example_id",
    }, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(control_frame.to_string(index=False))


if __name__ == "__main__":
    main()
