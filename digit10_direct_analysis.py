"""Pre-registered analysis for the digit10-direct format arm.

The elicitation pass is external (Modal). This script deliberately does not
impute missing cells or choose a fold after seeing results. It consumes one
``digit10_direct.pt`` per cell, paired with the existing letter11 cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from experiments.ladder import _split  # noqa: E402
from scripts.step_token_contribution import aurc, build_targets, fit_predict, load_cell  # noqa: E402
from utils.paths import OUTPUTS_ROOT, logprobs_path  # noqa: E402
from report_channel_robustness import _adequacy  # noqa: E402

MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
DATASETS = ("truthfulqa", "pavlick_nli")
SEEDS = (0, 1, 2)
OUT = HERE / "cached_results" / "digit10_direct"


def direct_path(model: str, dataset: str) -> Path:
    return OUTPUTS_ROOT / "step1_extract" / model / dataset / "digit10_direct.pt"


def _ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def _bootstrap_delta(letter: np.ndarray, direct: np.ndarray, sq: np.ndarray,
                     risk: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    """Item bootstrap for C = AURC(digit) - AURC(letter)."""
    rng = np.random.default_rng(seed)
    samples = []
    for ix in rng.integers(0, len(risk), size=(n_boot, len(risk))):
        samples.append(aurc(direct[ix], risk[ix]) - aurc(letter[ix], risk[ix]))
    return _ci(np.asarray(samples))


def analyse_cell(model: str, dataset: str, fold: str = "test") -> list[dict]:
    d, err = load_cell(model, dataset)
    if err:
        raise FileNotFoundError(err["error"])
    letter = torch.load(logprobs_path(model, dataset), map_location="cpu", weights_only=False)
    path = direct_path(model, dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing preregistered direct cache: {path}")
    digit = torch.load(path, map_location="cpu", weights_only=False)
    ids_l = np.asarray(letter["example_ids"])
    ids_d = np.asarray(digit["example_ids"])
    if not np.array_equal(ids_l, ids_d):
        raise ValueError(f"item IDs are not aligned for {model}/{dataset}")
    if list(np.asarray(digit["conf_values"]).tolist()) != list(range(0, 100, 10)):
        raise ValueError("direct cache does not have frozen values 0..90 by 10")
    y = build_targets(d)["mean"][0]
    rows = []
    for seed in SEEDS:
        tr, te = _split(len(y), seed)
        fit, ev = (tr, te) if fold == "test" else (te, tr)
        sq = fit_predict(d["h_q"][fit], y[fit], d["h_q"][ev], seed)
        ix = np.flatnonzero(ev)
        risk = 1.0 - y[ix]
        for readout, lkey, dkey in (("V_pos", "V_pos", "V_pos"),
                                    ("V_pos_minus_V_neg", "gap", "gap")):
            lscore = (np.asarray(letter["V_pos"]) if lkey == "V_pos"
                      else np.asarray(letter["V_pos"]) - np.asarray(letter["V_neg"]))[ix]
            dscore = (np.asarray(digit["V_pos"]) if dkey == "V_pos"
                      else np.asarray(digit["V_pos"]) - np.asarray(digit["V_neg"]))[ix]
            c = aurc(dscore, risk) - aurc(lscore, risk)
            lo, hi = _bootstrap_delta(lscore, dscore, sq, risk, 100000 + seed)
            rows.append({"model": model, "dataset": dataset, "fold": fold,
                         "seed": seed, "readout": readout, "n_eval": len(ix),
                         "C": c, "C_boot_lo": lo, "C_boot_hi": hi,
                         "letter_adequacy": _adequacy(lscore),
                         "digit10_adequacy": _adequacy(dscore)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=("test", "train"), default="test")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            rows.extend(analyse_cell(model, dataset, args.fold))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "conf_scheme": "digit10-direct", "fold": args.fold,
        "models": list(MODELS), "datasets": list(DATASETS), "seeds": list(SEEDS),
        "primary": "C on V_pos", "secondary": "C on V_pos_minus_V_neg",
        "C_definition": "AURC(digit10) - AURC(letter11)",
        "cross_format_level_comparisons": "forbidden",
        "bootstrap": {"unit": "item", "interval": "percentile", "level": 0.95},
    }, indent=2) + "\n")
    print(f"wrote {out / 'per_seed.csv'}")


if __name__ == "__main__":
    main()
