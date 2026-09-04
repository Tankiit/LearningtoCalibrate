"""Exploratory TRAIN-fold-only partition-invariance test.

The threshold is frozen below before ``main`` can print a result.  This test
uses the existing Llama/TruthfulQA caches and does not run a forward pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures_method import load_V
from experiments.ladder import _split
from scripts.step_token_contribution import aurc


# Frozen decision rule: ratio below one half of the reliability ceiling is
# treated as disagreement beyond instrument noise.
RATIO_THRESHOLD = 0.50
MODEL = "llama3_8b"
DATASET = "truthfulqa"
CELL = ROOT / "outputs" / "step1_extract" / MODEL / DATASET
OUT = Path(__file__).resolve().parent / "cached_results" / "partition_invariance"


def qtransform(v):
    """Within-format quantile transform; monotone, so it cannot change ranks."""
    r = np.argsort(np.argsort(v))
    return (r + 1) / (len(v) + 1)


def rel_spearman(v_a, v_b):
    return float(spearmanr(v_a, v_b).statistic)


def _align(ids_a, ids_b):
    index = {str(item_id): i for i, item_id in enumerate(ids_b)}
    if {str(x) for x in ids_a} != set(index):
        raise ValueError("item IDs do not match")
    return np.array([index[str(item_id)] for item_id in ids_a], dtype=int)


def _train_ids(ids):
    train, _ = _split(len(ids), seed=0)
    return {str(item_id) for item_id in np.asarray(ids, dtype=object)[train]}


def _alternate_reliability(path, train_ids, primary, ids):
    data = torch.load(path, map_location="cpu", weights_only=False)
    alt_index = np.asarray(data["alt_index"], dtype=int)
    alternate = np.asarray(data["Valt_pos"], dtype=float)
    primary_ids = np.asarray(ids, dtype=object)[alt_index]
    keep = np.asarray([str(item_id) in train_ids for item_id in primary_ids])
    if keep.sum() < 20:
        raise ValueError(f"too few training items for alternate reliability: {keep.sum()}")
    return rel_spearman(np.asarray(primary)[alt_index][keep], alternate[keep]), int(keep.sum())


def main():
    assert RATIO_THRESHOLD is not None, "commit the decision rule first"
    if not CELL.exists():
        raise FileNotFoundError(CELL)

    letter_path = CELL / "logprobs.pt"
    numeric_path = CELL / "fine_conf.pt"
    L_pos, _, L_ids = load_V(letter_path, "letter11")
    D_pos_raw, _, D_ids_raw = load_V(numeric_path, "digits101")
    digit_order = _align(L_ids, D_ids_raw)
    D_pos = D_pos_raw[digit_order]
    D_ids = np.asarray(D_ids_raw, dtype=object)[digit_order]
    train_ids = _train_ids(L_ids)
    train = np.asarray([str(item_id) in train_ids for item_id in L_ids])

    letter_data = torch.load(letter_path, map_location="cpu", weights_only=False)
    numeric_data = torch.load(numeric_path, map_location="cpu", weights_only=False)
    rel_L, n_rel_L = _alternate_reliability(letter_path, train_ids, L_pos, L_ids)
    rel_D, n_rel_D = _alternate_reliability(
        numeric_path, train_ids, D_pos_raw, D_ids_raw)
    rho = rel_spearman(L_pos[train], D_pos[train])
    ceiling = float(np.sqrt(rel_L * rel_D))
    ratio = float(rho / ceiling)

    # The transform is a sanity check, not an additional score.
    y = (np.asarray(letter_data["logprob_pos"], dtype=float)
         > np.asarray(letter_data["logprob_neg"], dtype=float)).astype(int)
    risk = 1.0 - y[train]
    assert np.isclose(aurc(qtransform(L_pos[train]), risk), aurc(L_pos[train], risk)), \
        "quantile transform changed AURC; tie handling is not invariant"

    result = {
        "model": MODEL, "dataset": DATASET, "fold": "seed0_train",
        "n_train": int(train.sum()), "n_rel_letter": n_rel_L,
        "n_rel_numeric": n_rel_D, "rho_cross_spearman": rho,
        "rel_letter_spearman": rel_L, "rel_numeric_spearman": rel_D,
        "ceiling": ceiling, "ratio": ratio,
        "ratio_threshold": RATIO_THRESHOLD,
        "verdict": "fails" if ratio < RATIO_THRESHOLD else "holds",
        "band_letter_before": np.percentile(L_pos[train], [5, 95]).tolist(),
        "band_letter_after": np.percentile(qtransform(L_pos[train]), [5, 95]).tolist(),
        "numeric_reliability_source": "Valt_pos alternate-form cache, same train fold",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
