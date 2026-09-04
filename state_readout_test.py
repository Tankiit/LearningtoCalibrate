"""Cached test of whether letter and digit reports retrieve the same state.

Frozen analysis: h_pos -> V_pos, PCA-64 (or the dimensionality guard), ridge
alpha=1, and the existing three held-out splits. No calibration or AUROC is
computed here.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from experiments.ladder import _split  # noqa: E402

OUT = HERE / "cached_results" / "state_readout"


def main():
    rows = []
    pred_rows = []
    comparison_rows = []
    for dataset in ("truthfulqa", "pavlick_nli"):
        cell = ROOT / "outputs" / "step1_extract" / "llama3_8b" / dataset
        h = torch.load(cell / "hidden_states.pt", map_location="cpu", weights_only=False)
        letter = torch.load(cell / "logprobs.pt", map_location="cpu", weights_only=False)
        digit = torch.load(cell / "fine_conf.pt", map_location="cpu", weights_only=False)
        if not np.array_equal(letter["example_ids"], digit["example_ids"]):
            raise ValueError(f"unaligned IDs: {dataset}")
        X = np.asarray(h["h_pos"], dtype=float)
        ys = {"letter11": np.asarray(letter["V_pos"], dtype=float),
              "digits101": np.asarray(digit["V_pos"], dtype=float)}
        for seed in (0, 1, 2):
            train, test = _split(len(X), seed)
            k = min(64, X.shape[1], max(2, X[train].shape[0] // 10))
            # One shared state basis per fold. The two heads are the only
            # format-specific fitted objects.
            transform = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed))
            Xtr, Xte = transform.fit_transform(X[train]), transform.transform(X[test])
            fitted = {}
            for fmt, y in ys.items():
                model = Ridge(alpha=1.0).fit(Xtr, y[train])
                pred = model.predict(Xte)
                fitted[fmt] = model
                rows.append({"dataset": dataset, "seed": seed, "format": fmt,
                             "n_items": len(X), "n_test": int(test.sum()),
                             "pca_components": k, "r2": r2_score(y[test], pred)})
                for local, item in enumerate(np.flatnonzero(test)):
                    pred_rows.append({"dataset": dataset, "seed": seed,
                                      "item_index": int(item), "format": fmt,
                                      "observed": float(y[item]), "predicted": float(pred[local])})
            a, b = fitted["letter11"].coef_, fitted["digits101"].coef_
            pred_l = fitted["letter11"].predict(Xte)
            pred_d = fitted["digits101"].predict(Xte)
            train_ix = np.flatnonzero(train).copy()
            np.random.default_rng(9000 + seed).shuffle(train_ix)
            half_a, half_b = train_ix[:len(train_ix)//2], train_ix[len(train_ix)//2:]
            split_coefs = {}
            for fmt, y in ys.items():
                split_coefs[fmt] = []
                for half in (half_a, half_b):
                    split_coefs[fmt].append(Ridge(alpha=1.0).fit(Xtr[np.isin(np.flatnonzero(train), half)], y[half]).coef_)
            split_cos = {}
            for fmt, coefs in split_coefs.items():
                u, v = coefs
                split_cos[fmt] = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            comparison_rows.append({"dataset": dataset, "seed": seed,
                                    "pca_components": k,
                                    "weight_cosine_shared_pc": float(np.dot(a, b) /
                                        (np.linalg.norm(a) * np.linalg.norm(b))),
                                    "split_half_cosine_letter": split_cos["letter11"],
                                    "split_half_cosine_digit": split_cos["digits101"],
                                    "heldout_prediction_pearson": float(np.corrcoef(pred_l, pred_d)[0, 1]),
                                    "heldout_prediction_spearman": float(spearmanr(pred_l, pred_d).statistic)})
        rows.append({"dataset": dataset, "seed": "paired_all_items", "format": "letter11_vs_digits101",
                     "n_items": len(X), "n_test": len(X), "pca_components": "",
                     "r2": "", "pearson": np.corrcoef(ys["letter11"], ys["digits101"])[0, 1],
                     "spearman": spearmanr(ys["letter11"], ys["digits101"]).statistic})
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "cross_fitted_predictions.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out / "shared_basis_comparisons.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved {out / 'per_seed.csv'}")


if __name__ == "__main__":
    main()
