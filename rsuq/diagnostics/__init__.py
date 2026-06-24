"""rsuq.diagnostics — The shared evaluation battery.

One battery, four papers:
  orthogonality_gate   Stage II (Q2.3) and any two-signal decomposition claim
  redundancy_screen    Token-Barycenter paper (signal vs probe), AAAI alignment claim
  smoking_gun_pairs    NeurIPS E1 + token-level variant (matched BetP, differing W)
  detection            AUROC / separation ratio / ECE (all papers' Table 1s)

Everything reports bootstrap CIs; thresholds live in the papers' pre-registration
docs, not here.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr, pointbiserialr


# ----------------------------------------------------------------- core stats
def partial_corr(x, y, z) -> float:
    """corr(x, y | z): linear-residualised Pearson. z: (n,) or (n, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.asarray(z, float)
    Z = np.column_stack([np.ones(len(x)), z if z.ndim > 1 else z[:, None]])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(pearsonr(rx, ry)[0])


def bootstrap_ci(stat_fn, n: int, B: int = 1000, seed: int = 0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(B)])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ------------------------------------------------------------------ the gates
def orthogonality_gate(A, Bx, err, confounder, name_a="A", name_b="B") -> dict:
    """Two signals A, B vs binary error, controlling each other + a confounder
    (e.g. barycenter confidence). The Stage II gate, signal-name agnostic."""
    A, Bx, err = map(lambda v: np.asarray(v, float), (A, Bx, err))
    n = len(A)
    res = {
        "n": n,
        f"rho_{name_a}_{name_b}": float(pearsonr(A, Bx)[0]),
        f"rho_{name_a}_{name_b}_spearman": float(spearmanr(A, Bx)[0]),
        f"{name_a}_vs_err": float(pointbiserialr(err, A)[0]),
        f"{name_b}_vs_err": float(pointbiserialr(err, Bx)[0]),
        f"{name_b}_vs_err_given_{name_a}": partial_corr(Bx, err, A),
        f"{name_a}_vs_err_given_{name_b}": partial_corr(A, err, Bx),
        f"{name_b}_added_ci": bootstrap_ci(
            lambda i: partial_corr(Bx[i], err[i], A[i]), n),
        f"{name_a}_vs_err_given_conf": partial_corr(A, err, confounder),
        f"{name_b}_vs_err_given_conf": partial_corr(Bx, err, confounder),
    }
    return res


def redundancy_screen(signal, probe, err) -> dict:
    """Token-Barycenter / AAAI alignment battery: STC, signal-probe rho, BDD."""
    signal, probe, err = map(lambda v: np.asarray(v, float),
                             (signal, probe, err))
    n = len(signal)
    med = np.median(signal)
    return {
        "STC": float(pointbiserialr(err, (signal > med).astype(float))[0]),
        "rho_signal_probe": float(pearsonr(signal, probe)[0]),
        "signal_vs_err_given_probe": partial_corr(signal, err, probe),
        "BDD_ci": bootstrap_ci(
            lambda i: pointbiserialr(err[i], signal[i])[0], n),
    }


# ------------------------------------------------------------- smoking gun
def smoking_gun_pairs(conf, W, gt_eu, eps: float = 0.02, delta: float = 0.05,
                      max_pairs: int = 5000, seed: int = 0) -> dict:
    """Matched-confidence pairs with differing W; does W order ground-truth EU?
    conf: matched statistic (max BetP / max BetP_R); gt_eu: any GT EU proxy
    (ensemble disagreement, surprisal, hallucination label)."""
    conf, W, gt = map(lambda v: np.asarray(v, float), (conf, W, gt_eu))
    order = np.argsort(conf)
    cs, Ws, gs = conf[order], W[order], gt[order]
    rng = np.random.default_rng(seed)
    pairs, correct = 0, 0
    i = 0
    while i < len(cs) - 1 and pairs < max_pairs:
        j = i + 1
        while j < len(cs) and cs[j] - cs[i] < eps:
            if abs(Ws[i] - Ws[j]) > delta and gs[i] != gs[j]:
                pairs += 1
                hi_w = i if Ws[i] > Ws[j] else j
                hi_g = i if gs[i] > gs[j] else j
                correct += int(hi_w == hi_g)
            j += 1
        i += 1
    return {"n_pairs": pairs,
            "W_discrimination_rate": correct / pairs if pairs else None}


# ---------------------------------------------------------------- detection
def auroc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def separation_ratio(scores, labels) -> float:
    s, l = np.asarray(scores, float), np.asarray(labels)
    return float(s[l == 1].mean() / (s[l == 0].mean() + 1e-12))


def ece(probs, correct, n_bins: int = 15) -> float:
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - correct[m].mean())
    return float(e)
