"""
beta as a proper scoring rule comparison, and alpha(beta) as a
PSR-motivated steering step size.

beta is a cluster-level log score differential:

    beta(C*, x) = S_log(C*, P_instruct) - S_log(C*, P_base)
                = log P_instruct(C*) - log P_base(C*)

where C* is the dominant cluster under the base model. Negative beta means
the instruct model assigns lower probability to the base model's preferred
cluster, which is the intended uncertainty/disagreement signal.

The beta-gated steering schedule uses:

    alpha(beta) = alpha_max * sigmoid(tau - beta)

This sign is intentional. It makes alpha approach alpha_max when beta is far
below the threshold tau, and decay toward zero when beta is high. The
equivalent written form alpha_max * sigmoid(beta - tau) would steer hardest
when the models agree, which is the opposite of the intended circuit breaker.

Reference: Gneiting & Raftery (2007) for proper scoring rules.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit as sigmoid


def log_score(outcome_logprob: float) -> float:
    """Log score S_log(outcome, forecaster) = log P_forecaster(outcome)."""
    return float(outcome_logprob)


def log_score_cluster(member_logprobs: list[float] | np.ndarray) -> float:
    """
    Log score for a cluster outcome.

    The forecaster probability of a cluster is the sum of the probabilities of
    its members, computed with log-sum-exp for numerical stability.
    """
    arr = np.asarray(member_logprobs, dtype=float)
    if arr.size == 0:
        raise ValueError("member_logprobs must be non-empty")
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def compute_beta_psr(
    dominant_cluster_logprobs_base: list[float] | np.ndarray,
    dominant_cluster_logprobs_inst: list[float] | np.ndarray,
) -> float:
    """
    Compute beta as a log score differential on the same cluster outcome.

    beta < 0: instruct assigns less probability to C* than base does.
    beta > 0: instruct assigns more probability to C* than base does.
    """
    log_p_base = log_score_cluster(dominant_cluster_logprobs_base)
    log_p_inst = log_score_cluster(dominant_cluster_logprobs_inst)
    return float(log_p_inst - log_p_base)


def compute_alpha(beta: float, tau: float, alpha_max: float) -> float:
    """
    PSR-motivated beta-gated steering step size.

    Uses alpha(beta) = alpha_max * sigmoid(tau - beta), so steering grows when
    beta falls below the threshold tau and decays when models agree.

    Behaviour:
        beta << tau -> alpha ~= alpha_max
        beta == tau -> alpha = alpha_max / 2
        beta >> tau -> alpha ~= 0
    """
    if alpha_max < 0:
        raise ValueError("alpha_max must be non-negative")
    return float(alpha_max * sigmoid(tau - beta))


def steering_fires(beta: float, tau: float) -> bool:
    """Circuit breaker: steering fires when beta is below threshold tau."""
    return bool(beta < tau)


def alpha_from_contraction_bound(
    lambda_max: float,
    safety_factor: float = 0.9,
) -> float:
    """
    Set alpha_max from the contraction theorem bound alpha < 2 / lambda_max.
    """
    if lambda_max <= 0:
        raise ValueError("lambda_max must be positive")
    if not 0 < safety_factor < 1:
        raise ValueError("safety_factor must be in (0, 1)")
    return float(safety_factor * 2.0 / lambda_max)


def tau_sweep(
    betas: np.ndarray,
    hallucination_labels: np.ndarray,
    tau_range: tuple[float, float] = (-2.0, 0.5),
    n_steps: int = 50,
) -> dict:
    """
    Sweep tau values for hallucination detection.

    hallucination_labels should be 1 for hallucination and 0 otherwise.
    Returns threshold-free AUROC for -beta plus best threshold metrics.
    """
    from sklearn.metrics import roc_auc_score

    betas = np.asarray(betas, dtype=float)
    labels = np.asarray(hallucination_labels, dtype=int)
    valid = ~np.isnan(betas)
    b = betas[valid]
    y = labels[valid]

    if len(np.unique(y)) < 2:
        return {"error": "only one class in hallucination_labels"}

    auroc = float(roc_auc_score(y, -b))
    results = []
    for tau in np.linspace(tau_range[0], tau_range[1], n_steps):
        pred = (b < tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append(
            {
                "tau": float(tau),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "n_flagged": int(pred.sum()),
                "flag_rate": float(pred.mean()),
            }
        )

    best = max(results, key=lambda row: row["f1"])
    return {
        "auroc": auroc,
        "best_tau": best["tau"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "tau_sweep": results,
    }


if __name__ == "__main__":
    lp_base = [-1.0, -2.0, -3.0]
    lp_inst = [-1.0, -2.0, -3.0]
    beta = compute_beta_psr(lp_base, lp_inst)
    assert abs(beta) < 1e-8

    beta2 = compute_beta_psr([-1.0, -2.0], [-3.0, -4.0])
    assert beta2 < 0

    tau, alpha_max = -0.5, 0.1
    a_low = compute_alpha(-3.0, tau, alpha_max)
    a_mid = compute_alpha(tau, tau, alpha_max)
    a_high = compute_alpha(2.0, tau, alpha_max)
    assert a_low > 0.09
    assert abs(a_mid - alpha_max / 2) < 1e-6
    assert a_high < 0.01

    assert steering_fires(-1.0, -0.5)
    assert not steering_fires(0.0, -0.5)

    stable = log_score_cluster([-1000.0, -1001.0, -1002.0])
    assert not np.isnan(stable)
    print("All beta_psr checks passed.")
