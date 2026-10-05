"""Derived quantities; statements and assumptions are in reports/theory.md."""
from .conformal import conformal_rank


def beta_law_cell_moments(n_cal_cell, alpha, floor=0):
    """T1: continuous iid scores, conditional on frozen fit partition and count.

    F_j(S_(k)) ~ Beta(k,n+1-k). Infinity/floor cases are point masses at 1.
    Ties preserve conservative conformal validity, not exact Beta equality.
    """
    n = int(n_cal_cell)
    if n != n_cal_cell or n < 0 or floor < 0:
        raise ValueError("Nonnegative integer count required")
    k = conformal_rank(n, alpha)
    if n < floor or k > n:
        mean, var = 1., 0.
    else:
        mean = k / (n + 1)
        var = k * (n + 1 - k) / ((n + 1)**2 * (n + 2))
    return {"mean": mean, "variance": var, "bias_squared": (mean - (1-alpha))**2,
            "mse": var + (mean - (1-alpha))**2, "rank": k}


def beta_law_cell_var(n_cal_cell, alpha):
    """Variance in T1, including the +infinity order-statistic case."""
    return beta_law_cell_moments(n_cal_cell, alpha)["variance"]


def approx_term(M, d, coefficient=1.0, smoothness=1.0):
    """T3 surrogate A*M^(-2s/d), NOT a universal equality for cell MSCE."""
    if M <= 0 or d <= 0 or coefficient < 0 or smoothness <= 0:
        raise ValueError("Invalid approximation parameters")
    return coefficient * float(M) ** (-2 * smoothness / d)


def optimal_power_balance(n_cal, a, b, A=1.0, B=1.0):
    """T4: argmin of A*M^-a + B*M^b/n, before finite-grid/count constraints."""
    if min(n_cal, a, b, A, B) <= 0:
        raise ValueError("All balance parameters must be positive")
    return (a * A * n_cal / (b * B)) ** (1 / (a + b))


def Mstar(n_cal, d, functional, A=1.0, B=1.0, smoothness=1.0):
    """T4 only: pointwise-MSCE surrogate with explicit A and B.

    T2: cell MSCE has no quantization term. Spread and concentrated objectives
    require separate joint/extreme-value derivations; no invented exponents.
    """
    if functional != "pointwise_msce":
        raise ValueError(f"No justified interior M* derivation for {functional}")
    if d <= 0 or smoothness <= 0:
        raise ValueError("Positive dimension and smoothness required")
    return optimal_power_balance(n_cal, 2*smoothness/d, 1., A, B)
