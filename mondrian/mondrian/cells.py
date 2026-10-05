"""Pointwise assignments; ties always favor the lower exemplar index."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def distances(X, E):
    X, E = np.asarray(X, float), np.asarray(E, float)
    if X.ndim != 2 or E.ndim != 2 or X.shape[1] != E.shape[1] or len(E) == 0:
        raise ValueError("Compatible 2D matrices and nonempty exemplars required")
    if not np.isfinite(X).all() or not np.isfinite(E).all():
        raise ValueError("Nonfinite coordinates")
    return cdist(X, E, metric="sqeuclidean")


def assign_voronoi(X, E):
    return np.argmin(distances(X, E), axis=1)


def fit_power_weights(X_fit, E, n_iter=200, lr=0.1, return_diagnostics=False):
    """Fit-only empirical approximation to the semi-discrete OT dual.

    Maximize mean_i min_j(d_ij-w_j)+mean(w), supergradient 1/M-mass.
    L-BFGS minimizes the convex negative dual. Finite empirical masses may
    prevent exact balance. lr controls diminishing-step refinement.
    """
    D = distances(X_fit, E)
    n, M = D.shape
    if n == 0 or n_iter < 1 or lr <= 0:
        raise ValueError("Positive n_fit, n_iter and lr required")
    def objective(w):
        labels = np.argmin(D - w, axis=1)
        masses = np.bincount(labels, minlength=M) / n
        value = -np.mean(D[np.arange(n), labels] - w[labels]) - np.mean(w)
        return value, masses - 1 / M
    opt = minimize(objective, np.zeros(M), jac=True, method="L-BFGS-B",
                   options={"maxiter": n_iter, "ftol": 1e-13, "gtol": 1e-8, "maxls": 40})
    w = opt.x - np.mean(opt.x)
    best = w.copy()
    best_gap = np.max(np.abs(objective(w)[1]))
    for t in range(n_iter):
        _, grad = objective(w)
        w -= lr / np.sqrt(t + 1) * grad
        w -= w.mean()
        gap = np.max(np.abs(objective(w)[1]))
        if gap < best_gap:
            best, best_gap = w.copy(), gap
    best.setflags(write=False)
    masses = np.bincount(np.argmin(D - best, axis=1), minlength=M) / n
    info = {"source": "D_fit", "max_mass_error": float(best_gap), "masses": masses.tolist(),
            "target_mass": 1 / M, "optimizer_success": bool(opt.success),
            "optimizer_message": str(opt.message), "iterations": int(opt.nit),
            "balanced_within_one_observation": bool(best_gap <= 1 / n + 1e-12)}
    return (best, info) if return_diagnostics else best


def assign_power(X, E, w):
    D = distances(X, E)
    w = np.asarray(w, float)
    if w.shape != (D.shape[1],) or not np.isfinite(w).all():
        raise ValueError("One finite weight per exemplar required")
    return np.argmin(D - w, axis=1)


def assign_topk(X, E, k):
    """Ordered overlapping membership, NOT a Mondrian partition."""
    D = distances(X, E)
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= D.shape[1]:
        raise ValueError("1 <= k <= M required")
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def residual_mask(X, E, radii):
    D = distances(X, E)
    radii = np.asarray(radii, float)
    if radii.shape != (D.shape[1],) or not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("One finite nonnegative radius per exemplar required")
    return np.all(D > radii ** 2, axis=1)


def assign_with_residual(X, E, radii):
    """Nearest eligible exemplar, or residual index M; radii frozen from fit."""
    D = distances(X, E)
    outside = residual_mask(X, E, radii)
    D[D > np.asarray(radii) ** 2] = np.inf
    labels = np.argmin(D, axis=1)
    labels[outside] = len(E)
    return labels
