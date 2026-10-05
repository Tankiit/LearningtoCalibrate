"""Known conditional score CDF: S = scale(X)*U, U~Uniform[0,1]."""
import numpy as np


def scale(X):
    return 0.4 + 1.6 * np.mean(np.asarray(X), axis=1)


def sample(n, d, rng):
    X = rng.uniform(0, 1, size=(n, d))
    return X, scale(X) * rng.uniform(size=n)


def coverage_cdf(X, q):
    return np.clip(np.asarray(q) / scale(X), 0, 1)


def true_quantile(X, alpha):
    return (1 - alpha) * scale(X)
