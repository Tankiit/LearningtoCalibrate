"""Exemplar indices always refer to the supplied fit fold."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from .splits import split_hash


def select_exemplars(X_fit, M, method, rng):
    X = np.asarray(X_fit, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or not 1 <= M <= len(X):
        raise ValueError("Finite matrix and 1 <= M <= n_fit required")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    state = copy.deepcopy(rng.bit_generator.state)
    selection_seed = int(rng.integers(0, 2**31 - 1))
    if method == "uniform":
        idx = rng.choice(len(X), M, replace=False)
    elif method == "kmeans":
        centers = KMeans(n_clusters=M, random_state=selection_seed, n_init=10).fit(X).cluster_centers_
        # One-to-one minimum-cost snapping avoids duplicate exemplar indices.
        _, idx = linear_sum_assignment(cdist(centers, X, metric="sqeuclidean"))
    elif method == "voronoi":
        # Provisional rule: Gonzalez farthest-first traversal, seeded start.
        # The archive did not supply its referenced existing-pipeline rule.
        chosen = [int(rng.integers(len(X)))]
        distance = np.full(len(X), np.inf)
        for _ in range(1, M):
            distance = np.minimum(distance, np.sum((X - X[chosen[-1]]) ** 2, axis=1))
            distance[chosen] = -np.inf
            chosen.append(int(np.argmax(distance)))
        idx = np.asarray(chosen)
    else:
        raise ValueError(f"Unknown selection method: {method}")
    idx = np.asarray(idx, dtype=np.int64)
    idx.setflags(write=False)
    return idx, {"M": int(M), "method": method, "selection_seed": selection_seed,
                 "rng_state_before": state, "index_hash": split_hash(idx), "source": "D_fit"}
