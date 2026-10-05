"""A fitted cell rule that sees only D_fit during fitting."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from mondrian.exemplars import select_exemplars
from mondrian.cells import assign_voronoi, assign_power, fit_power_weights
from mondrian.audit import array_hash


@dataclass(frozen=True)
class ExemplarPartition:
    exemplars: np.ndarray
    indices: np.ndarray
    weights: np.ndarray | None
    metadata: object

    @classmethod
    def fit(cls, X_fit, M, method="kmeans", seed=0, rule="voronoi"):
        idx, meta = select_exemplars(X_fit, M, method, np.random.default_rng(seed))
        E = np.array(X_fit[idx], dtype=float, copy=True)
        E.setflags(write=False)
        w = None
        if rule == "power":
            w, info = fit_power_weights(X_fit, E, return_diagnostics=True)
            meta["power_fit"] = info
        elif rule != "voronoi":
            raise ValueError(rule)
        meta.update(rule=rule, fit_hash=array_hash(X_fit), exemplar_hash=array_hash(E),
                    exemplar_fit_indices=idx.tolist(),
                    weights=None if w is None else w.tolist(),
                    weights_hash=None if w is None else array_hash(w))
        return cls(E, idx, w, MappingProxyType(meta))

    def __call__(self, X):
        return assign_voronoi(X, self.exemplars) if self.weights is None else assign_power(X, self.exemplars, self.weights)
