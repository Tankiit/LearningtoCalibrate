import numpy as np
from mondrian.calibration import metrics


def test_sce_is_binary_ece():
    rng = np.random.default_rng(1); p = rng.uniform(size=3000); y = (rng.uniform(size=3000) < p ** 1.5).astype(int)
    e = np.linspace(0, 1, 16); b = np.clip(np.digitize(p, e[1:-1]), 0, 14)
    hand = sum(abs(y[b == k].mean() - p[b == k].mean()) * np.mean(b == k) for k in range(15) if np.any(b == k))
    assert abs(metrics(p, y)["sce"] - hand) < 1e-12


def test_perfect_forecast_small_error():
    rng = np.random.default_rng(2); p = rng.uniform(size=200000); y = (rng.uniform(size=200000) < p).astype(int)
    m = metrics(p, y)
    assert m["sce"] < 0.01 and m["ece"] < 0.01
