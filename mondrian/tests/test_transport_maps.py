import numpy as np
from experiments.e8_transport import map_quantile_mid, map_iso


def test_midrank_map_batch_independent_and_exact_for_increasing_change():
    rng = np.random.default_rng(0); vA = rng.uniform(.05, .95, 4000); vB = vA ** 2   # increasing, continuous
    T = map_quantile_mid(vB, vA)
    probe = vB[:50]
    assert np.array_equal(T(probe), np.concatenate([T(probe[:25]), T(probe[25:])]))   # no batch dependence
    assert np.max(np.abs(T(vB) - vA)) < 2e-3


def test_midrank_map_handles_atoms():
    vA = np.linspace(.1, .9, 1000); vB = np.round(vA, 1)   # heavy ties
    out = map_quantile_mid(vB, vA)(vB)
    assert np.all(np.isfinite(out)) and len(np.unique(out)) <= len(np.unique(vB))


def test_forced_direction_isotonic():
    rng = np.random.default_rng(1); vA = rng.uniform(.05, .95, 2000); vB = 1 - vA
    assert np.max(np.abs(map_iso(vB, vA, False)(vB) - vA)) < 1e-6
    assert np.std(map_iso(vB, vA, True)(vB)) < 1e-6   # an increasing fit to a reversal is constant
