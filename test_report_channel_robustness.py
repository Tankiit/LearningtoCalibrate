import numpy as np

from report_channel_robustness import _normalise_rows, distribution_readouts


def _blob(pos, neg, values=(0, 50, 100)):
    return {
        "conf_values": np.asarray(values),
        "Vdist_pos": np.asarray(pos, dtype=float),
        "Vdist_neg": np.asarray(neg, dtype=float),
    }


def test_distribution_readouts_have_declared_semantics():
    blob = _blob([[0.1, 0.2, 0.7]], [[0.7, 0.2, 0.1]])
    got = distribution_readouts(blob, high_level=0.7)
    np.testing.assert_allclose(got["mean"], [0.6])
    np.testing.assert_allclose(got["argmax"], [1.0])
    np.testing.assert_allclose(got["median"], [1.0])
    np.testing.assert_allclose(got["mass_hi"], [0.6])


def test_readouts_renormalise_cached_rows_defensively():
    blob = _blob([[1, 2, 7]], [[7, 2, 1]])
    got = distribution_readouts(blob)
    np.testing.assert_allclose(got["mean"], [0.6])


def test_row_normalizer_rejects_zero_mass():
    try:
        _normalise_rows([[0, 0]])
    except ValueError as exc:
        assert "no probability mass" in str(exc)
    else:
        raise AssertionError("zero-mass row was accepted")
