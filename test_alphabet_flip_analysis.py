import numpy as np

from alphabet_flip_analysis import (
    LETTER_GRID,
    _aurc_influence,
    _pair_separation,
    aurc,
    restrict_to_letter_grid,
    side_scores,
)


def _numeric_blob(rows_pos, rows_neg):
    return {
        "conf_values": np.arange(0, 101),
        "Vdist_pos": np.asarray(rows_pos, dtype=float),
        "Vdist_neg": np.asarray(rows_neg, dtype=float),
        "V_pos": (np.asarray(rows_pos, dtype=float) @ np.arange(0, 101)) / 100.0,
        "V_neg": (np.asarray(rows_neg, dtype=float) @ np.arange(0, 101)) / 100.0,
    }


def _row(**mass):
    r = np.zeros(101)
    for value, m in mass.items():
        r[int(value)] = m
    return r


def test_letter_grid_restriction_drops_off_grid_mass_and_renormalises():
    # Half the mass at 30 (on grid), half at 37 (off grid). The restricted read
    # must be exactly 0.30, not the 0.335 the full distribution gives.
    blob = _numeric_blob([_row(**{"30": 0.5, "37": 0.5})],
                         [_row(**{"90": 0.25, "10": 0.25, "43": 0.5})])
    got = restrict_to_letter_grid(blob)
    np.testing.assert_allclose(got["V_pos"], [0.30])
    np.testing.assert_allclose(got["V_neg"], [0.50])
    np.testing.assert_allclose(got["grid_mass_pos"], [0.5])
    assert got["n_zero_grid_mass"] == 0


def test_letter_grid_restriction_reports_rows_with_no_grid_mass():
    blob = _numeric_blob([_row(**{"37": 1.0})], [_row(**{"50": 1.0})])
    got = restrict_to_letter_grid(blob)
    assert not np.isfinite(got["V_pos"][0])
    assert got["n_zero_grid_mass"] == 1


def test_letter_grid_is_the_elicitation_alphabet():
    assert list(LETTER_GRID) == list(range(0, 101, 10))
    assert len(LETTER_GRID) == 11


def test_side_scores_declare_their_directions():
    blob = {"V_pos": np.array([0.8, 0.2]), "V_neg": np.array([0.3, 0.5])}
    s = side_scores(blob)
    np.testing.assert_allclose(s["gap"], [0.5, -0.3])
    np.testing.assert_allclose(s["minus_V_neg"], -s["V_neg"])


def test_pair_separation_is_measured_at_the_gate_resolution():
    # A constant score orders nothing; a score spread one bin apart orders all.
    assert _pair_separation(np.zeros(10)) == 0.0
    assert _pair_separation(np.arange(10) * 0.05) == 1.0
    # Sub-bin jitter is not separation.
    assert _pair_separation(np.arange(10) * 1e-6) == 0.0


def test_pair_separation_reproduces_the_one_bin_degenerate_case():
    # 98.8% of items in one bin: the quoted 2.4% for llama/pavlick under letters.
    score = np.concatenate([np.zeros(988), np.full(12, 0.5)])
    assert abs(_pair_separation(score) - 0.0237) < 0.001


def test_leave_one_out_influence_matches_a_direct_recompute():
    rng = np.random.default_rng(0)
    score, risk = rng.normal(size=40), rng.integers(0, 2, 40).astype(float)
    infl = _aurc_influence(score, risk)
    keep = np.ones(40, bool)
    keep[7] = False
    assert abs(infl[7] - (aurc(score, risk) - aurc(score[keep], risk[keep]))) < 1e-12
    assert len(infl) == 40


def test_paired_flip_cancels_the_question_baseline():
    # (A_q - A_numeric) - (A_q - A_letter) == A_letter - A_numeric, exactly.
    rng = np.random.default_rng(1)
    risk = rng.integers(0, 2, 60).astype(float)
    sq, letter, numeric = (rng.normal(size=60) for _ in range(3))
    a_q, a_l, a_n = aurc(sq, risk), aurc(letter, risk), aurc(numeric, risk)
    assert abs(((a_q - a_n) - (a_q - a_l)) - (a_l - a_n)) < 1e-12
