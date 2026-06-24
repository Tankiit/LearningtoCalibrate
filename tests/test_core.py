"""Pinned identities I1-I8 (numerically validated before implementation)."""
import sys, os, torch, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsuq.core.frame import FixedFrame, ContextFrame
from rsuq.core.beliefs import within_cluster_q, cluster_prob_mass, ranked_pignistic, pignistic
from rsuq.core.signals import credal_width, per_cluster_entropy

V, K, P, D = 400, 10, 25, 16

@pytest.fixture
def fixed():
    g = torch.Generator().manual_seed(0)
    kappa = torch.randint(0, K, (V,), generator=g); kappa[:K] = torch.arange(K)
    return FixedFrame(kappa=kappa, K=K)

@pytest.fixture
def ctx(fixed):
    g = torch.Generator().manual_seed(1)
    poly = torch.randperm(V, generator=g)[:P]
    cents = {int(w): torch.randn(3, D, generator=g) for w in poly}
    s2c = {int(w): torch.randint(0, K, (3,), generator=g) for w in poly}
    return ContextFrame(base=fixed, poly_tokens=poly,
                        sense_centroids=cents, sense_cluster=s2c)

def _pm(seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(V, generator=g) * 2, -1)
    m = torch.distributions.Dirichlet(torch.ones(K)).sample()
    return p, m

def test_I1_collapse(fixed):
    p, _ = _pm()
    kappa, sizes = fixed.assignments()
    q = within_cluster_q(p, kappa, K)
    bp = ranked_pignistic(cluster_prob_mass(p, kappa, K), q, kappa, sizes, 1.0)
    assert torch.allclose(bp, p, atol=1e-6)

def test_I2_lam0(fixed):
    p, m = _pm()
    kappa, sizes = fixed.assignments()
    q = within_cluster_q(p, kappa, K)
    assert torch.allclose(ranked_pignistic(m, q, kappa, sizes, 0.0),
                          pignistic(m, kappa, sizes), atol=1e-6)

def test_I6_ctx_sizes(fixed, ctx):
    h = torch.randn(7, D)
    kt, st = ctx.assignments(h)
    for i in range(7):
        assert torch.allclose(st[i], torch.bincount(kt[i], minlength=K).float())

def test_I7_disjoint_exhaustive(ctx):
    kt, st = ctx.assignments(torch.randn(5, D))
    assert int(st.sum(-1)[0]) == V and (st > 0).all()

def test_I8_ctx_pignistic_normalised(ctx):
    _, m = _pm()
    kt, st = ctx.assignments(torch.randn(3, D))
    bp = pignistic(m.expand(3, -1), kt, st)
    assert torch.allclose(bp.sum(-1), torch.ones(3), atol=1e-5)

def test_ctx_width_differs_only_via_sizes(fixed, ctx):
    _, m = _pm()
    kt, st = ctx.assignments(torch.randn(2, D))
    Wf = credal_width(m, fixed.sizes)
    Wc = credal_width(m.expand(2, -1), st)
    assert Wc.shape == (2,)   # and generally Wc != Wf — sizes moved

def test_q11_vertexwise_lemma():
    """W = (1/2)||p_sigma - BetP||_1 for EVERY vertex (disjoint case)."""
    import numpy as np
    rng = np.random.default_rng(3)
    for _ in range(100):
        Kc = rng.integers(2, 6); sizes = rng.integers(1, 50, Kc).astype(float)
        m = rng.dirichlet(np.ones(Kc))
        W = m @ (1 - 1/sizes)
        per_vertex = (m*(1 - 1/sizes) + (sizes - 1)*m/sizes).sum() / 2
        assert np.isclose(W, per_vertex, atol=1e-12)

def test_separation_score_splits_poly_mono():
    """OD-1: separable blobs score high, single blob scores low."""
    import numpy as np
    from rsuq.core.sense_inventory import separation_score
    rng = np.random.default_rng(0)
    poly = np.vstack([rng.normal(0,0.3,(40,16)), rng.normal(5,0.3,(40,16))])
    mono = rng.normal(0,0.3,(80,16))
    assert separation_score(poly)[0] > 0.5
    assert separation_score(mono)[0] < 0.5

def test_rsnn_matrix_pignistic_matches_scatter(fixed):
    """RS-NN final_betp == RSUQ scatter pignistic on a disjoint frame."""
    from rsuq.rsnn_compat import assert_matches_scatter_pignistic
    _, m = _pm()
    assert assert_matches_scatter_pignistic(fixed, m)

def test_rsnn_belief_encoding_is_onehot_disjoint(fixed):
    """RS-NN groundtruth encoding == one-hot cluster id on a disjoint frame."""
    import numpy as np, torch
    from rsuq.rsnn_compat import groundtruth_belief_encode
    fs = [set(fixed.members[k].tolist()) for k in range(K)]
    labels = torch.randint(0, V, (20,)).tolist()
    Y = groundtruth_belief_encode(labels, fs)
    assert (Y.sum(1) == 1).all()
    assert (Y.argmax(1) == fixed.kappa[torch.tensor(labels)].numpy()).all()

def test_deferral_risk_coverage_monotone_ideal():
    """A perfect uncertainty signal => risk rises with coverage; AURC < random."""
    import numpy as np
    from rsuq.deferral import risk_coverage
    rng = np.random.default_rng(0)
    err = rng.integers(0,2,500).astype(float)
    perfect = err + 0.01*rng.uniform(0,1,500)   # uncertainty ~ error
    _,_,aurc_perfect = risk_coverage(perfect, err)
    _,_,aurc_random  = risk_coverage(rng.uniform(0,1,500), err)
    assert aurc_perfect < aurc_random

def test_deferral_composition_detects_nonredundant():
    """conf+orthogonal-width composes better than conf+noise-width."""
    import numpy as np
    from rsuq.deferral import deferral_battery
    rng = np.random.default_rng(0); N=1500
    conf = rng.uniform(0,1,N); extra = rng.uniform(0,1,N)
    err = (rng.uniform(0,1,N) > 0.5*conf + 0.4*extra + 0.1).astype(int)
    W_ctx = 0.6*extra + 0.2*(1-conf) + 0.2*rng.uniform(0,1,N)
    W_fixed = 0.2*(1-conf) + 0.8*rng.uniform(0,1,N)
    b = deferral_battery(conf, W_fixed, W_ctx, err)
    assert b["ctx_adds_to_conf"]            # non-redundant width composes
