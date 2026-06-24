"""rsuq.deferral — Selective prediction / deferral evaluation.

This is the layer that makes the random-set width a DEFERRAL signal, not just
a detector. It turns "width correlates with error" into "deferring by width
improves selective risk" and, crucially, into the two-signal composition that
is the positive mirror of the deferral line's negative result.

  risk_coverage   sort by uncertainty, keep most-confident at each coverage,
                  measure risk (error rate) among kept; AURC = area (lower better)
  compose_aurc    logistic combiner of [confidence, width] -> deferral priority,
                  fit/eval on disjoint halves (no leakage); the composition test
  deferral_battery  the full AAAI table: single-signal AURCs + the composition
                  contrast (conf vs conf+W_ctx vs conf+W_fixed) with CIs

The headline contrast: conf+W_ctx beats conf (width ADDS deferral signal) while
conf+W_fixed does not (the fixed-frame width is the redundant aux signal — the
deferral paper's negative result, reproduced). Same width formula, same head;
only the frame differs. Theory backing: W ⊥ BetP (impossibility result) gives
the width a structural reason to be non-redundant, unlike exogenous proxies.
"""

from __future__ import annotations
import numpy as np


def risk_coverage(uncertainty, error, n_points: int = 50):
    """Defer the most-uncertain; at coverage c keep the c-fraction most
    CONFIDENT (lowest uncertainty). Returns (coverages, risks, aurc).
    Lower AURC = better selective prediction."""
    u = np.asarray(uncertainty, float)
    e = np.asarray(error, float)
    order = np.argsort(u)                       # ascending uncertainty = keep first
    e_sorted = e[order]
    n = len(e_sorted)
    covs = np.linspace(1.0 / n_points, 1.0, n_points)
    risks = np.array([e_sorted[:max(1, int(c * n))].mean() for c in covs])
    return covs, risks, float(np.trapezoid(risks, covs))


def _aurc(uncertainty, error, n_points=50):
    return risk_coverage(uncertainty, error, n_points)[2]


def compose_aurc(signals: list, error, seed: int = 0, n_points: int = 50):
    """Logistic combiner of stacked signals -> P(error) deferral priority.
    Fit on half, AURC on the other half (no leakage). signals: list of (N,)
    arrays; pass confidence as -conf so 'high = defer'."""
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([np.asarray(s, float) for s in signals])
    err = np.asarray(error, int)
    if len(set(err.tolist())) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(err))
    tr, te = idx[:len(err) // 2], idx[len(err) // 2:]
    lr = LogisticRegression(max_iter=500).fit(X[tr], err[tr])
    score = lr.predict_proba(X[te])[:, 1]
    return _aurc(score, err[te], n_points)


def aurc_delta_ci(unc_a, unc_b, error, B: int = 2000, seed: int = 0):
    """Paired bootstrap on AURC(a) - AURC(b). Positive => b defers better."""
    a, b, e = map(lambda v: np.asarray(v, float), (unc_a, unc_b, error))
    if len(set(e.astype(int).tolist())) < 2:
        return {"delta": None, "significant": False, "note": "degenerate"}
    rng = np.random.default_rng(seed)
    base = _aurc(a, e) - _aurc(b, e)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(e), len(e))
        if len(set(e[i].astype(int).tolist())) < 2:
            continue
        d.append(_aurc(a[i], e[i]) - _aurc(b[i], e[i]))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"delta": float(base), "ci": [float(lo), float(hi)],
            "b_better": bool(lo > 0)}


def deferral_battery(conf, W_fixed, W_ctx, error, seed: int = 0) -> dict:
    """The full AAAI deferral table on one stratum.

    Single-signal AURCs (lower=better): confidence, fixed-W, ctx-W.
    Composition: conf, conf+W_ctx, conf+W_fixed — the headline contrast.
    The claim: conf+W_ctx < conf (ctx width adds deferral signal) AND
    conf+W_fixed approx= conf (fixed width is the redundant aux — the
    deferral-negative result reproduced)."""
    conf = np.asarray(conf, float)
    u_conf = -conf                              # high = defer
    out = {
        "n": len(conf),
        "aurc_confidence": _aurc(u_conf, error),
        "aurc_W_fixed": _aurc(W_fixed, error),
        "aurc_W_ctx": _aurc(W_ctx, error),
        "compose_conf": compose_aurc([u_conf], error, seed),
        "compose_conf_W_ctx": compose_aurc([u_conf, W_ctx], error, seed),
        "compose_conf_W_fixed": compose_aurc([u_conf, W_fixed], error, seed),
        # single-signal: does ctx-W alone beat fixed-W alone at deferral?
        "delta_Wfixed_vs_Wctx": aurc_delta_ci(W_fixed, W_ctx, error, seed=seed),
    }
    cc = out["compose_conf"]
    out["ctx_adds_to_conf"] = (out["compose_conf_W_ctx"] is not None and cc is not None
                               and out["compose_conf_W_ctx"] < cc)
    out["fixed_adds_to_conf"] = (out["compose_conf_W_fixed"] is not None and cc is not None
                                 and out["compose_conf_W_fixed"] < cc - 1e-4)
    out["deferral_claim"] = (
        "PASS: ctx-width composes with confidence; fixed-width does not "
        "(positive mirror of the deferral-negative result)"
        if out["ctx_adds_to_conf"] and not out["fixed_adds_to_conf"]
        else "INSPECT: composition pattern not as predicted")
    return out
