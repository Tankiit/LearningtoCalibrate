"""E10 (reports/estimand_note_E9_popqa_2026-10-03.md): controlled procedure changes on real reports.
Synthetic B = s*logit(V_A) + b + sigma*sd*eps; tests Proposition transport with known ground truth."""
import json
from pathlib import Path
import numpy as np
from mondrian.audit import write_json, source_hash
from mondrian.conformal import per_cell_quantile
from .e_llm import label_scores, FLOOR
from .e8_transport import platt, map_quantile, map_paired, evaluate, map_quantile_mid, map_iso
from .e2_frozen import MODELS, load as load_frozen
from data.tinker import load_tinker, DEFAULT as TINKER

ALPHA = 0.1
SIGMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def lg(v):
    v = np.clip(v, 1e-6, 1 - 1e-6); return np.log(v / (1 - v))


def make_B(vA, s, b, sigma, rng):
    z = lg(vA); sd = z.std()
    return 1 / (1 + np.exp(-(s * z + b + sigma * sd * rng.normal(size=len(z)))))


def oracle(vA, s, b):
    """True noiseless inverse of the synthetic change: logit v -> (logit v - b) / s (transport_controls_note)."""
    return lambda v: 1 / (1 + np.exp(-(lg(v) - b) / s))


def one(vA, vB, y, f, c, te, s=1, b=0.0):
    c1, c2 = c[:len(c) // 2], c[len(c) // 2:]   # honest split: Platt on c1, threshold on c2 (amendment 4 Oct)
    RA = platt(vA[c1], y[c1])
    q, _ = per_cell_quantile(np.zeros(len(c2), int), label_scores(RA(vA[c2]), y[c2]), 1, ALPHA, FLOOR)
    qt = np.full(len(te), q[0]); pA = RA(vA[te])
    Tq = map_quantile(vB[f], vA[f]); Tp, inc = map_paired(vB[f], vA[f]); Tm = map_quantile_mid(vB[f], vA[f])
    To = oracle(vA, s, b)
    out = {"A->A": evaluate(pA, pA, qt, y[te])}
    for name, p in (("naive", RA(vB[te])), ("quantile", RA(Tq(vB[te]))), ("paired_iso", RA(Tp(vB[te]))),
                    ("quantile_mid", RA(Tm(vB[te]))), ("paired_iso_inc", RA(map_iso(vB[f], vA[f], True)(vB[te]))),
                    ("paired_iso_dec", RA(map_iso(vB[f], vA[f], False)(vB[te]))), ("oracle_inverse", RA(To(vB[te])))):
        e = evaluate(pA, p, qt, y[te]); e["abs_dcov"] = abs(e["coverage"] - out["A->A"]["coverage"])
        e["bound_holds"] = bool(e["abs_dcov"] <= e["crossing_mass"] + 1e-12); out[name] = e
    out["paired_direction_increasing"] = bool(inc)
    return out


def summarise(per):
    keys = [k for k in per[0] if k != "paired_direction_increasing"]
    res = {k: {m: float(np.mean([r[k][m] for r in per])) for m in per[0][k]} for k in keys}
    for k in keys:
        if "abs_dcov" in per[0][k]:
            res[k]["bound_holds_rate"] = float(np.mean([r[k]["bound_holds"] for r in per]))
    res["paired_increasing_rate"] = float(np.mean([r["paired_direction_increasing"] for r in per]))
    return res


def main(out="runs/p1_e10"):
    res = {"criterion": "reports/estimand_note_E9_popqa_2026-10-03.md#e10", "source_sha256": source_hash()}
    grid = [(s, b, sg) for s in (1, -1) for b in (0.0, 1.0) for sg in SIGMAS]
    for model in MODELS:
        ids, y, V = load_frozen(model); vA = V["fwd"]; n = len(y); res[model] = {}
        for s, b, sg in grid:
            per = []
            for seed in range(100):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 1010, s + 2, int(b), int(sg * 100)]))
                vB = make_B(vA, s, b, sg, rng)
                f, c, te = np.array_split(np.random.default_rng(np.random.SeedSequence([seed, 1011])).permutation(n), 3)
                per.append(one(vA, vB, y, f, c, te, s, b))
            res[model][f"s{s}_b{b:g}_sig{sg:g}"] = summarise(per)
        print("done", model, flush=True)
    d = load_tinker(TINKER, cache="inputs/tinker_triviaqa/aligned.npz")
    sp, y = d["split"], d["correct"].astype(int); vA = d["p_qa_trained_normal"]
    f, c, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "cal"), np.flatnonzero(sp == "test")
    res["tinker_qa"] = {}
    for s, b, sg in grid:
        per = []
        for seed in range(20):   # resample the synthetic noise; split fixed
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1012, s + 2, int(b), int(sg * 100)]))
            per.append(one(vA, make_B(vA, s, b, sg, rng), y, f, c, te, s, b))
        res["tinker_qa"][f"s{s}_b{b:g}_sig{sg:g}"] = summarise(per)
    print("done tinker", flush=True)
    # predictions
    checks = {}
    for key in [m for m in MODELS] + ["tinker_qa"]:
        R = res[key]
        p_exact = all(R[f"s{s}_b{b:g}_sig0"]["paired_iso"]["abs_dcov"] < 0.01 for s in (1, -1) for b in (0.0, 1.0))
        q_exact_inc = all(R[f"s1_b{b:g}_sig0"]["quantile"]["abs_dcov"] < 0.01 for b in (0.0, 1.0))
        q_fail_dec = all(R[f"s-1_b{b:g}_sig0"]["quantile"]["abs_dcov"] >= 0.01 for b in (0.0, 1.0))
        cm0 = all(R[f"s{s}_b{b:g}_sig0"]["paired_iso"]["crossing_mass"] < 0.02 for s in (1, -1) for b in (0.0, 1.0))
        grows = all(R[f"s{s}_b{b:g}_sig2"]["paired_iso"]["crossing_mass"] > R[f"s{s}_b{b:g}_sig0"]["paired_iso"]["crossing_mass"]
                    for s in (1, -1) for b in (0.0, 1.0))
        bound = all(v.get("bound_holds_rate", 1.0) == 1.0 for g in R.values() if isinstance(g, dict) for v in g.values() if isinstance(v, dict))
        checks[key] = dict(paired_exact_sigma0=p_exact, quantile_exact_increasing=q_exact_inc, quantile_fails_decreasing=q_fail_dec,
                           crossing_small_sigma0=cm0, crossing_grows=grows, bound_always=bound)
    res["checks"] = checks
    # descriptive controls (transport_controls_note); not part of the status
    res["controls"] = {key: {"oracle_max_abs_dcov_sigma0": max(res[key][f"s{s}_b{b:g}_sig0"]["oracle_inverse"]["abs_dcov"]
                                                             for s in (1, -1) for b in (0.0, 1.0))}
                       for key in [m for m in MODELS] + ["tinker_qa"]}
    res["status"] = "PASS" if all(all(v.values()) for v in checks.values()) else "DESCRIPTIVE"
    write_json(Path(out) / "e10_result.json", res)
    print(res["status"], json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
