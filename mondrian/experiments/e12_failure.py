"""E12 (reports/estimand_note_E9_popqa_2026-10-03.md#e12): failure-mode suite for transport, cells,
multicalibration and direct predictors under controlled and real elicitation shifts.

  python -m experiments.e12_failure --data {tqa,popqa} [--seeds N]
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from mondrian.conformal import per_cell_quantile
from mondrian.calibration import metrics as calib
from .e_llm import label_scores, FLOOR
from .e5_partition import load_tqa, onehot
from .e5v2 import tuned, multicalibrate
from .e8_transport import map_quantile, map_paired
from .e2_frozen import MODELS

ALPHA, T = 0.1, 0.9
LEVELS = {"level_a1_b1": (1, 1), "level_a0.5_b0": (.5, 0), "level_a2_b-1": (2, -1), "reverse_a-1_b0": (-1, 0)}
RHOS = (0.1, 0.25, 0.5, 1.0)


def lg(v):
    v = np.clip(v, 1e-6, 1 - 1e-6); return np.log(v / (1 - v))


def sg(z):
    return 1 / (1 + np.exp(-z))


def shifts(VA, Valt, rng):
    out = {"none": VA.copy(), "reword": Valt.copy()}
    for k, (a, b) in LEVELS.items():
        out[k] = sg(a * lg(VA) + b)
    for rho in RHOS:
        V = VA.copy(); idx = rng.choice(len(V), int(round(rho * len(V))), replace=False)
        V[idx] = V[rng.permutation(idx)]; out[f"reorder_{rho:g}"] = V
    return out


def conformal(p_cal, y_cal, g_cal, p_te, y_te, g_te, M):
    q, counts = per_cell_quantile(g_cal, label_scores(p_cal, y_cal), M, ALPHA, FLOOR)
    thr = q[g_te]
    s1, s0 = label_scores(p_te, 1) <= thr, label_scores(p_te, 0) <= thr
    hit = np.where(y_te == 1, s1, s0)
    return hit, s1.astype(int) + s0, int(np.sum(counts < FLOOR)), (s1, s0)


def sgap(hit, g, M):
    """Signed per-cell gap on the representation groups (transport_controls_note)."""
    cov = [hit[g == j].mean() - T for j in range(M) if np.any(g == j)]
    return {"gap_rep_min": float(min(cov)), "gap_rep_max": float(max(cov))}


def gap(hit, g, M):
    cov = [hit[g == j].mean() for j in range(M) if np.any(g == j)]
    return float(max(abs(c - T) for c in cov))


def run_split(H, y, VA, Valt, seed):
    n = len(y); rng = np.random.default_rng(np.random.SeedSequence([seed, 1212]))
    perm = rng.permutation(n); mf, fi, ca, te = np.array_split(perm, 4)
    assert len(set(mf) | set(fi) | set(ca) | set(te)) == n
    Z = PCA(16, random_state=0).fit(H[mf]).transform(H)
    rep = ExemplarPartition.fit(Z[mf], 4, "voronoi", seed)(Z)
    zA = lg(VA)[:, None]
    # forecasters fitted under A
    R = tuned(zA[fi], y[fi].astype(float), seed)[0]
    XG = lambda z: np.hstack([z, onehot(rep, 4)])
    G = tuned(XG(zA)[fi], y[fi].astype(float), seed)[0]
    DS = tuned(Z[fi], y[fi].astype(float), seed)[0]
    XD = lambda z: np.hstack([z, Z])
    DR = tuned(XD(zA)[fi], y[fi].astype(float), seed)[0]
    bin_cuts = np.quantile(VA[fi], [.25, .5, .75]); binf = lambda v: np.searchsorted(bin_cuts, v)
    qA = np.quantile(VA[fi], [.25, .5, .75])
    res = {}
    for name, VB in shifts(VA, Valt, rng).items():
        Tq = map_quantile(VB[mf], VA[mf]); Tp, _ = map_paired(VB[mf], VA[mf])
        variants = {"raw": VB, "quantile": Tq(VB), "paired": Tp(VB)}
        out = {}
        for tv, Vt in variants.items():
            zB = lg(Vt)[:, None]
            # bins (cells move with the report)
            hA, _, _, setsA = conformal(R.predict(zA[ca]), y[ca], binf(VA[ca]), R.predict(zA[te]), y[te], binf(VA[te]), 4)
            pR = R.predict(zB[te])
            hit, size, below, sets = conformal(R.predict(zA[ca]), y[ca], binf(VA[ca]), pR, y[te], binf(Vt[te]), 4)
            occA = np.bincount(binf(VA[te]), minlength=4) / len(te); occB = np.bincount(binf(Vt[te]), minlength=4) / len(te)
            cross = float(np.mean((sets[0] != setsA[0]) | (sets[1] != setsA[1])))
            out[f"bins|{tv}"] = dict(coverage=float(hit.mean()), size=float(size.mean()), gap_own=gap(hit, binf(Vt[te]), 4),
                                     gap_rep=gap(hit, rep[te], 4), **sgap(hit, rep[te], 4), occupancy_tv=float(.5 * np.abs(occA - occB).sum()),
                                     below_floor=below, crossing=cross, **calib(pR, y[te]))
            if tv == "paired":
                continue
            # representation cells (do not move), forecaster G
            pG = G.predict(XG(zB)[te])
            hit, size, below, _ = conformal(G.predict(XG(zA)[ca]), y[ca], rep[ca], pG, y[te], rep[te], 4)
            out[f"repcells|{tv}"] = dict(coverage=float(hit.mean()), size=float(size.mean()), gap_own=gap(hit, rep[te], 4),
                                         gap_rep=gap(hit, rep[te], 4), **sgap(hit, rep[te], 4), occupancy_tv=0.0, below_floor=below, **calib(pG, y[te]))
            # multicalibration (holdout within fit fold), one-cell threshold
            gl = [rep[fi], np.searchsorted(qA, VA[fi])]
            pc = multicalibrate(lambda X, yy: tuned(X, yy, seed), zA[fi], y[fi].astype(float), gl, zA[ca], [rep[ca], np.searchsorted(qA, VA[ca])], seed)
            pt = multicalibrate(lambda X, yy: tuned(X, yy, seed), zA[fi], y[fi].astype(float), gl, zB[te], [rep[te], np.searchsorted(qA, Vt[te])], seed)
            hit, size, below, _ = conformal(pc, y[ca], np.zeros(len(ca), int), pt, y[te], np.zeros(len(te), int), 1)
            out[f"mc|{tv}"] = dict(coverage=float(hit.mean()), size=float(size.mean()), gap_own=gap(hit, np.zeros(len(te), int), 1),
                                   gap_rep=gap(hit, rep[te], 4), **sgap(hit, rep[te], 4), below_floor=below, **calib(pt, y[te]))
            # direct predictor with report
            pD = DR.predict(XD(zB)[te])
            hit, size, below, _ = conformal(DR.predict(XD(zA)[ca]), y[ca], np.zeros(len(ca), int), pD, y[te], np.zeros(len(te), int), 1)
            out[f"direct_state_report|{tv}"] = dict(coverage=float(hit.mean()), size=float(size.mean()), gap_rep=gap(hit, rep[te], 4), **sgap(hit, rep[te], 4),
                                                    below_floor=below, **calib(pD, y[te]))
        # state-only direct predictor: invariant to B
        pS = DS.predict(Z[te])
        hit, size, below, _ = conformal(DS.predict(Z[ca]), y[ca], np.zeros(len(ca), int), pS, y[te], np.zeros(len(te), int), 1)
        out["direct_state|raw"] = dict(coverage=float(hit.mean()), size=float(size.mean()), gap_rep=gap(hit, rep[te], 4), **sgap(hit, rep[te], 4), below_floor=below,
                                       **calib(pS, y[te]))
        res[name] = out
    return res


def mechanics(V):
    return {p: {"tied_fraction": float(1 - len(np.unique(np.round(v, 6))) / len(v)),
                "tail_fraction": float(np.mean((v < .02) | (v > .98)))} for p, v in V.items()}


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--data", choices=["tqa", "popqa"], default="tqa")
    pa.add_argument("--seeds", type=int, default=100)
    pa.add_argument("--out", default="runs/p1_e12")
    a = pa.parse_args()
    loader = load_tqa
    if a.data == "popqa":
        from .e9_popqa import load_popqa_tqa_style as loader
    from joblib import Parallel, delayed
    result = {"criterion": "reports/estimand_note_E9_popqa_2026-10-03.md#e12", "data": a.data}
    for model in MODELS:
        H, y, V = loader(model)
        per = Parallel(n_jobs=-2)(delayed(run_split)(H, y, V["fwd"], V["alt"], s) for s in range(a.seeds))
        agg = {}
        for sh in per[0]:
            agg[sh] = {}
            for arm in per[0][sh]:
                agg[sh][arm] = {m: float(np.mean([r[sh][arm][m] for r in per])) for m in per[0][sh][arm]}
                agg[sh][arm]["coverage_q10_q90"] = np.quantile([r[sh][arm]["coverage"] for r in per], [.1, .9]).tolist()
        none = agg["none"]
        chk = {"none_transport_moves_lt_0.005": all(abs(none[f"{k}|quantile"]["coverage"] - none[f"{k}|raw"]["coverage"]) < .005
                                                    for k in ("bins", "repcells", "mc", "direct_state_report")),
               "level_positive_control_within_0.01": all(abs(agg[k]["bins|quantile"]["coverage"] - none["bins|raw"]["coverage"]) < .01
                                                         for k in LEVELS if not k.startswith("reverse"))}
        result[model] = {"aggregate": agg, "mechanics": mechanics(V), "checks": chk}
        print(model, chk, flush=True)
    result["source_sha256"] = source_hash()
    write_json(Path(a.out) / f"e12_{a.data}.json", result)


if __name__ == "__main__":
    main()
