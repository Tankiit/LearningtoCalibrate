"""E8 (reports/estimand_note_E8_2026-10-02.md): label-free transport of a calibration across
elicitation procedures, using unpaired (quantile) and paired (isotonic) monotone maps."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from mondrian.conformal import per_cell_quantile
from mondrian.calibration import metrics as calib
from .e_llm import logit, label_scores, FLOOR
from .e2_frozen import MODELS
from .e5_partition import load_tqa
from data.tinker import load_tinker, DEFAULT as TINKER

ALPHA, M = 0.1, 4


def platt(v, y):
    m = LogisticRegression(C=1e4, max_iter=2000).fit(logit(v), y)
    return lambda u: np.clip(m.predict_proba(logit(u))[:, 1], 1e-4, 1 - 1e-4)


def map_quantile(vB_fit, vA_fit):
    sB, sA = np.sort(vB_fit), np.sort(vA_fit)
    return lambda v: np.quantile(sA, np.clip(np.searchsorted(sB, v, side="right") / len(sB), 0, 1))


def map_quantile_mid(vB_fit, vA_fit):
    """Mid-rank quantile map, both CDFs from the transport fold and frozen (transport_controls_note)."""
    sB, sA = np.sort(vB_fit), np.sort(vA_fit); n = len(sB)
    def T(v):
        v = np.asarray(v, float)
        F = (np.searchsorted(sB, v, side="left") + .5 * (np.searchsorted(sB, v, side="right") - np.searchsorted(sB, v, side="left"))) / n
        return np.quantile(sA, np.clip(F, .5 / n, 1 - .5 / n), method="inverted_cdf")
    return T


def map_iso(vB_fit, vA_fit, increasing):
    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(vB_fit, vA_fit)
    return lambda v: np.clip(iso.predict(v), 1e-6, 1 - 1e-6)


def tie_fraction(v):
    return float(1 - len(np.unique(np.round(v, 6))) / len(v))


def map_paired(vB_fit, vA_fit):
    best = None
    for inc in (True, False):
        iso = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(vB_fit, vA_fit)
        mse = float(np.mean((iso.predict(vB_fit) - vA_fit) ** 2))
        if best is None or mse < best[0]:
            best = (mse, iso, inc)
    return (lambda v: np.clip(best[1].predict(v), 1e-6, 1 - 1e-6)), best[2]


def map_paired_group(vB_fit, vA_fit, g_fit, g_all_te):
    maps = {}
    for g in np.unique(g_fit):
        m = g_fit == g
        maps[g] = map_paired(vB_fit[m], vA_fit[m])[0] if m.sum() >= 10 else map_paired(vB_fit, vA_fit)[0]
    def T(v, g):
        out = np.empty_like(v)
        for gg in np.unique(g):
            sel = g == gg
            out[sel] = maps.get(gg, map_paired(vB_fit, vA_fit)[0])(v[sel])
        return out
    return T


def sets(p, q_cells):
    s1 = label_scores(p, 1) <= q_cells; s0 = label_scores(p, 0) <= q_cells
    return s1, s0


def evaluate(pA_te, pT_te, q_cells, y, cal=True, cells=None):
    s1, s0 = sets(pT_te, q_cells); r1, r0 = sets(pA_te, q_cells)
    hit = np.where(y == 1, s1, s0)
    g = np.zeros(len(y), int) if cells is None else np.asarray(cells)
    gaps = [hit[g == j].mean() - (1 - ALPHA) for j in np.unique(g)]   # signed per-cell gap
    return (calib(pT_te, y) if cal else {}) | {"cell_gap_min": float(min(gaps)), "cell_gap_max": float(max(gaps)),"coverage": float(hit.mean()), "set_size": float((s1.astype(int) + s0).mean()),
            "brier": float(np.mean((pT_te - y) ** 2)),
            "crossing_mass": float(np.mean((s1 != r1) | (s0 != r0))),
            "label_loss_mass": float(np.mean((r1 & ~s1) | (r0 & ~s0)))}   # P(C_A not subset of C_T), Prop. transport (a)


def recal_reference(VB, y, lab, te, pA, qt, cells_te=None):
    """Labelled reference: Platt under B on half of k labels, one-cell threshold on the other half.
    Scored per frozen group when cells_te is given (amendment 4 Oct)."""
    h = len(lab) // 2; a, b = lab[:h], lab[h:]
    RB = platt(VB[a], y[a]) if len(np.unique(y[a])) > 1 else (lambda u: np.full(len(u), y[a].mean()))
    q, _ = per_cell_quantile(np.zeros(len(b), int), label_scores(RB(VB[b]), y[b]), 1, ALPHA, min(FLOOR, len(b)))
    return evaluate(pA, RB(VB[te]), np.full(len(te), q[0]), y[te], cells=cells_te)


def frozen(seeds=range(500)):
    out = {}
    for model in MODELS:
        H, y, V = load_tqa(model); n = len(y); res = {}
        for B in ("rev", "alt"):
            per = {}
            for seed in seeds:
                perm = np.random.default_rng(np.random.SeedSequence([seed, 8080])).permutation(n)
                f, c, te = np.array_split(perm, 3)
                Z = PCA(16, random_state=0).fit(H[f]).transform(H)
                cells = ExemplarPartition.fit(Z[f], M, "voronoi", seed)(Z)
                c1, c2 = c[:len(c) // 2], c[len(c) // 2:]   # honest split: Platt on c1, thresholds on c2 (amendment 4 Oct)
                RA = platt(V["fwd"][c1], y[c1])
                q, _ = per_cell_quantile(cells[c2], label_scores(RA(V["fwd"][c2]), y[c2]), M, ALPHA, FLOOR)
                qt = q[cells[te]]; pA = RA(V["fwd"][te])
                Tq = map_quantile(V[B][f], V["fwd"][f]); Tp, inc = map_paired(V[B][f], V["fwd"][f])
                Tg = map_paired_group(V[B][f], V["fwd"][f], cells[f], cells[te])
                Tm = map_quantile_mid(V[B][f], V["fwd"][f])
                Ti, Td = map_iso(V[B][f], V["fwd"][f], True), map_iso(V[B][f], V["fwd"][f], False)
                arms = {"A->A": pA, "naive": RA(V[B][te]), "ot_quantile": RA(Tq(V[B][te])),
                        "paired_iso": RA(Tp(V[B][te])), "paired_iso_group": RA(Tg(V[B][te], cells[te])),
                        "ot_quantile_mid": RA(Tm(V[B][te])), "paired_iso_inc": RA(Ti(V[B][te])),
                        "paired_iso_dec": RA(Td(V[B][te]))}
                for name, p in arms.items():
                    per.setdefault(name, []).append(evaluate(pA, p, qt, y[te], cells=cells[te]))
                for k in (30, 120):
                    per.setdefault(f"recal_B_k{k}", []).append(recal_reference(V[B], y, c[:k], te, pA, qt, cells[te]))
                per.setdefault("_direction_increasing", []).append(inc)
            res[B] = {name: ({k: float(np.mean([r[k] for r in v])) for k in v[0]} |
                             {"coverage_q10_q90": np.quantile([r["coverage"] for r in v], [.1, .9]).tolist(),
                              "bound_holds_rate": float(np.mean([abs(r["coverage"] - a["coverage"]) <= r["crossing_mass"] + 1e-12
                                                                 for r, a in zip(v, per["A->A"])]))})
                      if not name.startswith("_") else float(np.mean(v)) for name, v in per.items()}
        res["_tie_fraction"] = {p: tie_fraction(v) for p, v in V.items()}
        out[model] = res
        print("frozen done", model, flush=True)
    return out


def tinker(boot=200):
    d = load_tinker(TINKER, cache="inputs/tinker_triviaqa/aligned.npz")
    sp, y = d["split"], d["correct"].astype(int)
    f, c, te = sp == "train", sp == "cal", sp == "test"; out = {}
    rng = np.random.default_rng(8)
    for arm in ("q", "qa"):
        VA, VB = d[f"p_{arm}_trained_normal"], d[f"p_{arm}_trained_reversed"]
        ci = np.flatnonzero(c); c1, c2 = ci[:len(ci) // 2], ci[len(ci) // 2:]   # honest split (amendment 4 Oct)
        RA = platt(VA[c1], y[c1]); q, _ = per_cell_quantile(np.zeros(len(c2), int), label_scores(RA(VA[c2]), y[c2]), 1, ALPHA, FLOOR)
        qt = np.full(te.sum(), q[0]); pA = RA(VA[te])
        Tq = map_quantile(VB[f], VA[f]); Tp, inc = map_paired(VB[f], VA[f]); Tm = map_quantile_mid(VB[f], VA[f])
        arms = {"A->A": pA, "naive": RA(VB[te]), "ot_quantile": RA(Tq(VB[te])), "paired_iso": RA(Tp(VB[te])),
                "ot_quantile_mid": RA(Tm(VB[te])), "paired_iso_inc": RA(map_iso(VB[f], VA[f], True)(VB[te])),
                "paired_iso_dec": RA(map_iso(VB[f], VA[f], False)(VB[te]))}
        res = {"paired_direction_increasing": bool(inc), "tie_fraction": {"A": tie_fraction(VA), "B": tie_fraction(VB)}}
        yt = y[te]
        for name, p in arms.items():
            point = evaluate(pA, p, qt, yt)
            bs = [evaluate(pA[i], p[i], qt[i], yt[i], cal=False)["coverage"] for i in (rng.integers(len(yt), size=len(yt)) for _ in range(boot))]
            res[name] = point | {"coverage_ci95": np.quantile(bs, [.025, .975]).tolist()}
        for k in (30, 120):
            res[f"recal_B_k{k}"] = recal_reference(VB, y, np.flatnonzero(c)[:k], np.flatnonzero(te), pA, qt) | {"coverage_ci95": None}
        out[arm] = res
    return out


def main(out="runs/p1_e8"):
    res = {"criterion": "reports/estimand_note_E8_2026-10-02.md", "frozen": frozen(), "tinker": tinker(),
           "source_sha256": source_hash()}
    write_json(Path(out) / "e8_result.json", res)
    for m, r in res["frozen"].items():
        for B, arms in ((k, v) for k, v in r.items() if not k.startswith("_")):
            print(m, B, {k: (round(v["coverage"], 3), round(v["set_size"], 2), round(v["brier"], 4), round(v["crossing_mass"], 3), round(v["bound_holds_rate"], 3))
                         for k, v in arms.items() if isinstance(v, dict)}, "inc", arms["_direction_increasing"])
    for a, r in res["tinker"].items():
        print("tinker", a, {k: (round(v["coverage"], 3), (v["coverage_ci95"] and [round(x, 3) for x in v["coverage_ci95"]]), round(v["brier"], 4))
                            for k, v in r.items() if isinstance(v, dict) and "coverage" in v}, "inc", r["paired_direction_increasing"])


if __name__ == "__main__":
    main()
