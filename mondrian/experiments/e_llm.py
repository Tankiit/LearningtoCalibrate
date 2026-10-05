"""E1-E3 (dated note reports/estimand_note_E123_2026-10-02.md).

E1: TriviaQA, Tinker reporters q (stage 0) vs qa (stage 1): Brier-vs-M forecast sweep
    and Mondrian label-set conformal, incl. an answerable-only (stage-2) arm.
E2: same data, legend convention change between calibration and deployment.
E3: PopQA Llama-3.1-8B states, exemplar cells in representation space vs on the probe.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from data.tinker import load_tinker, DEFAULT as TINKER_DEFAULT
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash, write_free_check, array_hash
from mondrian.conformal import per_cell_quantile
from .common import validate_first, require_tests
from .x1_synthetic import _smoothed_argmin

ALPHA, FLOOR = 0.1, 20
M_GRID_E1 = [1, 2, 3, 4, 5, 6, 8, 11, 14, 18, 23, 30, 39, 51, 67, 87, 112, 146, 190]
N_GRID_E1 = [300, 500, 800, 1300, 2000, 3200, 5000, 8000]
M_GRID_E3 = [1, 2, 3, 4, 5, 6, 8, 11, 14, 18, 23, 30, 39, 51, 67]
N_GRID_E3 = [200, 300, 500, 800, 1300, 1770]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


# ------------------------------------------------------------------ forecast side
def binned_forecast(cc, yc, ct, M, floor=FLOOR):
    """Cell rate on calibration rows; below-floor cells fall back to the pooled rate."""
    n = np.bincount(cc, minlength=M)
    rate = np.divide(np.bincount(cc, weights=yc, minlength=M), n, out=np.zeros(M), where=n > 0)
    pooled = float(np.mean(yc))
    below = n < floor
    rate[below] = pooled
    return rate[ct], rate, n, below


def forecast_sweep(Zf, Zc_pool, yc_pool, Zt, yt, M_grid, n_grid, seeds, label):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 707]))
        order = rng.permutation(len(Zc_pool))
        for M in M_grid:
            part = ExemplarPartition.fit(Zf, M, "kmeans", seed)
            write_free_check(part, Zt[:10], Zt[10:40])
            ct, cpool = part(Zt), part(Zc_pool)
            nt = np.bincount(ct, minlength=M)
            yt_cell = np.divide(np.bincount(ct, weights=yt, minlength=M), nt, out=np.zeros(M), where=nt > 0)
            for n_cal in n_grid:
                idx = order[:n_cal]
                pt, rate, n, below = binned_forecast(cpool[idx], yc_pool[idx], ct, M)
                obs = nt > 0
                rows.append(dict(signal=label, seed=seed, M=M, n_cal=n_cal,
                    brier=float(np.mean((pt - yt) ** 2)),
                    cell_level=float(np.sum(nt[obs] * (rate[obs] - yt_cell[obs]) ** 2) / nt.sum()),
                    n_below_floor=int(below.sum()), test_mass_below_floor=float(np.mean(below[ct])),
                    counts_cal=n.tolist(), counts_test=nt.tolist(),
                    exemplar_hash=part.metadata["exemplar_hash"]))
        print(f"  {label}: seed {seed} done", flush=True)
    return rows


def forecast_analysis(rows, n_boot=1000, seed=2027, min_n=None):
    seeds = sorted({r["seed"] for r in rows}); ns = sorted({r["n_cal"] for r in rows})
    ms = sorted({r["M"] for r in rows}); log_m = np.log(ms)
    lk = {(r["seed"], r["n_cal"], r["M"]): r for r in rows}
    brier = np.array([[[lk[s, n, m]["brier"] for m in ms] for n in ns] for s in seeds])
    cell = np.array([[[lk[s, n, m]["cell_level"] for m in ms] for n in ns] for s in seeds])
    use = [j for j, n in enumerate(ns) if min_n is None or n >= min_n]

    def estimate(idx):
        curve = np.log(brier[idx].mean(0))
        fits = [_smoothed_argmin(curve[j], log_m) for j in range(len(ns))]
        lm = np.array([f[0] for f in fits]); b = np.array([f[1] for f in fits])
        upper = np.array([np.argmin(curve[j]) == len(ms) - 1 for j in range(len(ns))])
        slope = np.polyfit(np.log(ns)[use], lm[use], 1)[0]
        return slope, np.exp(lm), b, upper
    slope, mstar, boundary, upper = estimate(np.arange(len(seeds)))
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        s_, _, b_, _ = estimate(rng.integers(len(seeds), size=len(seeds)))
        boots.append((s_, b_[use].any()))
    boots = np.array(boots, float)
    mean_b = brier.mean(0)
    return {"n_cal_grid": ns, "M_grid": ms, "n_seeds": len(seeds), "criterion_n_cal": [ns[j] for j in use],
            "Mstar_smoothed": mstar.tolist(), "boundary_flags": boundary.tolist(),
            "upper_boundary_discrete": upper.tolist(),
            "interior_bootstrap_rate": float(1 - boots[:, 1].mean()),
            "slope": float(slope), "slope_ci95": np.quantile(boots[:, 0], [.025, .975]).tolist(),
            "cell_level_argmin": [int(ms[int(np.argmin(cell[:, j].mean(0)))]) for j in range(len(ns))],
            "brier_min": [float(mean_b[j].min()) for j in range(len(ns))],
            "brier_M1": [float(mean_b[j][0]) for j in range(len(ns))],
            "brier_curve_largest_n": dict(zip(map(str, ms), mean_b[-1].tolist()))}


def forecast_decision(a):
    use = [j for j, n in enumerate(a["n_cal_grid"]) if n in a["criterion_n_cal"]]
    checks = {"interior": a["interior_bootstrap_rate"] >= 0.90,
              "slope_positive": a["slope_ci95"][0] > 0,
              "cell_level_argmin_M1": all(a["cell_level_argmin"][j] == 1 for j in use)}
    if all(checks.values()):
        status = "PASS"
    elif (not checks["interior"]) and checks["cell_level_argmin_M1"] and \
            any(a["upper_boundary_discrete"][j] for j in use):
        status = "INCONCLUSIVE"
    else:
        status = "FAIL"
    return status, checks


# ------------------------------------------------------------------ conformal side
def label_scores(p, y):
    """s(x,y) = 1 - p_hat(y|x) with p_hat(1)=p, p_hat(0)=1-p."""
    p = np.asarray(p, float)
    return np.where(np.asarray(y) == 1, 1 - p, p)


def conformal_arm(name, cc, sc, ct, p_test, y_test, M):
    q, counts = per_cell_quantile(cc, sc, M, ALPHA, FLOOR)
    s1, s0 = label_scores(p_test, 1), label_scores(p_test, 0)
    size = (s1 <= q[ct]).astype(int) + (s0 <= q[ct]).astype(int)
    hit = label_scores(p_test, y_test) <= q[ct]
    nt = np.bincount(ct, minlength=M)
    cov = np.divide(np.bincount(ct, weights=hit, minlength=M), nt, out=np.full(M, np.nan), where=nt > 0)
    t, N = 1 - ALPHA, len(hit); c = hit.mean()
    z = 1.959964; denom = 1 + z**2 / N
    centre = (c + z**2 / (2 * N)) / denom; half = z * np.sqrt(c * (1 - c) / N + z**2 / (4 * N**2)) / denom
    tol = 4 * np.sqrt(t * (1 - t) / np.maximum(nt, 1))
    return {"arm": name, "M": M, "coverage": float(c), "wilson95": [float(centre - half), float(centre + half)],
            "coverage_Y1": float(hit[y_test == 1].mean()), "coverage_Y0": float(hit[y_test == 0].mean()),
            "mean_set_size": float(size.mean()), "empty_rate": float(np.mean(size == 0)),
            "full_rate": float(np.mean(size == 2)), "cell_coverage": cov.tolist(), "counts_test": nt.tolist(),
            "counts_cal": counts.tolist(), "thresholds": q.tolist(), "n_below_floor": int(np.sum(counts < FLOOR)),
            "test_mass_infinite": float(np.mean(np.isinf(q[ct]))),
            "cells_within_4se": bool(np.all((cov >= t - tol)[nt > 0])),
            "worst_cell_coverage": float(np.nanmin(cov))}


def valid_arm_ok(r):
    return r["wilson95"][1] >= 1 - ALPHA and r["cells_within_4se"]


# ------------------------------------------------------------------ E1 / E2
def run_e1_e2(out, seeds, path):
    d = load_tinker(path, cache="inputs/tinker_triviaqa/aligned.npz")
    sp, y = d["split"], d["correct"].astype(float)
    f, c, te = (sp == "train"), (sp == "cal"), (sp == "test")
    out = Path(out); result = {"source_sha256": str(d["source_sha256"]), "n": {"fit": int(f.sum()),
              "cal": int(c.sum()), "test": int(te.sum())}, "forecast": {}, "conformal": {}, "e2": {}}
    # E1 forecast sweep
    for arm in ("q", "qa"):
        p = d[f"p_{arm}_trained_normal"]; Z = logit(p)
        rows = forecast_sweep(Z[f], Z[c], y[c], Z[te], y[te], M_GRID_E1, N_GRID_E1, seeds, f"e1_{arm}")
        write_json(out / f"e1_forecast_rows_{arm}.json", rows)
        a = forecast_analysis(rows); status, checks = forecast_decision(a)
        result["forecast"][arm] = {"status": status, "checks": checks, **a}
    # E1 conformal, full pool, M=8
    M = 8; parts = {}
    for arm in ("q", "qa"):
        Z = logit(d[f"p_{arm}_trained_normal"])
        parts[arm] = ExemplarPartition.fit(Z[f], M, "kmeans", 0)
    p_ref = d["p_qa_trained_normal"]          # score model: answer-aware reporter
    sc = label_scores(p_ref[c], y[c].astype(int)); yt = y[te].astype(int)
    arms = [conformal_arm("marginal", np.zeros(c.sum(), int), sc, np.zeros(te.sum(), int), p_ref[te], yt, 1)]
    for arm in ("q", "qa"):
        Z = logit(d[f"p_{arm}_trained_normal"])
        arms.append(conformal_arm(f"{arm}_cells_stage{0 if arm == 'q' else 1}", parts[arm](Z[c]), sc,
                                  parts[arm](Z[te]), p_ref[te], yt, M))
    ans = c & (d["correct"] == 1)
    arms.append(conformal_arm("answerable_only_stage2", np.zeros(ans.sum(), int),
                              label_scores(p_ref[ans], 1), np.zeros(te.sum(), int), p_ref[te], yt, 1))
    result["conformal"] = {"score_reporter": "qa_trained_normal", "arms": arms,
        "status": "PASS" if all(valid_arm_ok(a) for a in arms[:3]) and arms[3]["coverage"] < 1 - ALPHA - .02 else "FAIL"}
    # E2: convention change. Cells, thresholds and forecasts from one legend; test from another.
    e2_ok_consistent, e2_break = True, False
    for arm in ("q", "qa"):
        res = {}
        for cal_leg, test_leg in (("normal", "normal"), ("reversed", "reversed"), ("normal", "reversed")):
            pc, pt = d[f"p_{arm}_trained_{cal_leg}"], d[f"p_{arm}_trained_{test_leg}"]
            part = ExemplarPartition.fit(logit(pc)[f], M, "kmeans", 0)
            cc, ct = part(logit(pc)[c]), part(logit(pt)[te])
            r = conformal_arm(f"{arm}:{cal_leg}->{test_leg}", cc, label_scores(pc[c], y[c].astype(int)),
                              ct, pt[te], yt, M)
            ptilde, *_ = binned_forecast(cc, y[c], ct, M)
            r["brier_binned"] = float(np.mean((ptilde - y[te]) ** 2))
            r["brier_raw"] = float(np.mean((pt[te] - y[te]) ** 2))
            res[f"{cal_leg}->{test_leg}"] = r
            if cal_leg == test_leg:
                e2_ok_consistent &= valid_arm_ok(r)
            elif r["coverage"] < 1 - ALPHA - .02 or r["worst_cell_coverage"] < 1 - ALPHA - .02:
                e2_break = True
        result["e2"][arm] = res
    result["e2"]["status"] = "PASS" if e2_ok_consistent and e2_break else "FAIL"
    result["e2"]["checks"] = {"consistent_arms_valid": e2_ok_consistent, "mismatch_breaks": e2_break}
    return result


# ------------------------------------------------------------------ E3
def run_e3(out, seeds, root="inputs/popqa_llama"):
    root = Path(root)
    rows = [json.loads(l) for l in (root / "rows.jsonl").open()]
    z = np.load(root / "states.npz")
    ids = [r["row_id"] for r in rows]
    if list(z["row_ids"]) != ids:
        raise ValueError("State rows not aligned with rows.jsonl")
    X = z["states"].astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True)
    sp = np.array([r["split"] for r in rows]); y = np.array([r["correct"] for r in rows], float)
    p = np.array([r["anchor_p"] for r in rows])
    f, c, te = sp == "fit", np.isin(sp, ["cal", "update"]), sp == "test"
    pca = PCA(32, random_state=0).fit(X[f]); Zrep = pca.transform(X)
    Zprobe = logit(p)
    result = {"n": {"fit": int(f.sum()), "cal_pool": int(c.sum()), "test": int(te.sum())},
              "pca_explained": float(pca.explained_variance_ratio_.sum()), "forecast": {}}
    for label, Z in (("rep_cells", Zrep), ("probe_cells", Zprobe)):
        r_ = forecast_sweep(Z[f], Z[c], y[c], Z[te], y[te], M_GRID_E3, N_GRID_E3, seeds, f"e3_{label}")
        write_json(Path(out) / f"e3_forecast_rows_{label}.json", r_)
        a = forecast_analysis(r_, min_n=500); status, checks = forecast_decision(a)
        result["forecast"][label] = {"status": status, "checks": checks, **a}
    result["brier_raw_probe"] = float(np.mean((p[te] - y[te]) ** 2))
    M = 8; part = ExemplarPartition.fit(Zrep[f], M, "kmeans", 0)
    sc = label_scores(p[c], y[c].astype(int)); yt = y[te].astype(int)
    arms = [conformal_arm("marginal", np.zeros(c.sum(), int), sc, np.zeros(te.sum(), int), p[te], yt, 1),
            conformal_arm("rep_cells", part(Zrep[c]), sc, part(Zrep[te]), p[te], yt, M)]
    ans = c & (y == 1)
    arms.append(conformal_arm("answerable_only_stage2", np.zeros(ans.sum(), int), label_scores(p[ans], 1),
                              np.zeros(te.sum(), int), p[te], yt, 1))
    result["conformal"] = {"arms": arms, "status": "PASS" if all(valid_arm_ok(a) for a in arms[:2])
                           and arms[2]["coverage"] < 1 - ALPHA - .02 else "FAIL"}
    return result


# ------------------------------------------------------------------ Addendum A (post hoc)
def forecast_sweep_resplit(Zf, Zpool, ypool, M_grid, n_grid, seeds, label, n_max):
    """Per seed: random disjoint cal/test split of the pool; test = rows after n_max."""
    rows = []
    for seed in seeds:
        perm = np.random.default_rng(np.random.SeedSequence([seed, 909])).permutation(len(Zpool))
        cal, te = perm[:n_max], perm[n_max:]
        rows += forecast_sweep(Zf, Zpool[cal], ypool[cal], Zpool[te], ypool[te], M_grid, n_grid, [seed], label)
    return rows


def expectation_check(results, t=1 - ALPHA):
    """X4 rule: seed-mean coverage >= t - 3 SE; each cell's seed-mean >= t - 4 SE."""
    cov = np.array([r["coverage"] for r in results])
    cells = np.array([r["cell_coverage"] for r in results], float)
    mean_c = np.nanmean(cells, 0); se_c = np.nanstd(cells, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(cells), 0))
    se = cov.std(ddof=1) / np.sqrt(len(cov))
    return {"coverage_mean": float(cov.mean()), "coverage_se": float(se),
            "cell_mean": mean_c.tolist(), "cell_se": se_c.tolist(),
            "worst_cell_mean": float(np.nanmin(mean_c)),
            "min_cell_z": float(np.nanmin((mean_c - t) / np.where(se_c > 0, se_c, 1e-12))),
            "valid": bool(cov.mean() >= t - 3 * se and np.nanmin((mean_c - t) / np.where(se_c > 0, se_c, 1e-12)) >= -4),
            "mean_set_size": float(np.mean([r["mean_set_size"] for r in results])), "n_seeds": len(cov)}


def run_e12_corrected(out, path, seeds_forecast=range(20), seeds_conf=range(50), n_max=8000, M=8):
    d = load_tinker(path, cache="inputs/tinker_triviaqa/aligned.npz")
    sp, y = d["split"], d["correct"].astype(float)
    f = sp == "train"; pool = np.flatnonzero(np.isin(sp, ["cal", "test"]))
    res = {"pool": int(len(pool)), "pool_accuracy": float(y[pool].mean()), "forecast": {}, "e2": {}}
    for arm in ("q", "qa"):
        Z = logit(d[f"p_{arm}_trained_normal"])
        rows = forecast_sweep_resplit(Z[f], Z[pool], y[pool], M_GRID_E1, N_GRID_E1, seeds_forecast,
                                      f"e1x_{arm}", n_max)
        write_json(Path(out) / f"e1x_forecast_rows_{arm}.json", rows)
        a = forecast_analysis(rows); status, checks = forecast_decision(a)
        res["forecast"][arm] = {"status": status, "checks": checks, **a}
    consistent_ok, broke = True, False
    for arm in ("q", "qa"):
        res["e2"][arm] = {}
        for cal_leg, test_leg in (("normal", "normal"), ("reversed", "reversed"), ("normal", "reversed")):
            pc, pt = d[f"p_{arm}_trained_{cal_leg}"], d[f"p_{arm}_trained_{test_leg}"]
            part = ExemplarPartition.fit(logit(pc)[f], M, "kmeans", 0)
            per = []
            for seed in seeds_conf:
                perm = pool[np.random.default_rng(np.random.SeedSequence([seed, 909])).permutation(len(pool))]
                cal, te = perm[:n_max], perm[n_max:]
                per.append(conformal_arm(f"{arm}:{cal_leg}->{test_leg}", part(logit(pc)[cal]),
                    label_scores(pc[cal], y[cal].astype(int)), part(logit(pt)[te]), pt[te], y[te].astype(int), M))
            e = expectation_check(per); res["e2"][arm][f"{cal_leg}->{test_leg}"] = e
            if cal_leg == test_leg:
                consistent_ok &= e["valid"]
            elif e["coverage_mean"] < 1 - ALPHA - .02 or e["worst_cell_mean"] < 1 - ALPHA - .02:
                broke = True
    res["e2"]["checks"] = {"consistent_arms_valid": consistent_ok, "mismatch_breaks": broke}
    res["e2"]["status"] = "PASS" if consistent_ok and broke else "FAIL"
    return res


def load_popqa_with_hq(root="inputs/popqa_llama", hq_path=None):
    import torch
    root = Path(root)
    rows = [json.loads(l) for l in (root / "rows.jsonl").open()]
    m = json.loads((root / "manifest.json").read_text())
    hq_path = hq_path or m["source_files"]["hidden_states"]["path"]
    h = torch.load(hq_path, map_location="cpu", weights_only=False)
    pos = {q: i for i, q in enumerate(np.asarray(h["example_ids"]).astype(str))}
    k = np.array([pos[r["question_id"]] for r in rows])
    S = np.load(root / "states.npz")["states"].astype(np.float64)
    side = np.array([r["candidate_side"] == "pos" for r in rows])
    ans = np.where(side[:, None], h["h_pos"].double().numpy()[k], h["h_neg"].double().numpy()[k])
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    cos = np.sum(unit(ans) * unit(S), 1)
    if cos.min() < 0.9999:
        raise ValueError(f"h_q alignment check failed: min cosine {cos.min()}")
    return rows, unit(S), unit(h["h_q"].double().numpy()[k]), float(cos.min())


def run_e3_fixed(out, seeds_forecast=range(20), seeds_conf=range(50), n_max=2500, M=8):
    rows, X, Hq, min_cos = load_popqa_with_hq()
    sp = np.array([r["split"] for r in rows]); y = np.array([r["correct"] for r in rows], float)
    p = np.array([r["anchor_p"] for r in rows])
    f = sp == "fit"; pool = np.flatnonzero(~f)
    feats = {}
    for name, A, dim in (("rep_pca8", X, 8), ("rep_pca32", X, 32), ("prompt_pca8", Hq, 8)):
        pca = PCA(dim, random_state=0).fit(A[f]); feats[name] = (pca.transform(A), float(pca.explained_variance_ratio_.sum()))
    feats["probe_cells"] = (logit(p), None)
    res = {"pool": int(len(pool)), "alignment_min_cosine": min_cos, "forecast": {}, "conformal": {}}
    n_grid = [200, 300, 500, 800, 1300, 2000, 2500]
    for name, (Z, ev) in feats.items():
        r_ = forecast_sweep_resplit(Z[f], Z[pool], y[pool], M_GRID_E3, n_grid, seeds_forecast, f"e3x_{name}", n_max)
        write_json(Path(out) / f"e3x_forecast_rows_{name}.json", r_)
        a = forecast_analysis(r_, min_n=500); status, checks = forecast_decision(a)
        res["forecast"][name] = {"status": status, "checks": checks, "pca_explained": ev, **a}
    parts = {n: ExemplarPartition.fit(feats[n][0][f], M, "kmeans", 0) for n in ("rep_pca8", "prompt_pca8")}
    arms = {"marginal": [], "rep_pca8_cells_stage1": [], "prompt_pca8_cells_stage0": [], "answerable_only_stage2": []}
    for seed in seeds_conf:
        perm = pool[np.random.default_rng(np.random.SeedSequence([seed, 909])).permutation(len(pool))]
        cal, te = perm[:n_max], perm[n_max:]
        sc, yt = label_scores(p[cal], y[cal].astype(int)), y[te].astype(int)
        arms["marginal"].append(conformal_arm("marginal", np.zeros(len(cal), int), sc, np.zeros(len(te), int), p[te], yt, 1))
        for n, key in (("rep_pca8", "rep_pca8_cells_stage1"), ("prompt_pca8", "prompt_pca8_cells_stage0")):
            Z = feats[n][0]
            arms[key].append(conformal_arm(key, parts[n](Z[cal]), sc, parts[n](Z[te]), p[te], yt, M))
        ans = cal[y[cal] == 1]
        arms["answerable_only_stage2"].append(conformal_arm("answerable", np.zeros(len(ans), int),
            label_scores(p[ans], 1), np.zeros(len(te), int), p[te], yt, 1))
    res["conformal"] = {k: expectation_check(v) for k, v in arms.items()}
    valid = all(res["conformal"][k]["valid"] for k in ("marginal", "rep_pca8_cells_stage1", "prompt_pca8_cells_stage0"))
    brk = res["conformal"]["answerable_only_stage2"]["coverage_mean"] < 1 - ALPHA - .02
    res["conformal"]["checks"] = {"valid_arms": valid, "stage2_breaks": brk}
    res["conformal"]["status"] = "PASS" if valid and brk else "FAIL"
    return res


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--which", choices=["e12", "e3", "e12x", "e3x"], required=True)
    pa.add_argument("--out", default="runs/p1_e")
    pa.add_argument("--seeds", type=int, default=20)
    pa.add_argument("--tinker", default=TINKER_DEFAULT)
    a = pa.parse_args(); validate_first(a.out); require_tests(a.out)
    res = {"e12": lambda: run_e1_e2(a.out, range(a.seeds), a.tinker), "e3": lambda: run_e3(a.out, range(a.seeds)),
           "e12x": lambda: run_e12_corrected(a.out, a.tinker), "e3x": lambda: run_e3_fixed(a.out)}[a.which]()
    if a.which.endswith("x"):
        res["status_note"] = "post hoc sensitivity (Addendum A); pre-set statuses unchanged"
    res.update(criterion="reports/estimand_note_E123_2026-10-02.md", source_sha256_code=source_hash(),
               paper_status="HOLD")
    write_json(Path(a.out) / f"{a.which}_result.json", res)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else None) for k, v in res.items()
                      if isinstance(v, dict)}, indent=1))
    for k in ("forecast",):
        for s, v in res[k].items():
            print(s, v["status"], v["checks"], "M*", np.round(v["Mstar_smoothed"], 1).tolist(), "slope", round(v["slope"], 3), v["slope_ci95"])


if __name__ == "__main__":
    main()
