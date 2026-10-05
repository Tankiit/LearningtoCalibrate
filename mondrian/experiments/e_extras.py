"""Addendum B (reports/estimand_note_E123_2026-10-02.md): top-k and balanced cells on the
E1 data, and the discrete-argmin interior check across X1, E1x and E3x."""
import argparse
import json
from pathlib import Path
import numpy as np
from data.tinker import load_tinker, DEFAULT as TINKER_DEFAULT
from models.partition import ExemplarPartition
from mondrian.cells import assign_topk
from mondrian.audit import write_json, source_hash, write_free_check
from .common import validate_first, require_tests
from .e_llm import logit, label_scores, conformal_arm, expectation_check, binned_forecast, ALPHA, FLOOR
from .topk import pooled_quantiles


def resplit(pool, seed, n_cal):
    perm = pool[np.random.default_rng(np.random.SeedSequence([seed, 909])).permutation(len(pool))]
    return perm[:n_cal], perm[n_cal:]


def topk_run(d, seeds=range(50), M=16, ks=(1, 2, 4), n_cal=8000):
    sp, y = d["split"], d["correct"].astype(int)
    p = d["p_qa_trained_normal"]; Z = logit(p)
    f = sp == "train"; pool = np.flatnonzero(np.isin(sp, ["cal", "test"]))
    part = ExemplarPartition.fit(Z[f], M, "kmeans", 0)
    out = {}
    for k in ks:
        rule = lambda x: assign_topk(x, part.exemplars, k)
        write_free_check(rule, Z[pool][:10], Z[pool][10:40])
        per = []
        for seed in seeds:
            cal, te = resplit(pool, seed, n_cal)
            mc, mt = rule(Z[cal]), rule(Z[te])
            thr, counts = pooled_quantiles(mc, label_scores(p[cal], y[cal]), mt, ALPHA, FLOOR)
            hit = label_scores(p[te], y[te]) <= thr
            size = (label_scores(p[te], 1) <= thr).astype(int) + (label_scores(p[te], 0) <= thr).astype(int)
            keys = [tuple(r) for r in np.sort(mt, axis=1)]
            pools = {}
            for kk, h in zip(keys, hit):
                pools.setdefault(kk, []).append(h)
            per.append({"coverage": float(hit.mean()), "set_size": float(size.mean()),
                        "pool_coverage": {",".join(map(str, kk)): (float(np.mean(v)), len(v)) for kk, v in pools.items()},
                        "test_fraction_below_floor": float(np.mean(counts < FLOOR))})
        cov = np.array([r["coverage"] for r in per])
        allkeys = sorted({kk for r in per for kk in r["pool_coverage"]})
        pool_means = {}
        for kk in allkeys:
            vals = [r["pool_coverage"][kk][0] for r in per if kk in r["pool_coverage"]]
            ns = [r["pool_coverage"][kk][1] for r in per if kk in r["pool_coverage"]]
            pool_means[kk] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None,
                              float(np.mean(ns)))
        worst = min(pool_means.items(), key=lambda kv: kv[1][0])
        res = {"k": k, "coverage_mean": float(cov.mean()), "coverage_se": float(cov.std(ddof=1) / np.sqrt(len(cov))),
               "set_size_mean": float(np.mean([r["set_size"] for r in per])), "n_pools": len(allkeys),
               "worst_pool": {"pool": worst[0], "coverage_mean": worst[1][0], "se": worst[1][1], "mean_n_test": worst[1][2]},
               "test_fraction_below_floor": float(np.mean([r["test_fraction_below_floor"] for r in per]))}
        if k == 1:
            t = 1 - ALPHA; m_ = np.array([v[0] for v in pool_means.values()]); s_ = np.array([v[1] for v in pool_means.values()])
            res["valid_A2_rule"] = bool(cov.mean() >= t - 3 * res["coverage_se"] and np.min((m_ - t) / s_) >= -4)
        out[f"k{k}"] = res
    return out


def balanced_run(d, seeds=range(20), Ms=(8, 16, 32), n_cal=8000):
    sp, y = d["split"], d["correct"]
    p = d["p_qa_trained_normal"]; Z = logit(p)
    f = sp == "train"; pool = np.flatnonzero(np.isin(sp, ["cal", "test"]))
    out = {}
    for M in Ms:
        parts = {rule: ExemplarPartition.fit(Z[f], M, "kmeans", 0, rule) for rule in ("voronoi", "power")}
        per = {rule: [] for rule in parts}
        for seed in seeds:
            cal, te = resplit(pool, seed, n_cal)
            for rule, part in parts.items():
                cc, ct = part(Z[cal]), part(Z[te])
                ptilde, *_ = binned_forecast(cc, y[cal].astype(float), ct, M)
                r = conformal_arm(rule, cc, label_scores(p[cal], y[cal]), ct, p[te], y[te].astype(int), M)
                r["brier"] = float(np.mean((ptilde - y[te]) ** 2))
                per[rule].append(r)
        diff_b = np.array([a["brier"] - b["brier"] for a, b in zip(per["power"], per["voronoi"])])
        diff_w = np.array([a["worst_cell_coverage"] - b["worst_cell_coverage"] for a, b in zip(per["power"], per["voronoi"])])
        ci = lambda x: [float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))), float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x)))]
        out[f"M{M}"] = {rule: {**expectation_check(v), "brier_mean": float(np.mean([r["brier"] for r in v])),
                               "fit_masses": parts[rule].metadata.get("power_fit", {}).get("masses") if rule == "power" else
                               (np.bincount(parts[rule](Z[f]), minlength=M) / f.sum()).tolist()}
                        for rule, v in per.items()}
        out[f"M{M}"]["power_minus_voronoi"] = {"brier": float(diff_b.mean()), "brier_ci95": ci(diff_b),
                                               "worst_cell": float(diff_w.mean()), "worst_cell_ci95": ci(diff_w)}
    return out


def discrete_interior(path, key, n_boot=1000, min_n=None):
    rows = json.load(open(path))
    seeds = sorted({r["seed"] for r in rows}); ns = sorted({r["n_cal"] for r in rows}); ms = sorted({r["M"] for r in rows})
    lk = {(r["seed"], r["n_cal"], r["M"]): r[key] for r in rows}
    cube = np.array([[[lk[s, n, m] for m in ms] for n in ns] for s in seeds], float)
    use = [j for j, n in enumerate(ns) if min_n is None or n >= min_n]
    rng = np.random.default_rng(2027); ok = 0; per_n = np.zeros(len(ns))
    for _ in range(n_boot):
        a = np.argmin(cube[rng.integers(len(seeds), size=len(seeds))].mean(0), axis=1)
        inner = (a > 0) & (a < len(ms) - 1)
        per_n += inner; ok += inner[use].all()
    return {"all_criterion_n_interior_rate": ok / n_boot, "per_n_cal": dict(zip(map(str, ns), (per_n / n_boot).round(3).tolist())),
            "criterion_n_cal": [ns[j] for j in use]}


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--out", default="runs/p1_extras")
    pa.add_argument("--tinker", default=TINKER_DEFAULT)
    a = pa.parse_args(); validate_first(a.out); require_tests(a.out)
    d = load_tinker(a.tinker, cache="inputs/tinker_triviaqa/aligned.npz")
    res = {"criterion": "reports/estimand_note_E123_2026-10-02.md#addendum-b", "status": "DESCRIPTIVE",
           "topk": topk_run(d), "balanced": balanced_run(d)}
    res["discrete_interior"] = {
        "x1_d1": discrete_interior("runs/p1_x1_v2_d1/rows.json", "pointwise_msce"),
        "x1_d2": discrete_interior("runs/p1_x1_v2_d2/rows.json", "pointwise_msce"),
        "e1x_q": discrete_interior("runs/p1_e12x/e1x_forecast_rows_q.json", "brier"),
        "e1x_qa": discrete_interior("runs/p1_e12x/e1x_forecast_rows_qa.json", "brier"),
        "e3x_probe": discrete_interior("runs/p1_e3x/e3x_forecast_rows_probe_cells.json", "brier", min_n=500)}
    res["source_sha256"] = source_hash(); res["paper_status"] = "HOLD"
    write_json(Path(a.out) / "extras_result.json", res)
    print(json.dumps({"topk": {k: {kk: v[kk] for kk in ("coverage_mean", "set_size_mean", "worst_pool")} for k, v in res["topk"].items()},
                      "balanced": {k: v["power_minus_voronoi"] for k, v in res["balanced"].items()},
                      "discrete": {k: v["all_criterion_n_interior_rate"] for k, v in res["discrete_interior"].items()}}, indent=1))


if __name__ == "__main__":
    main()
