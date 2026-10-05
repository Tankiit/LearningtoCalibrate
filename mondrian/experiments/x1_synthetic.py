"""X1: known score CDF, exact decomposition, Beta law, and M* gate."""
import argparse
from pathlib import Path
import time
import numpy as np
from data.synthetic import sample, coverage_cdf
from models.partition import ExemplarPartition
from mondrian.audit import array_hash, manifest, write_free_check, write_json, source_hash
from mondrian.conformal import per_cell_quantile, coverage_by_cell, functionals
from mondrian.splits import assert_disjoint, freeze_splits, split_hash
from mondrian.theory import beta_law_cell_moments
from .common import validate_first, require_tests, summarize

METRICS = ("msce", "spread", "concentrated", "pointwise_msce")


def decompose(probabilities, labels, M, target):
    """Exact empirical orthogonal decomposition using known CDF values.

    E_x[(F_x(q_cell)-t)^2] = within-cell heterogeneity + cell-average error.
    Heterogeneity uses the realized q; it is not a calibration-independent bias.
    """
    n = np.bincount(labels, minlength=M)
    means = np.divide(np.bincount(labels, weights=probabilities, minlength=M), n,
                      out=np.zeros(M), where=n>0)
    within = float(np.mean((probabilities-means[labels])**2))
    between = float(np.sum(n*(means-target)**2)/len(labels))
    total = float(np.mean((probabilities-target)**2))
    return within, between, total, means, n


def run(M_grid, n_cal_grid, d, seeds, functional="pointwise_msce", *,
        n_fit=2000, n_test=12000, alpha=0.1, floor=20, method="kmeans",
        rule="voronoi", out="runs/p1_x1"):
    require_tests(out)
    if functional not in METRICS or min(n_fit,n_test,d)<1:
        raise ValueError("Invalid functional or data sizes")
    if len(set(M_grid))!=len(M_grid) or len(set(n_cal_grid))!=len(n_cal_grid):
        raise ValueError("Duplicate grid entries")
    rows = []
    for seed in seeds:
        # Same fit/test data and nested calibration pools across n_cal and M.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
        n_max = max(n_cal_grid)
        X, S = sample(n_fit+n_max+n_test, d, rng)
        # Exact fold sizes, seeded permutation; nested cal subsets are never test rows.
        perm = np.random.default_rng(seed).permutation(len(X))
        splits = {"fit": np.sort(perm[:n_fit]), "cal": np.sort(perm[n_fit:n_fit+n_max]),
                  "test": np.sort(perm[n_fit+n_max:])}
        assert_disjoint(splits, n=len(X))
        dataset_hash = array_hash(np.column_stack([X,S]))
        freeze_splits(Path(out)/"splits"/f"seed{seed}.json", splits, dataset_hash)
        Xf, Xt, St = X[splits["fit"]], X[splits["test"]], S[splits["test"]]
        for M in M_grid:
            started = time.perf_counter()
            partition = ExemplarPartition.fit(Xf, M, method, seed, rule)
            write_free_check(partition, Xt[:20], Xt[20:80])
            ct = partition(Xt)
            fitting_seconds = time.perf_counter()-started
            for n_cal in n_cal_grid:
                start = time.perf_counter()
                cal_idx = splits["cal"][:n_cal]
                Xc, Sc = X[cal_idx], S[cal_idx]
                cc = partition(Xc)
                q, counts = per_cell_quantile(cc, Sc, M, alpha, floor)
                probabilities = coverage_cdf(Xt, q[ct])
                within, between, total, means, nt = decompose(probabilities, ct, M, 1-alpha)
                mask = nt > 0
                # CDF-based metrics avoid Bernoulli outcome noise, but retain integration error.
                metrics = functionals(np.where(mask, means, np.nan), nt, alpha)
                # Note 2026-10-02: concentrated undercoverage over finite-threshold cells only;
                # eligibility (infinite test mass <= 1%) is applied at argmin time.
                finite = mask & np.isfinite(q)
                metrics["concentrated_finite"] = (float(max(0., np.max((1-alpha)-means[finite])))
                                                  if finite.any() else None)
                observed, _ = coverage_by_cell(ct, St, q)
                beta = [beta_law_cell_moments(n, alpha, floor) for n in counts]
                weights = nt / nt.sum()
                beta_variance = sum(w*b["variance"] for w,b in zip(weights,beta))
                beta_bias = sum(w*b["bias_squared"] for w,b in zip(weights,beta))
                hashes = {k:split_hash(v) for k,v in splits.items()}
                hashes["cal"] = split_hash(cal_idx)
                row = dict(M=M, n_cal=n_cal, seed=seed, d=d, rule=rule, functional=functional,
                    **metrics, pointwise_msce=total, approx=within,
                    variance=beta_variance, beta_bias_squared=beta_bias,
                    beta_expected_cell_msce=beta_variance+beta_bias,
                    total_gap=total, decomposition_residual=total-within-between,
                    n_below_floor=int(np.sum(counts<floor)),
                    n_infinite=int(np.isinf(q).sum()),
                    test_mass_infinite=float(np.mean(np.isinf(q[ct]))),
                    counts_cal=counts.tolist(), counts_test=nt.tolist(),
                    thresholds=q.tolist(), infinite_threshold_cells=np.flatnonzero(np.isinf(q)).tolist(),
                    cell_coverage_cdf=np.where(mask,means,np.nan).tolist(), cell_coverage_observed=observed.tolist(),
                    fit_seconds=fitting_seconds, calibration_evaluation_seconds=time.perf_counter()-start,
                    manifest=manifest(split_hashes=hashes, M=M, seed=seed, floor=floor, rule=rule,
                        n_below_floor=int(np.sum(counts<floor)), n_fit=n_fit, n_cal=n_cal,
                        n_test=n_test, alpha=alpha, dataset_hash=dataset_hash,
                        partition=dict(partition.metadata), score="scale(X)*Uniform(0,1)",
                        rng_streams={"data":[seed,101],"split":seed,"exemplars":seed},
                        metric_definitions="provisional-v1"))
                rows.append(row)
        print(f"X1 completed seed {seed}", flush=True)
    return rows


def exponent_test(rows, n_boot=1000, seed=2027):
    """Joint seed bootstrap across n_cal, M and all functionals; boundary aware.

    Estimate M* from the seed-mean curve for each n_cal, then its log-log slope.
    Resample whole seed blocks, recompute minima, and use identical draws for
    paired differences. A wide CI containing a target is not an equivalence test.
    """
    seeds = sorted({r["seed"] for r in rows})
    ns = sorted({r["n_cal"] for r in rows})
    ms = sorted({r["M"] for r in rows})
    if len(ns)<2 or len(ms)<3 or len(seeds)<2:
        return {"status":"HOLD", "reason":"Need >=2 seeds, >=2 sizes, >=3 cell counts"}
    cube = np.empty((len(seeds),len(ns),len(ms),len(METRICS)))
    lookup = {(r["seed"],r["n_cal"],r["M"]):r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate seed/n_cal/M; compare arms separately")
    for i,s in enumerate(seeds):
        for j,n in enumerate(ns):
            for k,m in enumerate(ms):
                cube[i,j,k] = [lookup[s,n,m][metric] for metric in METRICS]
    def estimate(sample):
        indices = np.argmin(sample.mean(axis=0), axis=1)
        optima = np.asarray(ms)[indices]
        slopes = np.array([np.polyfit(np.log(ns),np.log(optima[:,f]),1)[0] for f in range(len(METRICS))])
        return slopes, optima, indices
    slopes, optima, indices = estimate(cube)
    rng = np.random.default_rng(seed)
    boots = np.asarray([estimate(cube[rng.integers(len(seeds),size=len(seeds))])[0] for _ in range(n_boot)])
    result = {"status":"DIAGNOSTIC", "n_cal_grid":ns, "M_grid":ms, "n_seeds":len(seeds),
              "bootstrap_replicates":n_boot, "bootstrap_seed":seed, "functionals":{}, "paired_slope_differences":{}}
    for f,name in enumerate(METRICS):
        result["functionals"][name] = {"slope":float(slopes[f]),
            "ci95":np.quantile(boots[:,f],[.025,.975]).tolist(), "Mstar":optima[:,f].tolist(),
            "boundary_minimum":bool(np.any((indices[:,f]==0)|(indices[:,f]==len(ms)-1)))}
    for a in range(len(METRICS)):
        for b in range(a):
            result["paired_slope_differences"][METRICS[a]+" minus "+METRICS[b]] = {
                "estimate":float(slopes[a]-slopes[b]),
                "ci95":np.quantile(boots[:,a]-boots[:,b],[.025,.975]).tolist()}
    return result


def _smoothed_argmin(curve, log_m):
    """Vertex of a quadratic in log M fitted to 5 points around the discrete argmin.

    Returns (log M*, boundary_flag). Boundary when the fit is not convex or the
    vertex leaves the window or reaches the outermost grid points.
    """
    i = int(np.argmin(curve))
    lo = min(max(i-2, 0), max(len(curve)-5, 0))
    window = slice(lo, min(lo+5, len(curve)))
    a, b, _ = np.polyfit(log_m[window], curve[window], 2)
    if a <= 0:
        return log_m[i], True
    vertex = -b/(2*a)
    lo_w, hi_w = log_m[window][0], log_m[window][-1]
    boundary = not (lo_w <= vertex <= hi_w) or vertex <= log_m[0] or vertex >= log_m[-1] \
        or i in (0, len(curve)-1)
    return (float(np.clip(vertex, lo_w, hi_w)), boundary)


def agreement_test(rows, n_boot=1000, seed=2027, max_infinite_mass=0.01):
    """Dated-note (2026-10-02) criterion: mechanism slope s vs 1/(a_hat+1)."""
    seeds = sorted({r["seed"] for r in rows})
    ns = sorted({r["n_cal"] for r in rows})
    ms = sorted({r["M"] for r in rows})
    lookup = {(r["seed"], r["n_cal"], r["M"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate seed/n_cal/M")
    def cube(key):
        return np.array([[[np.nan if lookup[s, n, m][key] is None else lookup[s, n, m][key]
                           for m in ms] for n in ns] for s in seeds], float)
    pw, approx, msce = cube("pointwise_msce"), cube("approx"), cube("msce")
    conc, infmass = cube("concentrated_finite"), cube("test_mass_infinite")
    log_m, log_n = np.log(ms), np.log(ns)

    def estimate(idx):
        mean_pw = np.log(pw[idx].mean(0))
        fits = [_smoothed_argmin(mean_pw[j], log_m) for j in range(len(ns))]
        log_mstar = np.array([f[0] for f in fits]); boundary = np.array([f[1] for f in fits])
        slope = np.polyfit(log_n, log_mstar, 1)[0]
        mstar = np.exp(log_mstar)
        keep = (np.asarray(ms) >= mstar.min()/1.3) & (np.asarray(ms) <= mstar.max()*1.3)
        # Amendment (note, before the v2 run): drop (n_cal, M) with any infinite-set mass.
        ok = keep[None, :] & (infmass[idx].max(0) == 0) & (approx[idx].mean(0) > 0)
        a_hat = -np.polyfit(np.broadcast_to(log_m, ok.shape)[ok], np.log(approx[idx].mean(0)[ok]), 1)[0]
        return slope, a_hat, mstar, boundary

    full = np.arange(len(seeds))
    slope, a_hat, mstar, boundary = estimate(full)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s_, a_, _, b_ = estimate(rng.integers(len(seeds), size=len(seeds)))
        boots.append((s_, a_, s_-1/(a_+1), b_.any()))
    boots = np.array(boots, float)
    diff_ci = np.quantile(boots[:, 2], [.025, .975]).tolist()
    interior_rate = float(1-boots[:, 3].mean())
    msce_argmin = [int(ms[int(np.argmin(msce[:, j].mean(0)))]) for j in range(len(ns))]
    conc_mean, inf_mean = np.nanmean(conc, 0), infmass.mean(0)
    conc_argmin = []
    for j in range(len(ns)):
        eligible = (inf_mean[j] <= max_infinite_mass) & np.isfinite(conc_mean[j])
        conc_argmin.append(int(np.asarray(ms)[eligible][np.argmin(conc_mean[j][eligible])])
                           if eligible.any() else None)
    return {"status": "DIAGNOSTIC", "criterion": "reports/estimand_note_2026-10-02.md",
            "n_cal_grid": ns, "M_grid": ms, "n_seeds": len(seeds), "bootstrap_replicates": n_boot,
            "pointwise": {"Mstar_smoothed": mstar.tolist(), "boundary_flags": boundary.tolist(),
                          "slope": float(slope), "slope_ci95": np.quantile(boots[:, 0], [.025, .975]).tolist(),
                          "a_hat": float(a_hat), "a_hat_ci95": np.quantile(boots[:, 1], [.025, .975]).tolist(),
                          "predicted_slope": float(1/(a_hat+1)),
                          "slope_minus_prediction": float(slope-1/(a_hat+1)),
                          "slope_minus_prediction_ci95": diff_ci,
                          "interior_bootstrap_rate": interior_rate},
            "cell_msce_argmin": msce_argmin,
            "concentrated_finite_argmin_eligible": conc_argmin,
            "descriptive": {"asymptotic_slope": None, "asymptotic_a": None}}


def floor_table(rows, agreement, ceiling):
    """Per n_cal: R_pt minimum vs the floor ceiling (1-t)^2, floor activity at M*."""
    ms = np.asarray(agreement["M_grid"]); table = []
    for j, n in enumerate(agreement["n_cal_grid"]):
        mean = lambda m, k: float(np.mean([r[k] for r in rows if r["n_cal"] == n and r["M"] == m]))
        curve = np.array([mean(m, "pointwise_msce") for m in ms]); i = int(np.argmin(curve))
        active = [int(m) for m in ms if mean(m, "n_below_floor") > 0]
        table.append({"n_cal": n, "Mstar_smoothed": agreement["pointwise"]["Mstar_smoothed"][j],
                      "M_discrete": int(ms[i]), "R_pt_min": float(curve[i]),
                      "R_pt_min_over_ceiling": float(curve[i] / ceiling),
                      "n_below_floor_at_M_discrete": mean(ms[i], "n_below_floor"),
                      "infinite_mass_at_M_discrete": mean(ms[i], "test_mass_infinite"),
                      "first_M_with_floor_activity": active[0] if active else None})
    return table


def gate_decision(checks, ci_width, inconclusive_width):
    """PASS if all hold; INCONCLUSIVE if only the equivalence check fails and its CI is
    too wide to fit the margin (more seeds, never a changed margin); FAIL otherwise."""
    if all(checks.values()):
        return "PASS"
    others = [k for k in checks if k != "mechanism"]
    if all(checks[k] for k in others) and ci_width > inconclusive_width:
        return "INCONCLUSIVE"
    return "FAIL"


def gate_report_v2(rows, agreement, d, margin=0.10, alpha=0.1, min_mstar=3.0):
    # Added 2 Oct after the first v2 run (none of these changes that run's outcome):
    # INCONCLUSIVE outcome, floor-ceiling table, M* < min_mstar treated as a grid-edge boundary.
    residual = max(abs(r["decomposition_residual"]) for r in rows)
    beta_diffs = [np.mean([r["msce"]-r["beta_expected_cell_msce"] for r in rows if r["seed"] == s])
                  for s in sorted({r["seed"] for r in rows})]
    beta_mean = float(np.mean(beta_diffs)); beta_se = float(np.std(beta_diffs, ddof=1)/np.sqrt(len(beta_diffs)))
    pw = agreement["pointwise"]
    ci = pw["slope_minus_prediction_ci95"]; ci_width = ci[1] - ci[0]
    edge = [n for n, m in zip(agreement["n_cal_grid"], pw["Mstar_smoothed"]) if m < min_mstar]
    checks = {
        "interior": pw["interior_bootstrap_rate"] >= 0.90 and not edge,
        "mechanism": -margin <= ci[0] and ci[1] <= margin,
        "b0_argmin_M1": all(m == 1 for m in agreement["cell_msce_argmin"]),
        "b0_beta": abs(beta_mean) <= 3*beta_se,
        "decomposition": residual <= 1e-12}
    agreement["descriptive"] = {"asymptotic_slope": d/(d+2), "asymptotic_a": 2/d}
    status = gate_decision(checks, ci_width, inconclusive_width=2*margin)
    return {"stage": "x1", "status": status,
            "paper_status": "HOLD", "criterion": "reports/estimand_note_2026-10-02.md",
            "checks": checks, "failed": [k for k, v in checks.items() if not v],
            "equivalence_ci": ci, "equivalence_ci_width": ci_width, "inconclusive_width": 2*margin,
            "edge_n_cal_below_min_mstar": edge,
            "floor_table": floor_table(rows, agreement, ceiling=alpha**2),
            "source_sha256": source_hash(),
            "max_decomposition_residual": residual,
            "beta_msce_minus_expectation_seed_mean": beta_mean,
            "beta_msce_minus_expectation_seed_se": beta_se,
            "scope": "PASS supports the decomposition mechanism and an interior optimum; not the asymptotic rate"}


def gate_report(rows, exponents, d):
    residual = max(abs(r["decomposition_residual"]) for r in rows)
    # Missing manuscript functionals and unproved M* laws are substantive blockers.
    reasons = ["Paper definitions of spread/concentrated functional are missing",
               "No interior M* theory is justified for the provisional cell-MSCE functional",
               "Pointwise M* is a surrogate scaling argument, not an established asymptotic equivalence"]
    beta_diffs = []
    for seed in sorted({r["seed"] for r in rows}):
        subset = [r for r in rows if r["seed"]==seed]
        beta_diffs.append(np.mean([r["msce"]-r["beta_expected_cell_msce"] for r in subset]))
    pw = exponents.get("functionals",{}).get("pointwise_msce",{})
    target = d/(d+2)
    if pw.get("boundary_minimum",True):
        reasons.append("Pointwise empirical optimum hits a grid boundary")
    ci = pw.get("ci95",[float("nan"),float("nan")])
    if not ci[0] <= target <= ci[1]:
        reasons.append("Pointwise slope interval does not include the surrogate exponent")
    if residual > 1e-12:
        reasons.append("Orthogonal decomposition failed")
    beta_mean=float(np.mean(beta_diffs))
    beta_se=float(np.std(beta_diffs,ddof=1)/np.sqrt(len(beta_diffs))) if len(beta_diffs)>1 else None
    if beta_se is not None and abs(beta_mean)>3*beta_se:
        reasons.append("Beta expectation discrepancy exceeds three seed standard errors; inspect finite-seed/integration effects")
    return {"stage":"x1", "status":"HOLD", "paper_status":"HOLD", "reasons":reasons,
            "source_sha256":source_hash(), "max_decomposition_residual":residual,
            "beta_msce_minus_expectation_seed_mean":beta_mean,
            "beta_msce_minus_expectation_seed_se":beta_se,
            "surrogate_pointwise_exponent":target,
            "downstream":"Do not execute top-k, X2, level2, or X5 until this gate clears"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/p1_x1")
    p.add_argument("--seeds",type=int,default=8)
    p.add_argument("--d",type=int,default=1)
    p.add_argument("--n-fit",type=int,default=2000)
    p.add_argument("--n-test",type=int,default=12000)
    p.add_argument("--M-grid",type=int,nargs="+",default=[1,2,4,8,16,32,64])
    p.add_argument("--n-cal-grid",type=int,nargs="+",default=[400,800,1600,3200])
    p.add_argument("--floor",type=int,default=20)
    p.add_argument("--protocol",choices=["v1","v2"],default="v1",
                   help="v2: dated-note 2026-10-02 criterion and PASS/FAIL gate")
    p.add_argument("--n-boot",type=int,default=1000)
    p.add_argument("--method",choices=["kmeans","voronoi","uniform"],default="kmeans",
                   help="exemplar selection; voronoi = farthest-first (the lemma's arm)")
    args=p.parse_args()
    validate_first(args.out)
    rows=run(args.M_grid,args.n_cal_grid,args.d,range(args.seeds), n_fit=args.n_fit,
             n_test=args.n_test,floor=args.floor,method=args.method,out=args.out)
    if args.protocol=="v2":
        out=Path(args.out)
        write_json(out/"rows.json",rows)
        write_json(out/"summary.json",summarize(rows,["n_cal","M","rule"],list(METRICS)+["approx","variance","n_below_floor","test_mass_infinite"]))
        agreement=agreement_test(rows,n_boot=args.n_boot)
        gate=gate_report_v2(rows,agreement,args.d)
        write_json(out/"agreement.json",agreement)
        write_json(out/"gate.json",gate)
        print("X1 v2 gate:",gate["status"],gate["checks"])
        return
    out=Path(args.out)
    write_json(out/"rows.json",rows)
    write_json(out/"summary.json",summarize(rows,["n_cal","M","rule"],list(METRICS)+["approx","variance","n_below_floor","test_mass_infinite"]))
    exponents=exponent_test(rows)
    write_json(out/"exponents.json",exponents)
    gate=gate_report(rows,exponents,args.d)
    write_json(out/"gate.json",gate)
    print("X1 gate:",gate["status"],"; ".join(gate["reasons"]))


if __name__ == "__main__":
    main()
