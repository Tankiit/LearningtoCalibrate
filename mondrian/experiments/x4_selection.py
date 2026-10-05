"""X4: stage-indexed selection counterexample (Claim M), one synthetic DGP.

Unit: X ~ U[0,1]; latent answerability a(X) = sigmoid(c + 4(X - 1/2)).
Two independent decodes: the shown answer (decode 1) is correct with Y = 1{u1 < a(X)};
a separate confidence call re-samples (decode 2) with Y' = 1{u2 < a(X)}.
Score of the shown answer: S = Y*U + (1-Y)*(1+U), U ~ U[0,1] (Proposition M-fail).
Confidence signals (interval bins, cutpoints = D_fit quantiles, frozen):
  V_pre        = a(X) + N(0, 0.1^2)                  stage 0
  V_post       = 0.6*Y  + 0.4*a(X) + N(0, 0.1^2)     stage 1, same decode as the score
  V_post_fresh = 0.6*Y' + 0.4*a(X) + N(0, 0.1^2)     stage 1, different decode
Arms:
  marginal              one cell
  pre                   V_pre bins, calibration and test alike
  post_matched          V_post bins, calibration and test alike
  post_fresh_consistent V_post_fresh bins, calibration and test alike (still valid)
  post_mismatched       calibrate on V_post, assign test cells by V_post_fresh
  answerable            calibrate on {Y = 1} only, apply to the full test stream
"""
import argparse
from pathlib import Path
import numpy as np
from mondrian.audit import write_free_check, write_json, manifest, array_hash, source_hash
from mondrian.conformal import per_cell_quantile, coverage_by_cell
from mondrian.splits import make_splits, freeze_splits, split_hash
from .common import validate_first, require_tests, summarize

ARMS = ("marginal", "pre", "post_matched", "post_fresh_consistent", "post_mismatched", "answerable")
VALID_ARMS = ("marginal", "pre", "post_matched", "post_fresh_consistent")


def sample(n, c, rng):
    X = rng.uniform(size=n)
    a = 1 / (1 + np.exp(-(c + 4 * (X - .5))))
    u1, u2, U = rng.uniform(size=(3, n))
    Y, Y2 = (u1 < a).astype(int), (u2 < a).astype(int)
    S = Y * U + (1 - Y) * (1 + U)
    noise = rng.normal(0, .1, size=(3, n))
    V = {"pre": a + noise[0], "post": .6 * Y + .4 * a + noise[1],
         "post_fresh": .6 * Y2 + .4 * a + noise[2]}
    return X, a, Y, S, V


class IntervalBins:
    """Frozen interval binning of a scalar signal; cutpoints from D_fit quantiles."""

    def __init__(self, v_fit, M):
        self.cuts = np.quantile(v_fit, np.arange(1, M) / M)
        self.cuts.setflags(write=False)
        self.M = M

    def __call__(self, v):
        return np.searchsorted(self.cuts, np.asarray(v).ravel(), side="right")


def run(out="runs/p1_x4", seeds=range(50), cs=(-1.5, -.6, 0., .9, 2.2, 5.), M=5,
        n=24000, alpha=.1, floor=20):
    require_tests(out)
    t, rows = 1 - alpha, []
    for c in cs:
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 404, int(round(1000 * c)) + 10000]))
            X, a, Y, S, V = sample(n, c, rng)
            splits = make_splits(n, fracs=(1 / 12, 1 / 12, 10 / 12), seed=seed)
            dataset_hash = array_hash(np.column_stack([X, Y, S] + list(V.values())))
            freeze_splits(Path(out) / "splits" / f"c{c:+.1f}_seed{seed}.json", splits, dataset_hash)
            f, cal, te = (splits[k] for k in ("fit", "cal", "test"))
            bins = {k: IntervalBins(V[k][f], M) for k in V}
            for b in bins.values():
                write_free_check(b, V["pre"][te][:20, None], V["pre"][te][20:80, None])
            for arm in ARMS:
                if arm == "marginal":
                    cc, ct, m, cal_rows = np.zeros(len(cal), int), np.zeros(len(te), int), 1, cal
                elif arm == "answerable":
                    cal_rows = cal[Y[cal] == 1]
                    cc, ct, m = np.zeros(len(cal_rows), int), np.zeros(len(te), int), 1
                else:
                    key_cal = {"pre": "pre", "post_matched": "post", "post_fresh_consistent": "post_fresh",
                               "post_mismatched": "post"}[arm]
                    key_test = "post_fresh" if arm == "post_mismatched" else key_cal
                    cal_rows, m = cal, M
                    cc, ct = bins[key_cal](V[key_cal][cal]), bins[key_test](V[key_test][te])
                q, counts = per_cell_quantile(cc, S[cal_rows], m, alpha, floor)
                cov, nt = coverage_by_cell(ct, S[te], q)
                observed = nt > 0
                hit = S[te] <= q[ct]
                rows.append(dict(arm=arm, c=c, seed=seed, p_answerable=float(a.mean()),
                    coverage=float(hit.mean()),
                    worst_cell_coverage=float(np.min(cov[observed])),
                    coverage_given_Y1=float(hit[Y[te] == 1].mean()),
                    coverage_given_Y0=float(hit[Y[te] == 0].mean()) if (Y[te] == 0).any() else None,
                    cell_coverages=cov.tolist(), counts_cal=counts.tolist(), counts_test=nt.tolist(),
                    thresholds=q.tolist(), n_below_floor=int(np.sum(counts < floor)),
                    test_mass_infinite=float(np.mean(np.isinf(q[ct]))),
                    manifest=manifest(split_hashes={k: split_hash(v) for k, v in splits.items()},
                        M=m, seed=seed, floor=floor, rule=arm, n_below_floor=int(np.sum(counts < floor)),
                        alpha=alpha, c=c, dataset_hash=dataset_hash, n_cal_used=len(cal_rows),
                        cutpoints={k: b.cuts.tolist() for k, b in bins.items()})))
        print(f"X4 completed c={c:+.1f}", flush=True)
    return rows


def gate(rows, alpha=.1, tol=.01):
    """Diagnostic PASS: valid arms cover marginally and per cell; the answerable arm
    matches its closed form p * P(cover | Y=1); the mismatched arm breaks per-cell coverage."""
    t, checks, detail = 1 - alpha, {}, {}
    cs = sorted({r["c"] for r in rows})
    for arm in ARMS:
        detail[arm] = {}
        for c in cs:
            sub = [r for r in rows if r["arm"] == arm and r["c"] == c]
            cov = np.array([r["coverage"] for r in sub])
            worst = np.array([r["worst_cell_coverage"] for r in sub])
            detail[arm][f"{c:+.1f}"] = {"p": float(np.mean([r["p_answerable"] for r in sub])),
                "coverage_mean": float(cov.mean()), "coverage_se": float(cov.std(ddof=1) / np.sqrt(len(cov))),
                "worst_cell_mean": float(worst.mean()), "worst_cell_se": float(worst.std(ddof=1) / np.sqrt(len(worst))),
                "coverage_given_Y1_mean": float(np.mean([r["coverage_given_Y1"] for r in sub]))}
    checks["valid_arms_marginal"] = all(detail[a][k]["coverage_mean"] >= t - 3 * detail[a][k]["coverage_se"]
                                        for a in VALID_ARMS for k in detail[a])
    # Per-cell validity holds in expectation over calibration draws. Criterion fixed after a
    # 4-seed smoke run, before the full run: every cell's seed-mean coverage >= t - 4 SE
    # (about 120 one-sided comparisons); the mismatched arm must have a cell below t - max(4 SE, tol).
    per_cell, slack = {}, {}
    for a in VALID_ARMS + ("post_mismatched",):
        mins, worst_z = [], []
        for c in cs:
            cells = np.array([r["cell_coverages"] for r in rows if r["arm"] == a and r["c"] == c], float)
            mean = np.nanmean(cells, axis=0)
            se = np.nanstd(cells, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(cells), axis=0))
            se = np.where(se > 0, se, 1e-12)
            mins.append(float(np.nanmin(mean)))
            worst_z.append(float(np.nanmin((mean - t) / se)))
        per_cell[a], slack[a] = mins, worst_z
    checks["valid_arms_per_cell_in_expectation"] = all(min(slack[a]) >= -4 for a in VALID_ARMS)
    checks["mismatched_breaks_per_cell"] = (min(slack["post_mismatched"]) < -4
                                            and min(per_cell["post_mismatched"]) < t - tol)
    closed = {k: v["p"] * v["coverage_given_Y1_mean"] for k, v in detail["answerable"].items()}
    checks["answerable_matches_closed_form"] = all(abs(detail["answerable"][k]["coverage_mean"] - closed[k]) <= tol
                                                   for k in closed)
    return {"stage": "x4", "status": "PASS" if all(checks.values()) else "FAIL", "paper_status": "HOLD",
            "checks": checks, "per_cell_min_of_mean_coverage": per_cell, "per_cell_min_z": slack, "detail": detail,
            "answerable_closed_form": closed, "source_sha256": source_hash(),
            "scope": "implementation diagnostic of Propositions M-suff / M-fail; not an LLM result"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/p1_x4")
    p.add_argument("--seeds", type=int, default=50)
    a = p.parse_args()
    validate_first(a.out)
    rows = run(a.out, seeds=range(a.seeds))
    out = Path(a.out)
    write_json(out / "rows.json", rows)
    write_json(out / "summary.json", summarize(rows, ["arm", "c"],
               ["coverage", "worst_cell_coverage", "coverage_given_Y1", "test_mass_infinite"]))
    g = gate(rows)
    write_json(out / "gate.json", g)
    print("X4 gate:", g["status"], g["checks"])


if __name__ == "__main__":
    main()
