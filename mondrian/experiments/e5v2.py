"""E5 re-analysis (Addendum E of reports/estimand_note_E5_2026-10-02.md).

Tuned forecasters, controls, holdout multicalibration, corrected inference, report
informativeness, cross-procedure transfer and conformal sets with headroom.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from mondrian.conformal import per_cell_quantile
from mondrian.calibration import metrics as calib, NAMES as CAL
from .common import validate_first, require_tests
from .e_llm import logit, label_scores, FLOOR
from .e5_partition import load_tqa, load_popqa, onehot, EPS
from .e2_frozen import MODELS

CS = (0.01, 0.1, 1.0, 10.0, 100.0)
MS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------- tuned logistic
REPORT_WEIGHT = 10.0   # amendment: report column penalised 100x less than group/feature columns


class Logit:
    def __init__(self, C, report_col=None):
        self.C, self.report_col = C, report_col

    def fit(self, X, y, offset=None):
        self.mu, self.sd = X.mean(0), X.std(0); self.sd[self.sd == 0] = 1
        if self.report_col is not None:
            self.sd = self.sd.copy(); self.sd[self.report_col] /= REPORT_WEIGHT
        self.const = None
        if len(np.unique(y)) < 2:
            self.const = float(np.clip(y.mean(), EPS, 1 - EPS)); return self
        self.m = LogisticRegression(C=self.C, max_iter=3000).fit((X - self.mu) / self.sd, y)
        return self

    def predict(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return np.clip(self.m.predict_proba((X - self.mu) / self.sd)[:, 1], EPS, 1 - EPS)


def logloss(p, y):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cv_score(X, y, C, folds=5, seed=0, report_col=None):
    if min(np.bincount(y.astype(int), minlength=2)) < folds:
        return np.inf
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed); tot = 0.0
    for tr, va in skf.split(X, y):
        tot += logloss(Logit(C, report_col).fit(X[tr], y[tr]).predict(X[va]), y[va]) * len(va)
    return tot / len(y)


def tuned(X, y, seed=0, report_col=None):
    """report_col: index of the logit-report column, up-weighted when X also has group/feature columns."""
    if report_col is None and X.shape[1] > 1:
        report_col = 0
    scores = [cv_score(X, y, C, seed=seed, report_col=report_col) for C in CS]
    C = CS[int(np.argmin(scores))] if np.isfinite(min(scores)) else 1.0
    return Logit(C, report_col).fit(X, y), C, float(min(scores))


def multicalibrate(base_fit, Xb_lab, y, groups_lab, Xb_te, groups_te, seed, tau=.01, min_n=10, passes=20):
    """Holdout MC: base fitted on 60% of labelled rows, patches estimated on 40%."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); a, b = idx[:int(.6 * len(y))], idx[int(.6 * len(y)):]
    model, _, _ = base_fit(Xb_lab[a], y[a])
    pb, pt = model.predict(Xb_lab[b]), model.predict(Xb_te)
    for _ in range(passes):
        changed = False
        for gl, gt in zip(groups_lab, groups_te):
            for g in np.unique(gl[b]):
                m = gl[b] == g
                if m.sum() < min_n:
                    continue
                r = float(np.mean(y[b][m] - pb[m]))
                if abs(r) > tau:
                    pb[m] = np.clip(pb[m] + r, EPS, 1 - EPS); pt[gt == g] = np.clip(pt[gt == g] + r, EPS, 1 - EPS)
                    changed = True
        if not changed:
            break
    return pt


def random_cells(ids, M, seed):
    return np.array([int.from_bytes(hashlib.sha256(f"rand|{seed}|{M}|{q}".encode()).digest()[:4], "little") % M for q in ids])


# ---------------------------------------------------------------- one split
def forecast_arms(Z, cells, cat, rand, V, y, lab, te, procs, seed):
    """Brier per arm (averaged over procedures) and selected M for G_cv."""
    out, picked = {}, {}
    for p in procs:
        lv = logit(V[p])
        yl = y[lab].astype(float)
        res = {}
        res["K"] = np.full(len(te), np.clip(yl.mean(), EPS, 1 - EPS))
        mR, CR, _ = tuned(lv[lab], yl, seed); res["R"] = mR.predict(lv[te])
        cv_best = (np.inf, None, None)
        for M in MS[1:]:
            X = np.hstack([lv, onehot(cells[M], M)])
            m, C, sc = tuned(X[lab], yl, seed); res[f"G{M}"] = m.predict(X[te])
            if sc < cv_best[0]:
                cv_best = (sc, M, res[f"G{M}"])
        _, _, scR = tuned(lv[lab], yl, seed)
        if scR <= cv_best[0]:
            cv_best = (scR, 1, res["R"])
        res["Gcv"] = cv_best[2]; picked[p] = cv_best[1]
        for M in (4, 8):
            n = np.bincount(cells[M][lab], weights=yl, minlength=M); c = np.bincount(cells[M][lab], minlength=M)
            rate = np.where(c > 0, n / np.maximum(c, 1), yl.mean())
            res[f"C{M}"] = np.clip(rate[cells[M][te]], EPS, 1 - EPS)
            Xr = np.hstack([lv, onehot(rand[M], M)])
            res[f"Rand{M}"] = tuned(Xr[lab], yl, seed)[0].predict(Xr[te])
        if cat is not None:
            Xc = np.hstack([lv, onehot(cat, cat.max() + 1)])
            res["Cat"] = tuned(Xc[lab], yl, seed)[0].predict(Xc[te])
        XD = np.hstack([lv, Z]); res["D"] = tuned(XD[lab], yl, seed)[0].predict(XD[te])
        q = np.quantile(V[p][lab], [.25, .5, .75])
        gl = [cells[4][lab], np.searchsorted(q, V[p][lab])]; gt = [cells[4][te], np.searchsorted(q, V[p][te])]
        res["MC"] = multicalibrate(lambda X, yy: tuned(X, yy, seed), lv[lab], yl, gl, lv[te], gt, seed)
        res["MC_D"] = multicalibrate(lambda X, yy: tuned(X, yy, seed), XD[lab], yl, gl, XD[te], gt, seed)
        for k, v in res.items():
            out.setdefault(k, []).append((float(np.mean((v - y[te]) ** 2)), logloss(v, y[te].astype(float)), calib(v, y[te])))
    return {k: (float(np.mean([a for a, _, _ in v])), float(np.mean([b for _, b, _ in v])),
                {m: float(np.mean([c[m] for _, _, c in v])) for m in CAL}) for k, v in out.items()}, picked


def transfer(Z, cells, V, y, pool, te, n_lab, ks, seed):
    out = {}
    lab = pool[:n_lab]; yl = y[lab].astype(float)
    lvA = logit(V["fwd"]); M = 4; O = onehot(cells[M], M)
    XA = np.hstack([lvA, O]); gA, _, _ = tuned(XA[lab], yl, seed)
    coef = gA.m.coef_[0] / gA.sd if gA.const is None else np.zeros(XA.shape[1])
    offs = O @ coef[1:]                                   # fixed group offsets from A (standardised back)
    mRA, _, _ = tuned(lvA[lab], yl, seed)
    for B in ("rev", "alt"):
        lvB = logit(V[B])
        for k in ks:
            rows = pool[n_lab:n_lab + k]; yk = y[rows].astype(float)
            if k == 0:
                Xt = np.hstack([lvB, O]); T = gA.predict(Xt[te]); Rk = mRA.predict(lvB[te]); Gk = None
            else:
                # T_k: refit a_B, b_B with the A offsets as a fixed offset (logistic with offset via glm-free trick)
                T = fit_offset(lvB[rows][:, 0], offs[rows], yk, lvB[te][:, 0], offs[te])
                Rk = tuned(lvB[rows], yk, seed)[0].predict(lvB[te]) if len(np.unique(yk)) > 1 else np.full(len(te), yk.mean())
                XB = np.hstack([lvB, O])
                Gk = tuned(XB[rows], yk, seed)[0].predict(XB[te]) if len(np.unique(yk)) > 1 else np.full(len(te), yk.mean())
            br = lambda p: float(np.mean((np.clip(p, EPS, 1 - EPS) - y[te]) ** 2))
            out[f"{B}|n{n_lab}|k{k}"] = {"T": br(T), "R": br(Rk), "G": br(Gk) if Gk is not None else None}
    return out


def fit_offset(x, off, y, x_te, off_te, ridge=1e-2, iters=50):
    """Two-parameter logistic regression a + b x with a fixed offset (Newton, small ridge)."""
    w = np.zeros(2); X = np.column_stack([np.ones_like(x), x]); Xt = np.column_stack([np.ones_like(x_te), x_te])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + off))); g = X.T @ (p - y) + ridge * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(2); w -= np.linalg.solve(H, g)
    return np.clip(1 / (1 + np.exp(-(Xt @ w + off_te))), EPS, 1 - EPS)


def conformal(Z, cells, V, y, lab, te, procs, seed, alphas=(.1, .2, .3)):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 55])); h = rng.permutation(len(lab))
    a, b = lab[h[:len(lab) // 2]], lab[h[len(lab) // 2:]]
    out = {}
    for p in procs:
        lv = logit(V[p]); O = onehot(cells[4], 4); XG = np.hstack([lv, O]); XD = np.hstack([lv, Z])
        fR = tuned(lv[a], y[a].astype(float), seed)[0]; fG = tuned(XG[a], y[a].astype(float), seed)[0]
        fD = tuned(XD[a], y[a].astype(float), seed)[0]
        arms = {"R_unstrat": (fR.predict(lv[b]), fR.predict(lv[te]), np.zeros(len(b), int), np.zeros(len(te), int), 1),
                "G4_mondrian": (fG.predict(XG[b]), fG.predict(XG[te]), cells[4][b], cells[4][te], 4),
                "D_unstrat": (fD.predict(XD[b]), fD.predict(XD[te]), np.zeros(len(b), int), np.zeros(len(te), int), 1)}
        for alpha in alphas:
            for name, (pb, pt, cb, ct, M) in arms.items():
                q, counts = per_cell_quantile(cb, label_scores(pb, y[b]), M, alpha, FLOOR)
                thr = q[ct]; hit = label_scores(pt, y[te]) <= thr
                s1 = label_scores(pt, 1) <= thr; s0 = label_scores(pt, 0) <= thr; size = s1.astype(int) + s0
                single = size == 1
                out.setdefault(f"{name}|a{alpha}", []).append(
                    (hit.mean(), size.mean(), single.mean(), hit[single].mean() if single.any() else np.nan,
                     float(np.mean(np.isinf(thr)))))
    return {k: [float(np.nanmean([x[i] for x in v])) for i in range(5)] for k, v in out.items()}


# ---------------------------------------------------------------- inference
def nb_interval(d, n_test, n_train):
    d = np.asarray(d, float); J = len(d); sd = d.std(ddof=1)
    se = sd * np.sqrt(1 / J + n_test / n_train); t = stats.t.ppf(.975, J - 1)
    tstat = d.mean() / se if se > 0 else 0.0
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - t * se), float(d.mean() + t * se)],
            "win_rate": float(np.mean(d < 0)), "p_one_sided_less": float(stats.t.cdf(tstat, J - 1)),
            "naive_ci95": [float(d.mean() - 1.96 * sd / np.sqrt(J)), float(d.mean() + 1.96 * sd / np.sqrt(J))]}


def holm(pvals, alpha=.05):
    order = np.argsort(pvals); m = len(pvals); rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if pvals[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej.tolist()


# ---------------------------------------------------------------- informativeness
def informativeness(V, y):
    out = {}
    for p, v in V.items():
        q = np.quantile(v, np.linspace(0, 1, 11)[1:-1]); b = np.searchsorted(q, v)
        ybar = y.mean(); res = sum((np.mean(b == j)) * (y[b == j].mean() - ybar) ** 2 for j in np.unique(b))
        out[p] = {"auroc": float(roc_auc_score(y, v)), "brier_raw": float(np.mean((v - y) ** 2)),
                  "resolution": float(res), "uncertainty": float(ybar * (1 - ybar))}
    return out


# ---------------------------------------------------------------- driver
def run_dataset(name, H, y, V, ids, cat, seeds, budgets, split_fn, do_transfer):
    procs = list(V); rows, picks, trans, conf = [], [], [], []
    for seed in seeds:
        f, pool, te = split_fn(seed)
        pca = PCA(16, random_state=0).fit(H[f]); Z = pca.transform(H)
        cells = {M: ExemplarPartition.fit(Z[f], M, "voronoi", seed)(Z) for M in MS[1:]}
        rand = {M: random_cells(ids, M, seed) for M in (4, 8)}
        for n in budgets:
            lab = pool[:n]
            res, picked = forecast_arms(Z, cells, cat, rand, V, y, lab, te, procs, seed)
            for k, (b, ll, cm) in res.items():
                rows.append({"seed": seed, "n_lab": n, "method": k, "brier": b, "logloss": ll, **cm})
            picks.append({"seed": seed, "n_lab": n, **{f"M_{p}": m for p, m in picked.items()}})
        if do_transfer:
            tr = {"seed": seed}
            for n in (120, 180):
                tr.update(transfer(Z, cells, V, y, pool, te, n, (0, 10, 30, 60), seed))
            trans.append(tr)
        conf.append(conformal(Z, cells, V, y, pool[:max(budgets)], te, procs, seed))
        if seed % 20 == 0:
            print(f"  {name}: seed {seed}", flush=True)
    return rows, picks, trans, conf


def summarise(rows, picks, trans, conf, budgets, n_test):
    seeds = sorted({r["seed"] for r in rows}); methods = sorted({r["method"] for r in rows})
    lk = {(r["seed"], r["n_lab"], r["method"]): r for r in rows}
    S = {"brier": {}, "skill_vs_K": {}, "logloss": {}, "contrasts": {}, "selected_M": {}}
    for n in budgets:
        S["brier"][n] = {m: float(np.mean([lk[s, n, m]["brier"] for s in seeds])) for m in methods}
        S["logloss"][n] = {m: float(np.mean([lk[s, n, m]["logloss"] for s in seeds])) for m in methods}
        S["skill_vs_K"][n] = {m: 1 - S["brier"][n][m] / S["brier"][n]["K"] for m in methods}
        S["contrasts"][n] = {}
        for a, b in [("G4", "R"), ("G8", "R"), ("Gcv", "R"), ("C4", "R"), ("Rand4", "R"), ("Rand8", "R"), ("Cat", "R"),
                     ("MC", "R"), ("MC_D", "D"), ("D", "R"), ("G4", "D"), ("Gcv", "D"), ("R", "K"), ("G4", "Rand4"),
                     ("G4", "MC"), ("G4", "Cat")]:
            if a in methods and b in methods:
                d = [lk[s, n, a]["brier"] - lk[s, n, b]["brier"] for s in seeds]
                S["contrasts"][n][f"{a}-{b}"] = nb_interval(d, n_test, n)
        # calibration-toolbox metrics (mondrian/calibration.py); sce = binary ECE of the forecast
        S.setdefault("calibration", {})[n] = {m: {c: float(np.mean([lk[s, n, m][c] for s in seeds])) for c in CAL} for m in methods}
        S.setdefault("calibration_contrasts", {})[n] = {
            f"{a}-{b}|{c}": nb_interval([lk[s, n, a][c] - lk[s, n, b][c] for s in seeds], n_test, n)
            for a, b in [("G4", "R"), ("Gcv", "R"), ("MC", "R"), ("D", "R"), ("G4", "D")] if a in methods and b in methods
            for c in ("sce", "ace")}
        ms = [v for pk in picks if pk["n_lab"] == n for k, v in pk.items() if k.startswith("M_")]
        S["selected_M"][n] = {str(M): float(np.mean(np.array(ms) == M)) for M in MS}
    if trans:
        T = {}
        for key in trans[0]:
            if key == "seed":
                continue
            d_TR = [t[key]["T"] - t[key]["R"] for t in trans]
            entry = {"T": float(np.mean([t[key]["T"] for t in trans])), "R": float(np.mean([t[key]["R"] for t in trans])),
                     "T-R": nb_interval(d_TR, n_test, int(key.split("|n")[1].split("|")[0]))}
            if trans[0][key]["G"] is not None:
                entry["G"] = float(np.mean([t[key]["G"] for t in trans]))
                entry["T-G"] = nb_interval([t[key]["T"] - t[key]["G"] for t in trans], n_test, int(key.split("|n")[1].split("|")[0]))
            T[key] = entry
        S["transfer"] = T
    S["conformal"] = {k: dict(zip(["coverage", "set_size", "singleton_rate", "singleton_coverage", "infinite_mass"],
                                  [float(np.nanmean([c[k][i] for c in conf])) for i in range(5)])) for k in conf[0]}
    for alpha in (.1, .2, .3):
        a = [c[f"G4_mondrian|a{alpha}"][1] - c[f"R_unstrat|a{alpha}"][1] for c in conf]
        S["conformal"][f"size_G4-R|a{alpha}"] = nb_interval(a, n_test, max(budgets) // 2)
        a = [c[f"D_unstrat|a{alpha}"][1] - c[f"R_unstrat|a{alpha}"][1] for c in conf]
        S["conformal"][f"size_D-R|a{alpha}"] = nb_interval(a, n_test, max(budgets) // 2)
    return S


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--out", default="runs/p1_e5v2")
    pa.add_argument("--tqa-seeds", type=int, default=200)
    pa.add_argument("--popqa-seeds", type=int, default=100)
    a = pa.parse_args(); validate_first(a.out); require_tests(a.out)
    out = Path(a.out); result = {"criterion": "reports/estimand_note_E5_2026-10-02.md#addendum-e", "paper_status": "HOLD"}
    cats = {json.loads(l)["question_id"]: json.loads(l)["audit_group"] for l in open("inputs/truthfulqa_llama/rows.jsonl")}
    primary = []
    for model in MODELS:
        H, y, V = load_tqa(model)
        import torch
        ids = np.asarray(torch.load(f"{__import__('experiments.e5_partition', fromlist=['HSTATES']).HSTATES}/{model}/truthfulqa/hidden_states.pt",
                                    map_location="cpu", weights_only=False)["example_ids"]).astype(str)
        vocab = {c: i for i, c in enumerate(sorted(set(cats.values())))}; cat = np.array([vocab[cats[q]] for q in ids])
        def split(seed, n=len(y)):
            perm = np.random.default_rng(np.random.SeedSequence([seed, 2525])).permutation(n)
            f, c, te = np.array_split(perm, 3); return f, c, te
        rows, picks, trans, conf = run_dataset(model, H, y, V, ids, cat, range(a.tqa_seeds), (60, 120, 180, 272), split, True)
        write_json(out / f"rows_tqa_{model}.json", {"rows": rows, "picks": picks, "transfer": trans, "conformal": conf})
        S = summarise(rows, picks, trans, conf, (60, 120, 180, 272), 272)
        S["informativeness"] = informativeness(V, y.astype(float))
        result[f"tqa_{model}"] = S
        for n in (60, 120, 180, 272):
            primary.append((model, n, S["contrasts"][n]["G4-R"]["p_one_sided_less"]))
        print("done", model, flush=True)
    rej = holm([p for *_, p in primary])
    result["primary_family_holm"] = [{"model": m, "n_lab": n, "p": p, "reject": r} for (m, n, p), r in zip(primary, rej)]
    X, y, V, sp = load_popqa()
    rows_p = [json.loads(l) for l in open("inputs/popqa_llama/rows.jsonl")]; ids = np.array([r["question_id"] for r in rows_p])
    f_idx = np.flatnonzero(sp == "fit"); pool = np.flatnonzero(sp != "fit")
    def split_p(seed):
        perm = pool[np.random.default_rng(np.random.SeedSequence([seed, 2526])).permutation(len(pool))]
        return f_idx, perm[2000:], perm[:2000]
    budgets = (100, 200, 500, 1000, 2720)
    rows, picks, trans, conf = run_dataset("popqa", X, y, V, ids, None, range(a.popqa_seeds), budgets, split_p, False)
    write_json(out / "rows_popqa.json", {"rows": rows, "picks": picks, "conformal": conf})
    S = summarise(rows, picks, [], conf, budgets, 2000)
    te_all = np.concatenate([split_p(0)[2], split_p(0)[1]])
    S["informativeness"] = informativeness({"probe": V["probe"][pool]}, y[pool].astype(float))
    result["popqa"] = S; result["source_sha256"] = source_hash()
    write_json(out / "e5v2_result.json", result)
    print(json.dumps(result["primary_family_holm"], indent=0))


if __name__ == "__main__":
    main()
