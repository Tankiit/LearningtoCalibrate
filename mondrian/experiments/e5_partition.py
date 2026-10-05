"""E5 (reports/estimand_note_E5_2026-10-02.md): learned groups from a frozen representation
x elicitation procedure, against recalibration, multicalibration and direct predictors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from mondrian.conformal import per_cell_quantile
from .common import validate_first, require_tests
from .e_llm import logit, label_scores, ALPHA, FLOOR
from .e2_frozen import RUN1, MODELS, VARIANTS

HSTATES = "/Users/tanmoy/research/OOD_detection/OVA_ARR/outputs/step1_extract_variants/token_mode_mean"
MS = (1, 2, 4, 8, 16)
EPS = 1e-4


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def load_tqa(model):
    import torch
    h = torch.load(f"{HSTATES}/{model}/truthfulqa/hidden_states.pt", map_location="cpu", weights_only=False)
    ids = np.asarray(h["example_ids"]).astype(str)
    pick_pos = np.array([hashlib.sha256(("p1-e2f|" + q).encode()).digest()[0] & 1 == 0 for q in ids])
    H = np.where(pick_pos[:, None], h["h_pos"].double().numpy(), h["h_neg"].double().numpy())
    V = {}
    for k, rel in VARIANTS.items():
        d = torch.load(f"{RUN1}/{model}/{rel}.pt", map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(d["example_ids"]).astype(str), ids):
            raise ValueError("ids differ")
        V[k] = np.where(pick_pos, np.asarray(d["V_pos"], float), np.asarray(d["V_neg"], float))
    return unit(H), pick_pos.astype(int), V


def load_popqa(root="inputs/popqa_llama"):
    rows = [json.loads(l) for l in (Path(root) / "rows.jsonl").open()]
    X = unit(np.load(Path(root) / "states.npz")["states"].astype(np.float64))
    sp = np.array([r["split"] for r in rows]); y = np.array([r["correct"] for r in rows])
    return X, y, {"probe": np.array([r["anchor_p"] for r in rows])}, sp


def onehot(c, M):
    return np.eye(M)[c]


def fit_logit(Xl, yl):
    mu, sd = Xl.mean(0), Xl.std(0); sd[sd == 0] = 1
    if len(np.unique(yl)) < 2:
        const = float(np.clip(yl.mean(), EPS, 1 - EPS))
        return lambda X: np.full(len(X), const)
    m = LogisticRegression(C=1.0, max_iter=2000).fit((Xl - mu) / sd, yl)
    return lambda X: np.clip(m.predict_proba((X - mu) / sd)[:, 1], EPS, 1 - EPS)


def multicalibrate(p_lab, y_lab, g_lab, p_test, g_test, tau=.02, min_n=10, passes=50):
    p_lab, p_test = p_lab.copy(), p_test.copy()
    for _ in range(passes):
        changed = False
        for g in np.unique(g_lab):
            m = g_lab == g
            if m.sum() < min_n:
                continue
            r = float(np.mean(y_lab[m] - p_lab[m]))
            if abs(r) > tau:
                p_lab[m] = np.clip(p_lab[m] + r, EPS, 1 - EPS); p_test[g_test == g] = np.clip(p_test[g_test == g] + r, EPS, 1 - EPS)
                changed = True
        if not changed:
            break
    return p_test


def forecasters(Z_lab, Z_te, cells_lab, cells_te, V_lab, V_te, y_lab, procs):
    """Returns {name: {proc: p_test}}. cells_* dict M -> labels."""
    out = {}
    lv = {p: logit(V_lab[p]) for p in procs}; lt = {p: logit(V_te[p]) for p in procs}
    for M in MS:
        out[f"G{M}"] = {p: fit_logit(np.hstack([lv[p], onehot(cells_lab[M], M)]), y_lab)(
            np.hstack([lt[p], onehot(cells_te[M], M)])) for p in procs}
        n = np.bincount(cells_lab[M], weights=y_lab, minlength=M); c = np.bincount(cells_lab[M], minlength=M)
        rate = np.where(c > 0, n / np.maximum(c, 1), y_lab.mean())
        out[f"C{M}"] = {p: np.clip(rate[cells_te[M]], EPS, 1 - EPS) for p in procs}
        if len(procs) > 1:
            P = len(procs)
            def stack(lvv, cells, n_):
                blocks = []
                for i, p in enumerate(procs):
                    pv = np.zeros((n_, P)); pv[:, i] = lvv[p][:, 0]
                    pd = np.zeros((n_, P - 1))
                    if i > 0: pd[:, i - 1] = 1
                    blocks.append(np.hstack([pv, pd, onehot(cells, M)]))
                return blocks
            bl = stack(lv, cells_lab[M], len(y_lab)); bt = stack(lt, cells_te[M], len(cells_te[M]))
            f = fit_logit(np.vstack(bl), np.tile(y_lab, P))
            out[f"GS{M}"] = {p: f(bt[i]) for i, p in enumerate(procs)}
    out["R"] = out["G1"]
    # multicalibration over 8 cells x V-quartiles, from R
    out["MC"] = {}
    for p in procs:
        q = np.quantile(V_lab[p], [.25, .5, .75])
        g_l = cells_lab[8] * 4 + np.searchsorted(q, V_lab[p]); g_t = cells_te[8] * 4 + np.searchsorted(q, V_te[p])
        p_lab_R = fit_logit(lv[p], y_lab)(lv[p])
        out["MC"][p] = multicalibrate(p_lab_R, y_lab, g_l, out["R"][p], g_t)
    out["D"] = {p: fit_logit(np.hstack([Z_lab, lv[p]]), y_lab)(np.hstack([Z_te, lt[p]])) for p in procs}
    return out


def conformal_eval(Z_fit, Z_a, Z_b, cells_a, cells_b, V_a, V_b, y_a, y_b, procs, y_te, Z_te, cells_te, V_te, M=4):
    """Forecasters fitted on half a, Mondrian thresholds on half b, evaluated on test."""
    fa = forecasters(Z_a, np.vstack([Z_b, Z_te]), cells_a, {m: np.concatenate([cells_b[m], cells_te[m]]) for m in MS},
                     V_a, {p: np.concatenate([V_b[p], V_te[p]]) for p in procs}, y_a, procs)
    nb = len(y_b); res = {}
    for name in ("R", "G4", "MC", "D"):
        per = []
        for p in procs:
            pb, pt = fa[name][p][:nb], fa[name][p][nb:]
            q, counts = per_cell_quantile(cells_b[M], label_scores(pb, y_b), M, ALPHA, FLOOR)
            ct = cells_te[M]; thr = q[ct]
            hit = label_scores(pt, y_te) <= thr
            size = (label_scores(pt, 1) <= thr).astype(int) + (label_scores(pt, 0) <= thr).astype(int)
            nt = np.bincount(ct, minlength=M)
            cov = np.divide(np.bincount(ct, weights=hit, minlength=M), nt, out=np.full(M, np.nan), where=nt > 0)
            per.append((hit.mean(), np.nanmin(cov), size.mean(), cov))
        res[name] = {"coverage": float(np.mean([x[0] for x in per])), "worst_cell": float(np.mean([x[1] for x in per])),
                     "set_size": float(np.mean([x[2] for x in per])),
                     "cell_cov": np.nanmean(np.array([x[3] for x in per]), 0).tolist()}
    return res


def run_dataset(H, y, V, procs, seeds, budgets, split_fn, pca_dim=16):
    rows, conf = [], []
    for seed in seeds:
        f, lab_pool, te = split_fn(seed)
        pca = PCA(pca_dim, random_state=0).fit(H[f]); Z = pca.transform(H)
        parts = {M: ExemplarPartition.fit(Z[f], M, "voronoi", seed) for M in MS}
        cells = {M: parts[M](Z) for M in MS}
        for n in budgets:
            lab = lab_pool[:n]
            fc = forecasters(Z[lab], Z[te], {M: cells[M][lab] for M in MS}, {M: cells[M][te] for M in MS},
                             {p: V[p][lab] for p in procs}, {p: V[p][te] for p in procs}, y[lab], procs)
            for name, d in fc.items():
                b = np.mean([np.mean((d[p] - y[te]) ** 2) for p in procs])
                ll = np.mean([-np.mean(y[te] * np.log(d[p]) + (1 - y[te]) * np.log(1 - d[p])) for p in procs])
                rows.append({"seed": seed, "n_lab": n, "method": name, "brier": float(b), "logloss": float(ll)})
        lab = lab_pool[:max(budgets)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 55])); half = rng.permutation(len(lab))
        a, b_ = lab[half[: len(lab) // 2]], lab[half[len(lab) // 2:]]
        conf.append(conformal_eval(Z[f], Z[a], Z[b_], {M: cells[M][a] for M in MS}, {M: cells[M][b_] for M in MS},
                                   {p: V[p][a] for p in procs}, {p: V[p][b_] for p in procs}, y[a], y[b_], procs,
                                   y[te], Z[te], {M: cells[M][te] for M in MS}, {p: V[p][te] for p in procs}))
    return rows, conf


def summarise(rows, conf, budgets):
    methods = sorted({r["method"] for r in rows}); seeds = sorted({r["seed"] for r in rows})
    lk = {(r["seed"], r["n_lab"], r["method"]): r["brier"] for r in rows}
    out = {"brier_mean": {}, "vs_R": {}, "Q": {}}
    ci = lambda x: [float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))), float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x)))]
    for n in budgets:
        out["brier_mean"][str(n)] = {m: float(np.mean([lk[s, n, m] for s in seeds])) for m in methods}
        out["vs_R"][str(n)] = {m: {"diff": float(np.mean(d := np.array([lk[s, n, m] - lk[s, n, "R"] for s in seeds]))), "ci95": ci(d)}
                               for m in methods if m != "R"}
        gm = {M: out["brier_mean"][str(n)][f"G{M}"] for M in MS}
        bestM = min((M for M in MS if M >= 2), key=lambda M: gm[M])
        out["Q"][str(n)] = {"argmin_M_over_G": int(min(MS, key=lambda M: gm[M])), "best_G_M>=2": bestM,
                            "best_G_beats_R": out["vs_R"][str(n)][f"G{bestM}"]["ci95"][1] < 0,
                            "G4_beats_R": out["vs_R"][str(n)]["G4"]["ci95"][1] < 0,
                            "all_G_not_better": all(out["vs_R"][str(n)][f"G{M}"]["ci95"][1] >= 0 for M in MS if M >= 2)}
    nmax = max(budgets)
    out["Q4"] = {k: {"diff": float(np.mean(d := np.array([lk[s, nmax, "G4"] - lk[s, nmax, k] for s in seeds]))), "ci95": ci(d)} for k in ("MC", "D")}
    if "GS4" in methods:
        out["Q5"] = {"diff_GS4_minus_G4": float(np.mean(d := np.array([lk[s, nmax, "GS4"] - lk[s, nmax, "G4"] for s in seeds]))), "ci95": ci(d)}
    out["conformal"] = {name: {k: float(np.mean([c[name][k] for c in conf])) for k in ("coverage", "worst_cell", "set_size")}
                        for name in conf[0]}
    for name in conf[0]:
        cov = np.array([c[name]["coverage"] for c in conf]); out["conformal"][name]["coverage_se"] = float(cov.std(ddof=1) / np.sqrt(len(cov)))
        cells = np.array([c[name]["cell_cov"] for c in conf]); out["conformal"][name]["cell_mean_min"] = float(np.nanmin(np.nanmean(cells, 0)))
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--out", default="runs/p1_e5")
    a = pa.parse_args(); validate_first(a.out); require_tests(a.out)
    result = {"criterion": "reports/estimand_note_E5_2026-10-02.md", "paper_status": "HOLD"}
    for model in MODELS:
        H, y, V = load_tqa(model); n = len(y)
        def split(seed, n=n):
            perm = np.random.default_rng(np.random.SeedSequence([seed, 2525])).permutation(n)
            f, c, te = np.array_split(perm, 3); return f, c, te
        rows, conf = run_dataset(H, y, V, list(V), range(200), (60, 120, 180, 272), split)
        write_json(Path(a.out) / f"rows_tqa_{model}.json", rows)
        result[f"tqa_{model}"] = summarise(rows, conf, (60, 120, 180, 272))
        print("done", model, flush=True)
    X, y, V, sp = load_popqa()
    f_idx = np.flatnonzero(sp == "fit"); pool = np.flatnonzero(sp != "fit")
    def split_p(seed):
        perm = pool[np.random.default_rng(np.random.SeedSequence([seed, 2526])).permutation(len(pool))]
        return f_idx, perm[2000:], perm[:2000]
    rows, conf = run_dataset(X, y, V, ["probe"], range(100), (100, 200, 500, 1000, 2720), split_p)
    write_json(Path(a.out) / "rows_popqa.json", rows)
    result["popqa"] = summarise(rows, conf, (100, 200, 500, 1000, 2720))
    result["source_sha256"] = source_hash()
    write_json(Path(a.out) / "e5_result.json", result)
    for k, v in result.items():
        if isinstance(v, dict) and "Q" in v:
            print(k, {n: (q["argmin_M_over_G"], q["best_G_beats_R"], q["G4_beats_R"]) for n, q in v["Q"].items()}, "Q4", {m: round(x["diff"], 4) for m, x in v["Q4"].items()})


if __name__ == "__main__":
    main()
