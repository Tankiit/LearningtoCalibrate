"""E6' (Addendum E, B10/B11): null references and uncertainty for the motivating results."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from .e2_frozen import RUN1, MODELS, load as load_e2, ARMS
from .e_llm import logit, label_scores, conformal_arm, ALPHA
from .fig_transport import load as load_letters, value_plan, cost

B = 2000


def v_pos(model, rel):
    import torch
    return np.asarray(torch.load(f"{RUN1}/{model}/{rel}.pt", map_location="cpu", weights_only=False)["V_pos"], float)


def quintile_stay(a, b, k=5):
    ga = np.ceil(rankdata(-a, method="average") / len(a) * k).clip(1, k)
    gb = np.ceil(rankdata(-b, method="average") / len(b) * k).clip(1, k)
    return float(np.mean([np.mean(gb[ga == g] == g) for g in range(1, k + 1)]))


def median_move(a, b):
    return float(np.median(np.abs(rankdata(-a) - rankdata(-b))))


def main(out="runs/p1_nulls"):
    rng = np.random.default_rng(2027); res = {"bootstrap": B, "source_sha256": source_hash()}
    n = 817
    perm_moves = [float(np.median(np.abs(np.arange(1, n + 1) - rng.permutation(np.arange(1, n + 1))))) for _ in range(10000)]
    res["rank_null"] = {"median_move_independent_mean": float(np.mean(perm_moves)),
                        "q025_q975": np.quantile(perm_moves, [.025, .975]).tolist(),
                        "note": "the report is a deterministic expected value: a same-prompt redraw moves no rank"}
    for m in MODELS:
        f, r, a = v_pos(m, "main/fwd"), v_pos(m, "main/rev"), v_pos(m, "alt/fwd")
        entry = {}
        for tag, other in (("reversed", r), ("reworded", a)):
            stays, moves = [], []
            for _ in range(B):
                i = rng.integers(n, size=n)
                stays.append(quintile_stay(f[i], other[i])); moves.append(median_move(f[i], other[i]))
            entry[tag] = {"stay": quintile_stay(f, other), "stay_ci": np.quantile(stays, [.025, .975]).tolist(),
                          "median_move": median_move(f, other), "median_move_ci": np.quantile(moves, [.025, .975]).tolist()}
        # transport costs with question bootstrap (rows: pos block then neg block)
        Pf, vf = load_letters(m, "main/fwd")
        for tag, rel in (("reversed", "main/rev"), ("reworded", "alt/fwd")):
            Q, vq = load_letters(m, rel); vals = {"C_real": [], "C_self": [], "C_shuf": [], "iota": []}
            def costs(idx):
                Pi, Qi = Pf[idx], Q[idx]
                cr = cost(value_plan(Pi.T @ Qi / len(idx), vf, vq)); cs = cost(value_plan(Pi.T @ Pi / len(idx), vf, vf))
                csh = cost(value_plan(np.outer(Pi.mean(0), Qi.mean(0)), vf, vq))
                return cr, cs, csh
            point = costs(np.arange(len(Pf)))
            for _ in range(B):
                q = rng.integers(n, size=n); idx = np.concatenate([q, q + n])
                cr, cs, csh = costs(idx)
                vals["C_real"].append(cr); vals["C_self"].append(cs); vals["C_shuf"].append(csh)
                vals["iota"].append((cr - cs) / (csh - cs) if csh - cs > 1e-9 else np.nan)
            gap = point[2] - point[1]
            entry[f"transport_{tag}"] = {"C_real": point[0], "C_self": point[1], "C_shuf": point[2],
                                         **{f"{k}_ci": np.nanquantile(v, [.025, .975]).tolist() for k, v in vals.items()},
                                         "iota": (point[0] - point[1]) / gap if gap > 0 else None,
                                         "iota_interpretable": bool(gap >= 2.0)}
        res[m] = entry
        print("done", m, flush=True)
    # B11: E2-frozen per-split coverage quantiles (same protocol as experiments.e2_frozen, 500 splits, M=4)
    e2 = {}
    for m in MODELS:
        ids, y, V = load_e2(m); per = {f"{a}->{b}": [] for a, b in ARMS}
        for seed in range(500):
            perm = np.random.default_rng(np.random.SeedSequence([seed, 1717])).permutation(len(y))
            f_, c, te = np.array_split(perm, 3)
            for a_, b_ in ARMS:
                part = ExemplarPartition.fit(logit(V[a_])[f_], 4, "kmeans", 0)
                r_ = conformal_arm("x", part(logit(V[a_])[c]), label_scores(V[a_][c], y[c]), part(logit(V[b_])[te]), V[b_][te], y[te], 4)
                per[f"{a_}->{b_}"].append(r_["coverage"])
        e2[m] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                     "q10_q90": np.quantile(v, [.1, .9]).tolist()} for k, v in per.items()}
        print("e2 done", m, flush=True)
    res["e2_frozen"] = e2
    write_json(Path(out) / "nulls.json", res)


if __name__ == "__main__":
    main()
