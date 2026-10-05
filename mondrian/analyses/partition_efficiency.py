"""Post-lock analysis A (reports/postlock_analyses_note_2026-10-05.md): partition efficiency with a fixed score.

  python -m analyses.partition_efficiency [--seeds 100]   -> runs_postlock/partition_efficiency.json

The correctness model (score) is fitted once per split and procedure and held fixed; only the partition and its
per-cell conformal thresholds change. Reads cached states and reports only; imports locked code read-only.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from models.partition import ExemplarPartition
from mondrian.conformal import per_cell_quantile
from experiments.e_llm import label_scores, logit, FLOOR
from experiments.e5_partition import load_tqa
from experiments.e5v2 import tuned
from experiments.e9_popqa import load_popqa_tqa_style

ALPHA, MS = 0.1, (2, 4, 8)
OUT = Path("runs_postlock")


def one(H, y, V, seed, salt):
    n = len(y)
    f, c, te = np.array_split(np.random.default_rng(np.random.SeedSequence([seed, salt])).permutation(n), 3)
    c1, c2 = c[:len(c) // 2], c[len(c) // 2:]
    Z = PCA(16, random_state=0).fit(H[f]).transform(H)
    rep = {M: ExemplarPartition.fit(Z[f], M, "voronoi", seed)(Z) for M in MS}
    out = {}
    for p, v in V.items():
        lv = logit(v).reshape(-1, 1)
        scores = {"R": tuned(lv[c1], y[c1].astype(float), seed)[0].predict(lv),
                  "D": tuned(np.hstack([lv, Z])[c1], y[c1].astype(float), seed)[0].predict(np.hstack([lv, Z]))}
        parts = {"M1": (np.zeros(n, int), 1)}
        for M in MS:
            edges = np.quantile(v[f], np.linspace(0, 1, M + 1)[1:-1])
            parts[f"bins{M}"] = (np.searchsorted(edges, v), M)
            parts[f"rep{M}"] = (rep[M], M)
        for sname, ps in scores.items():
            for pname, (g, M) in parts.items():
                q, counts = per_cell_quantile(g[c2], label_scores(ps[c2], y[c2]), M, ALPHA, FLOOR)
                thr = q[g[te]]
                s1, s0 = label_scores(ps[te], 1) <= thr, label_scores(ps[te], 0) <= thr
                hit = np.where(y[te] == 1, s1, s0); size = s1.astype(int) + s0
                cov_cells = [hit[g[te] == j].mean() for j in range(M) if np.any(g[te] == j)]
                out.setdefault(f"{sname}|{pname}", []).append(
                    [hit.mean(), size.mean(), np.mean(size == 2), np.mean(np.isinf(thr)), min(cov_cells)])
    return {k: np.mean(v, 0).tolist() for k, v in out.items()}


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--seeds", type=int, default=100)
    a = pa.parse_args()
    OUT.mkdir(exist_ok=True)
    res = {"note": "reports/postlock_analyses_note_2026-10-05.md#a", "fields": ["coverage", "set_size", "full_set_rate",
                                                                                  "fallback_mass", "worst_cell_coverage"]}
    jobs = max(1, (os.cpu_count() or 2) - 1)
    for dname, loader, salt in (("truthfulqa", load_tqa, 2525), ("popqa_elicit", load_popqa_tqa_style, 2527)):
        for m in ("llama3_8b", "mistral_7b", "qwen2_5_7b"):
            H, y, V = loader(m)
            per = Parallel(n_jobs=jobs)(delayed(one)(H, y, V, s, salt) for s in range(a.seeds))
            res[f"{dname}|{m}"] = {k: np.mean([r[k] for r in per], 0).tolist() for k in per[0]}
            res[f"{dname}|{m}"]["n_labelled"] = int(len(np.array_split(np.arange(len(y)), 3)[1]))
            print("done", dname, m, flush=True)
    (OUT / "partition_efficiency.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
