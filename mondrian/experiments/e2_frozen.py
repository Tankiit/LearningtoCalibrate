"""Addendum C: E2 on frozen models (Run 1 legend-reversal extraction, TruthfulQA)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from models.partition import ExemplarPartition
from mondrian.audit import write_json, source_hash
from .common import validate_first, require_tests
from .e_llm import logit, label_scores, conformal_arm, expectation_check, binned_forecast, ALPHA

RUN1 = ("/Users/tanmoy/research/OOD_detection/OVA_ARR/verbalized_conformal_confidence/LearningtoCalibrate/"
        "cached_results/p2_run1/run1_20260919_464acbb")
MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
VARIANTS = {"fwd": "main/fwd", "rev": "main/rev", "alt": "alt/fwd"}
ARMS = (("fwd", "fwd"), ("rev", "rev"), ("alt", "alt"), ("fwd", "rev"), ("fwd", "alt"))


def load(model, root=RUN1):
    import torch
    out, ids = {}, None
    for k, rel in VARIANTS.items():
        d = torch.load(f"{root}/{model}/{rel}.pt", map_location="cpu", weights_only=False)
        e = np.asarray(d["example_ids"]).astype(str)
        if ids is None:
            ids = e
        elif not np.array_equal(ids, e):
            raise ValueError(f"{model}: example ids differ across variants")
        out[k] = (np.asarray(d["V_pos"], float), np.asarray(d["V_neg"], float))
    pick_pos = np.array([hashlib.sha256(("p1-e2f|" + q).encode()).digest()[0] & 1 == 0 for q in ids])
    y = pick_pos.astype(int)
    V = {k: np.where(pick_pos, vp, vn) for k, (vp, vn) in out.items()}
    return ids, y, V


def run(seeds=range(500), M=4):
    res = {}
    for model in MODELS:
        ids, y, V = load(model)
        per = {f"{a}->{b}": [] for a, b in ARMS}
        for seed in seeds:
            perm = np.random.default_rng(np.random.SeedSequence([seed, 1717])).permutation(len(y))
            f, c, te = np.array_split(perm, 3)
            for a, b in ARMS:
                part = ExemplarPartition.fit(logit(V[a])[f], M, "kmeans", 0)
                cc, ct = part(logit(V[a])[c]), part(logit(V[b])[te])
                r = conformal_arm(f"{a}->{b}", cc, label_scores(V[a][c], y[c]), ct, V[b][te], y[te], M)
                pt, *_ = binned_forecast(cc, y[c].astype(float), ct, M)
                r["brier_binned"] = float(np.mean((pt - y[te]) ** 2))
                per[f"{a}->{b}"].append(r)
        res[model] = {k: {**expectation_check(v), "brier_binned": float(np.mean([r["brier_binned"] for r in v])),
                          "coverage_Y1": float(np.mean([r["coverage_Y1"] for r in v])),
                          "coverage_Y0": float(np.mean([r["coverage_Y0"] for r in v]))} for k, v in per.items()}
        res[model]["n"] = int(len(y)); res[model]["share_Y1"] = float(y.mean())
    t = 1 - ALPHA
    consistent = all(res[m][k]["valid"] for m in MODELS for k in ("fwd->fwd", "rev->rev", "alt->alt"))
    legend_break = any(res[m]["fwd->rev"]["coverage_mean"] < t - .02 or res[m]["fwd->rev"]["worst_cell_mean"] < t - .02
                       for m in MODELS)
    prompt_break = {m: bool(res[m]["fwd->alt"]["coverage_mean"] < t - .02 or res[m]["fwd->alt"]["worst_cell_mean"] < t - .02)
                    for m in MODELS}
    res["checks"] = {"consistent_arms_valid": consistent, "legend_change_breaks": legend_break,
                     "prompt_change_breaks_descriptive": prompt_break}
    res["status"] = "PASS" if consistent and legend_break else "FAIL"
    return res


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--out", default="runs/p1_e2_frozen")
    a = pa.parse_args(); validate_first(a.out); require_tests(a.out)
    res = run()
    res.update(criterion="reports/estimand_note_E123_2026-10-02.md#addendum-c", source_sha256=source_hash(),
               paper_status="HOLD")
    write_json(Path(a.out) / "e2_frozen_result.json", res)
    print(res["status"], res["checks"])
    for m in MODELS:
        for k in ("fwd->fwd", "rev->rev", "alt->alt", "fwd->rev", "fwd->alt"):
            e = res[m][k]
            print(f"{m:11s} {k:9s} cov {e['coverage_mean']:.3f} (se {e['coverage_se']:.4f}) worst {e['worst_cell_mean']:.3f} "
                  f"z {e['min_cell_z']:7.2f} size {e['mean_set_size']:.3f} brier {e['brier_binned']:.4f} "
                  f"cov|Y1 {e['coverage_Y1']:.3f} cov|Y0 {e['coverage_Y0']:.3f}")


if __name__ == "__main__":
    main()
