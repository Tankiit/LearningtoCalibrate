"""Compile every P1 result into one Markdown file, read directly from the run outputs.

  python reports/compile_results.py            -> reports/README_RESULTS.md

Lives in reports/ (outside the hashed source folders), so running it never stales the manifest.
Every number below is read from runs/*; nothing is typed by hand except the section prose.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "README_RESULTS.md"
MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
SHORT = {"llama3_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "qwen2_5_7b": "Qwen2.5-7B"}
L = []


def J(p):
    return json.loads((ROOT / p).read_text())


def w(s=""):
    L.append(s)


def f(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{d}f}"


def ci(c, d=3, scale=1.0):
    return f"[{c[0] * scale:.{d}f}, {c[1] * scale:.{d}f}]"


def table(head, rows):
    w("| " + " | ".join(head) + " |")
    w("|" + "---|" * len(head))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def holm_adjust(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- provenance
def provenance():
    man = J("runs/run_all_manifest.json")
    srcs = {r["source_sha256"] for r in man.values()}
    w("## 10. Provenance and reproduction")
    w()
    from mondrian.audit import source_hash
    cur = source_hash()
    ok = all(r["returncode"] == 0 and r["outputs_present"] for r in man.values())
    w(f"- Manifest `runs/run_all_manifest.json`: {len(man)} jobs, source hash(es) {', '.join(s[:12] for s in srcs)}; "
      f"current source {cur[:12]} ({'matches' if srcs == {cur} else 'DOES NOT match'}); all return codes 0: {f(ok)}.")
    w("- Reproduce everything: `python -m experiments.run_all --jobs 3 --popqa`; verify: `python -m experiments.run_all --popqa --check`.")
    w("- Registered abstract (binding scope): `reports/registered_abstract_2026-09-29.md`.")
    w("- Dated protocol notes: `reports/estimand_note_2026-10-02.md` (Known-Law), `estimand_note_E123_2026-10-02.md` "
      "(Reporters, Elicitation-Shift), `estimand_note_E5_2026-10-02.md` (Granularity, Synthetic-Reliability), "
      "`estimand_note_E8_2026-10-02.md` (Label-Free-Transfer), `estimand_note_E9_popqa_2026-10-03.md` "
      "(PopQA-Elicit, Controlled-Shifts, Real-Geometry, Failure-Modes), `calibration_toolbox_note_2026-10-03.md`, "
      "`transport_controls_note_2026-10-03.md`.")
    w()
    rows = [(n, f"`{r['command']}`", r["wall_seconds"], "ok" if r["returncode"] == 0 else "FAIL") for n, r in man.items()]
    w("<details><summary>Jobs (name, command, seconds, status)</summary>")
    w()
    table(["experiment", "command", "s", "status"], rows)
    w("</details>")
    w()
    w("**Experiment names and identifiers.** Known-Law = X1, Communities = X2, Answerability = X4, "
      "Trained-Reporters / Legend-Swap / PopQA-Probe = E1 / E2 / E3, Elicitation-Shift = E2-frozen, Granularity = E5, "
      "Synthetic-Reliability = E7, Label-Free-Transfer = E8, PopQA-Elicit = E9, Controlled-Shifts = E10, "
      "Real-Geometry = E11, Failure-Modes = E12.")
    w()
    w("**Conventions.** Brier differences are *method − recalibration (R)*, below 0 = better. Intervals on real data are "
      "Nadeau–Bengio-corrected 95% over repeated random splits. Conformal target 0.90 (α = 0.1). SCE = class-wise "
      "calibration error from calibration-toolbox (= binary ECE, 15 bins); a diagnostic only.")
    w()


# ---------------------------------------------------------------- 1. breaks
def breaks():
    w("## 1. Elicitation changes break calibration")
    w()
    n = J("runs/p1_nulls/nulls.json")
    w(f"Rank null (two independent rankings): median move {n['rank_null']['median_move_independent_mean']:.0f} "
      f"{ci(n['rank_null']['q025_q975'], 0)}. TruthfulQA, 817 questions, correct candidate.")
    w()
    rows = []
    for m in MODELS:
        r = n[m]
        rows.append([SHORT[m], f"{r['reversed']['median_move']:.0f} {ci(r['reversed']['median_move_ci'], 0)}",
                     f"{100 * r['reversed']['stay']:.1f}% {ci(np.array(r['reversed']['stay_ci']) * 100, 1)}",
                     f"{r['reworded']['median_move']:.0f} {ci(r['reworded']['median_move_ci'], 0)}",
                     f"{100 * r['reworded']['stay']:.1f}% {ci(np.array(r['reworded']['stay_ci']) * 100, 1)}"])
    table(["model", "median rank move, reversed", "stay in fifth, reversed", "median move, reworded", "stay, reworded"], rows)
    w("**Coverage when the procedure changes between calibration and deployment** (Elicitation-Shift; 500 random thirds, "
      "M = 4 report cells; PopQA rows: PopQA-Elicit, 7,189 questions).")
    w()
    rows = []
    for name, path in (("TruthfulQA", "runs/p1_e2_frozen/e2_frozen_result.json"), ("PopQA", "runs/p1_e9/e9a_e2frozen.json")):
        r = J(path)
        for m in MODELS:
            same = [r[m][k]["coverage_mean"] for k in ("fwd->fwd", "rev->rev", "alt->alt")]
            rows.append([name, SHORT[m], f"{min(same):.3f}–{max(same):.3f}", f(r[m]["fwd->rev"]["coverage_mean"]),
                         f(r[m]["fwd->alt"]["coverage_mean"]), f(r[m]["fwd->fwd"]["mean_set_size"], 2)])
        w(f"- {name}: status **{r['status']}**; checks {r['checks']}") if False else None
    table(["data", "model", "same procedure", "legend reversed", "prompt reworded", "set size (same)"], rows)
    for name, path in (("TruthfulQA", "runs/p1_e2_frozen/e2_frozen_result.json"), ("PopQA", "runs/p1_e9/e9a_e2frozen.json")):
        r = J(path); w(f"- {name}: status **{r['status']}**; consistent arms valid: {f(r['checks']['consistent_arms_valid'])}; "
                       f"legend change breaks: {f(r['checks']['legend_change_breaks'])}.")
    w()


# ---------------------------------------------------------------- 2. granularity
def gran_table(S, budgets, arms=("G4-R", "Gcv-R", "Rand4-R", "MC-R", "D-R", "G4-D"), scale=1e3):
    rows = []
    for n in budgets:
        c = S["contrasts"][str(n)]
        row = [n]
        for a in arms:
            if a in c:
                row.append(f"{c[a]['mean'] * scale:+.1f} {ci(c[a]['ci95'], 1, scale)}")
            else:
                row.append("–")
        row.append(f(c["G4-R"]["p_one_sided_less"]))
        row.append(f(S["selected_M"][str(n)].get("16", 0), 2))
        rows.append(row)
    table(["labels"] + [a + " (×10³)" for a in arms] + ["p (G4<R)", "P(M=16)"], rows)


def granularity():
    w("## 2. Granularity: groups vs recalibration, multicalibration and a direct predictor")
    w()
    r = J("runs/p1_e5v2/e5v2_result.json")
    w("Arms: K base rate; R Platt recalibration per procedure; G_M groups (farthest-first on PCA-16 of the frozen hidden "
      "state, model logit P = a_π + b_π logit V + c_{G,π}); G_cv with M ∈ {1,2,4,8,16} by inner CV; Rand_M random groups; "
      "MC holdout multicalibration; D logistic on representation + report. All tuned by inner CV; report coefficient up-weighted.")
    w()
    for m in MODELS:
        S = r[f"tqa_{m}"]
        w(f"### TruthfulQA, {SHORT[m]} (200 random splits)")
        w()
        gran_table(S, (60, 120, 180, 272))
        cal = S["calibration"]["272"]
        w("SCE at 272 labels: " + ", ".join(f"{k} {cal[k]['sce']:.3f}" for k in ("K", "R", "G4", "Gcv", "MC", "D")) +
          f". Brier R {S['brier']['272']['R']:.4f}, K {S['brier']['272']['K']:.4f}.")
        w()
    tq = [r[f"tqa_{m}"]["contrasts"][str(n)]["G4-R"]["p_one_sided_less"] for m in MODELS for n in (60, 120, 180, 272)]
    adj = holm_adjust(tq)
    w(f"**Pre-declared Holm family (TruthfulQA, 3 models × 4 budgets, G4−R):** smallest raw p {min(tq):.3f}; "
      f"smallest Holm-adjusted p {adj.min():.3f}; rejections: {int((adj <= .05).sum())}.")
    w()
    ex = J("runs/p1_e5v2_extra/e5v2_extra.json")
    rows = [[SHORT[m], ci(ex[m]["tost_GS4_minus_G4"]["272"]["ci90"], 4)] for m in MODELS]
    w("Shared group offsets vs per-procedure groups (TOST, margin ±0.002, 90% corrected CI at 272 labels):")
    w()
    table(["model", "GS4 − G4"], rows)
    w("Conformal sets (α = 0.1; forecaster on half the labels, thresholds on the other half):")
    w()
    rows = []
    for m in MODELS:
        C = r[f"tqa_{m}"]["conformal"]
        rows.append([SHORT[m]] + [f"{C[f'{a}|a0.1']['coverage']:.3f} / {C[f'{a}|a0.1']['set_size']:.2f}"
                                  for a in ("R_unstrat", "G4_mondrian", "D_unstrat")] +
                    [ci(C["size_G4-R|a0.1"]["ci95"]), ci(C["size_D-R|a0.1"]["ci95"])])
    table(["model", "R cov/size", "G4 Mondrian cov/size", "D cov/size", "size G4−R CI", "size D−R CI"], rows)
    S = r["popqa"]
    w("### PopQA-Probe (probe report, one procedure, 100 splits)")
    w()
    gran_table(S, sorted(map(int, S["contrasts"])), arms=("G4-R", "Gcv-R", "MC-R", "D-R"))


def popqa_elicit():
    w("## 3. PopQA-Elicit: the registered question at ~2,400 labels")
    w()
    w("7,189 PopQA questions (188 alias collisions removed), same three elicitations, frozen TruthfulQA templates; "
      "100 random thirds. Added after the TruthfulQA family returned no significant result.")
    w()
    P, K = [], []
    for m in MODELS:
        S = J(f"runs/p1_e9/e9b_e5_{m}.json")[m]
        w(f"### {SHORT[m]} — report AUROC: " + ", ".join(f"{p} {v['auroc']:.3f}" for p, v in S["informativeness"].items()))
        w()
        ns = sorted(map(int, S["contrasts"]))
        gran_table(S, ns)
        C = S["conformal"]
        w(f"Conformal (α = 0.1) size G4−R {ci(C['size_G4-R|a0.1']['ci95'])}, D−R {ci(C['size_D-R|a0.1']['ci95'])}; singletons "
          + ", ".join(f"{a.split('_')[0]} {C[f'{a}|a0.1']['singleton_rate']:.2f} (acc. {C[f'{a}|a0.1']['singleton_coverage']:.2f})"
                      for a in ("R_unstrat", "G4_mondrian", "D_unstrat")) + ".")
        cal = S["calibration"][str(ns[-1])]
        w(f"SCE at {ns[-1]}: " + ", ".join(f"{k} {cal[k]['sce']:.3f}" for k in ("K", "R", "G4", "Gcv", "D")) + ".")
        w()
        for n in ns:
            P.append(S["contrasts"][str(n)]["G4-R"]["p_one_sided_less"]); K.append((m, n))
    adj = holm_adjust(P)
    rows = sorted(zip(K, P, adj), key=lambda t: t[1])[:6]
    w("**Pre-declared Holm family (3 models × 5 budgets, G4−R).** Smallest six:")
    w()
    table(["model", "labels", "raw p", "Holm-adjusted p"], [[SHORT[k[0]], k[1], f"{p:.4f}", f"{a:.3f}"] for k, p, a in rows])
    r = J("runs/p1_e5v2/e5v2_result.json")
    tq = [r[f"tqa_{m}"]["contrasts"][str(n)]["G4-R"]["p_one_sided_less"] for m in MODELS for n in (60, 120, 180, 272)]
    pooled = holm_adjust(tq + P)
    w(f"Rejections at 0.05: {int((adj <= .05).sum())} of 15. Pooled with TruthfulQA (27 tests): smallest Holm-adjusted p "
      f"{pooled.min():.3f}, rejections {int((pooled <= .05).sum())}.")
    w()


# ---------------------------------------------------------------- 4. synthetic
def synthetic():
    w("## 4. Synthetic-Reliability and Real-Geometry")
    w()
    s = J("runs/p1_synth_v2/synth_summary.json")
    w(f"**Synthetic-Reliability** ({s['clusters']} clusters in {s['dims']}-d; 30 seeds). Differences ×10³ with 95% CI, "
      "averaged over report informativeness κ (κ does not change the pattern; full rows in the JSON).")
    w()
    rows = []
    for st in ("cluster", "linear"):
        for tau in sorted({q["tau"] for q in s["rows"]}):
            for n in sorted({q["n_lab"] for q in s["rows"]}):
                sel = [q for q in s["rows"] if q["structure"] == st and q["tau"] == tau and q["n_lab"] == n]
                if sel:
                    gr = np.mean([q["G-R"] for q in sel]) * 1e3; gd = np.mean([q["G-D"] for q in sel]) * 1e3
                    sig_r = sum(q["G-R_ci"][1] < 0 for q in sel); sig_d = sum(q["G-D_ci"][1] < 0 for q in sel)
                    rows.append([st, tau, n, f"{gr:+.1f}", f"{sig_r}/{len(sel)}", f"{gd:+.1f}", f"{sig_d}/{len(sel)}"])
    table(["structure", "τ", "labels", "G−R", "κ with G<R (CI)", "G−D", "κ with G<D (CI)"], rows)
    e = J("runs/p1_e11/e11_result.json")
    w("**Real-Geometry** (Llama PopQA hidden states; known η; pointwise error E(f̂−η)² ×10³; 20 seeds), averaged over κ:")
    w()
    rows = []
    for st in ("cluster", "linear"):
        for tau in (0.0, .5, 1.0, 2.0):
            for n in (100, 300, 1000, 2000):
                sel = [q for q in e["rows"] if q["structure"] == st and q["tau"] == tau and q["n_lab"] == n]
                rows.append([st, tau, n, f"{np.mean([q['G-R'] for q in sel]) * 1e3:+.1f}",
                             f"{sum(q['G-R_ci'][1] < 0 for q in sel)}/{len(sel)}",
                             f"{np.mean([q['G-D'] for q in sel]) * 1e3:+.1f}",
                             f"{sum(q['G-D_ci'][0] > 0 for q in sel)}/{len(sel)}",
                             f"{np.mean([q['mean_M'] for q in sel]):.1f}"])
    table(["structure", "τ", "labels", "G−R", "G<R sig.", "G−D", "D<G sig.", "mean M"], rows)


# ---------------------------------------------------------------- 5. transport
def transport():
    w("## 5. Transport across procedures")
    w()
    arms = ("A->A", "naive", "ot_quantile", "ot_quantile_mid", "paired_iso", "paired_iso_inc", "paired_iso_dec",
            "paired_iso_group", "recal_B_k30", "recal_B_k120")
    for name, path, key in (("TruthfulQA (Label-Free-Transfer, 500 splits)", "runs/p1_e8/e8_result.json", "frozen"),
                            ("PopQA-Elicit (100 splits)", "runs/p1_e9/e9c_e8_llama3_8b_mistral_7b_qwen2_5_7b.json", None)):
        R = J(path); R = R[key] if key else R
        w(f"### {name}")
        w()
        for m in MODELS:
            for B in ("rev", "alt"):
                a = R[m][B]
                w(f"**{SHORT[m]}, forward → {'reversed' if B == 'rev' else 'reworded'}** "
                  f"(paired direction increasing in {a['_direction_increasing']:.0%} of splits; tie fractions "
                  + ", ".join(f"{k} {v:.3f}" for k, v in R[m]["_tie_fraction"].items()) + ")")
                w()
                table(["arm", "coverage", "10–90%", "size", "Brier", "SCE", "crossing", "label-loss", "worst group gap"],
                      [[k, f(a[k]["coverage"]), ci(a[k]["coverage_q10_q90"]), f(a[k]["set_size"], 2), f(a[k]["brier"], 4),
                        f(a[k]["sce"]), f(a[k]["crossing_mass"]), f(a[k]["label_loss_mass"]), f(a[k]["cell_gap_min"])]
                       for k in arms if k in a])
    t = J("runs/p1_e8/e8_result.json")["tinker"]
    w("### TriviaQA trained reporters (one fixed split; 95% bootstrap CI)")
    w()
    for arm, a in t.items():
        w(f"**{'question-only (q)' if arm == 'q' else 'answer-aware (qa)'}**, normal → reversed; paired direction "
          f"increasing: {f(a['paired_direction_increasing'])}; tie fractions A {a['tie_fraction']['A']:.3f}, B {a['tie_fraction']['B']:.3f}")
        w()
        table(["arm", "coverage", "95% CI", "size", "Brier", "SCE", "crossing", "label-loss"],
              [[k, f(v["coverage"]), ci(v["coverage_ci95"]) if v.get("coverage_ci95") else "–", f(v["set_size"], 2),
                f(v["brier"], 4), f(v["sce"]), f(v["crossing_mass"]), f(v["label_loss_mass"])]
               for k, v in a.items() if isinstance(v, dict) and "coverage" in v])
    rel = J("paper/figs/fig_reliability_numbers.json")
    w("**Reliability figure (pooled SCE):**")
    w()
    table(["case"] + list(next(iter(rel["cases"].values())).keys()),
          [[c.replace("\n", " ")] + [f(v["sce"]) for v in d.values()] for c, d in rel["cases"].items()])


def controlled():
    w("## 6. Controlled-Shifts (transport proposition on real reports)")
    w()
    e = J("runs/p1_e10/e10_result.json")
    w(f"Status **{e['status']}**. Synthetic B: logit V_B = s·logit V_A + b + σ·sd·ε. Pre-set checks per dataset:")
    w()
    keys = list(next(iter(e["checks"].values())).keys())
    table(["dataset"] + keys, [[k] + [f(v[c]) for c in keys] for k, v in e["checks"].items()])
    w("Oracle (true noiseless inverse) max |Δcoverage| at σ = 0: " +
      ", ".join(f"{k} {v['oracle_max_abs_dcov_sigma0']:.3f}" for k, v in e["controls"].items()) + ".")
    w()
    arms = ("naive", "quantile", "quantile_mid", "paired_iso", "paired_iso_inc", "paired_iso_dec", "oracle_inverse")
    for k in list(MODELS) + ["tinker_qa"]:
        R = e[k]
        w(f"**{SHORT.get(k, 'TriviaQA answer-aware reporter')}** — mean per-split |Δcoverage| (coverage of A→A "
          f"{R['s1_b0_sig0']['A->A']['coverage']:.3f}); b = 0")
        w()
        rows = []
        for s_ in (1, -1):
            for sg in (0, 0.1, 0.25, 0.5, 1, 2):
                g = R[f"s{s_}_b0_sig{sg:g}"]
                rows.append([s_, sg] + [f(g[a]["abs_dcov"]) for a in arms] + [f(g["paired_iso"]["crossing_mass"])])
        table(["s", "σ"] + list(arms) + ["crossing (paired)"], rows)


def failure():
    w("## 7. Failure-Modes")
    w()
    for data, n in (("tqa", 100), ("popqa", 50)):
        e = J(f"runs/p1_e12/e12_{data}.json")
        w(f"### {'TruthfulQA' if data == 'tqa' else 'PopQA-Elicit'} ({n} splits)")
        w()
        for m in MODELS:
            A = e[m]["aggregate"]
            w(f"**{SHORT[m]}** — checks {e[m]['checks']}; tails/ties: " +
              ", ".join(f"{p} {v['tail_fraction']:.3f}/{v['tied_fraction']:.3f}" for p, v in e[m]["mechanics"].items()))
            w()
            arms = ["bins|raw", "bins|quantile", "bins|paired", "repcells|raw", "repcells|quantile", "mc|raw", "mc|quantile",
                    "direct_state_report|raw", "direct_state_report|quantile", "direct_state|raw"]
            rows = []
            for sh in A:
                rows.append([sh] + [f"{A[sh][a]['coverage']:.3f}" if a in A[sh] else "–" for a in arms])
            table(["shift"] + arms, rows)
            w("Set sizes, no shift: " + ", ".join(f"{a} {A['none'][a]['size']:.2f}" for a in arms if a in A["none"]) + ".")
            w()


# ---------------------------------------------------------------- 8. reporters & theory checks
def reporters():
    w("## 8. Reporters (Trained-Reporters, Legend-Swap, PopQA-Probe) — appendix experiments")
    w()
    for name, path in (("Trained-Reporters / Legend-Swap (pre-set)", "runs/p1_e12/e12_result.json"),
                       ("re-split sensitivity (post hoc)", "runs/p1_e12x/e12x_result.json")):
        r = J(path)
        w(f"**{name}**: forecast q status {r['forecast']['q'].get('status')}, qa status {r['forecast']['qa'].get('status')}; "
          f"Legend-Swap status {r['e2']['status']} {r['e2']['checks']}" +
          (f"; conformal status {r['conformal']['status']}" if "conformal" in r else "") + ".")
    for name, path in (("PopQA-Probe (pre-set)", "runs/p1_e3/e3_result.json"), ("PopQA-Probe re-split (post hoc)", "runs/p1_e3x/e3x_result.json")):
        r = J(path)
        w(f"**{name}**: conformal status {r['conformal']['status']}.")
    w()
    rows = []
    for a in J("runs/p1_e12/e12_result.json")["conformal"]["arms"]:
        rows.append([a["arm"], a["M"], f(a["coverage"]), ci(a["wilson95"]), f(a["mean_set_size"], 2), f(a["full_rate"])])
    w("Trained-Reporters conformal arms (TriviaQA, α = 0.1):")
    w()
    table(["arm", "M", "coverage", "Wilson 95%", "size", "full-set rate"], rows)


def theory():
    w("## 9. Checks of the theory")
    w()
    rows = []
    for d, ff in ((1, False), (2, False), (1, True), (2, True)):
        g = J(f"runs/p1_x1_v2{'_ff' if ff else ''}_d{d}/gate.json")
        rows.append([f"Known-Law d={d} {'farthest-first' if ff else 'k-means'}", g["status"],
                     ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in g["checks"].items()), ci(g["equivalence_ci"])])
    x4 = J("runs/p1_x4/gate.json"); x2 = J("runs/p1_x2/gate.json")
    rows.append(["Answerability (X4)", x4["status"], ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in x4["checks"].items()), "–"])
    rows.append(["Communities (X2)", x2["status"], x2.get("shape", ""), "–"])
    table(["experiment", "status", "checks", "slope equivalence CI"], rows)
    dm = J("runs/p1_diameter/summary.json")["fits"]
    w("Farthest-first diameter lemma (empirical log-diameter slope in M, theory −1/d): " +
      ", ".join(f"{k} {v['slope_M>=4']:.3f}" for k, v in dm.items()) + ".")
    w()
    ex = J("runs/p1_extras/extras_result.json")
    w("Top-k retrieval (coverage / size): " + ", ".join(f"{k} {v['coverage_mean']:.3f}/{v['set_size_mean']:.2f}" for k, v in ex["topk"].items()) + ".")
    w()


def headlines():
    w("## Headlines (computed from the sections below)")
    w()
    e2 = J("runs/p1_e2_frozen/e2_frozen_result.json"); e9a = J("runs/p1_e9/e9a_e2frozen.json")
    r = J("runs/p1_e5v2/e5v2_result.json")
    tq = [r[f"tqa_{m}"]["contrasts"][str(n)]["G4-R"]["p_one_sided_less"] for m in MODELS for n in (60, 120, 180, 272)]
    P = []; dr_p = []; g_p = []; m16 = []
    for m in MODELS:
        S = J(f"runs/p1_e9/e9b_e5_{m}.json")[m]
        P += [S["contrasts"][str(n)]["G4-R"]["p_one_sided_less"] for n in sorted(map(int, S["contrasts"]))]
        c = S["contrasts"]["2396"]; dr_p.append(c["D-R"]["mean"] * 1e3); g_p.append(c["G4-R"]["mean"] * 1e3)
        m16.append(S["selected_M"]["2396"]["16"])
    adj = holm_adjust(P)
    dr_t = [r[f"tqa_{m}"]["contrasts"]["272"]["D-R"]["mean"] for m in MODELS]
    e8 = J("runs/p1_e8/e8_result.json"); q = e8["tinker"]["q"]
    mis = [e8["frozen"]["mistral_7b"][B] for B in ("rev", "alt")]
    w(f"1. **Elicitation changes break calibration.** Coverage under a changed procedure: TruthfulQA Mistral "
      f"{e2['mistral_7b']['fwd->rev']['coverage_mean']:.2f}/{e2['mistral_7b']['fwd->alt']['coverage_mean']:.2f} (reversed/reworded); "
      f"PopQA Llama {e9a['llama3_8b']['fwd->rev']['coverage_mean']:.2f} and Mistral {e9a['mistral_7b']['fwd->rev']['coverage_mean']:.2f} "
      f"under reversal; trained reporter {q['naive']['coverage']:.2f} (target 0.90).")
    w(f"2. **Groups vs recalibration.** TruthfulQA: no pre-declared test survives Holm (smallest raw p {min(tq):.3f}). "
      f"PopQA-Elicit: {int((adj <= .05).sum())} of 15 survives (smallest adjusted p {adj.min():.3f}); gain at 2,396 labels "
      f"{min(g_p):.1f} to {max(g_p):.1f} ×10⁻³ Brier; P(M = 16 chosen) {min(m16):.2f}–{max(m16):.2f}.")
    w(f"3. **Direct predictor does better.** D − R: TruthfulQA {min(dr_t) * 1e3:.0f} to {max(dr_t) * 1e3:.0f} ×10⁻³ at 272 labels; "
      f"PopQA {min(dr_p):.0f} to {max(dr_p):.0f} ×10⁻³ at 2,396.")
    w(f"4. **Transport.** Paired isotonic, monotone change (trained q reporter): {q['naive']['coverage']:.3f} → "
      f"{q['paired_iso']['coverage']:.3f}, label-loss {q['paired_iso']['label_loss_mass']:.3f}. Quantile map, Mistral TruthfulQA: "
      f"{min(a['naive']['coverage'] for a in mis):.2f}–{max(a['naive']['coverage'] for a in mis):.2f} → "
      f"{min(a['ot_quantile']['coverage'] for a in mis):.2f}–{max(a['ot_quantile']['coverage'] for a in mis):.2f}. "
      "Marginal repair, not per group (Section 5, worst group gap).")
    w()


def overview():
    w("# How Finely to Group? — results README")
    w()
    w(f"*AISTATS 2027 submission (full paper due 6 Oct 2026, 23:59 AoE). Generated {time.strftime('%Y-%m-%d %H:%M')} by "
      "`python reports/compile_results.py`; every number below is read from the run outputs in `runs/`.*")
    w()
    w("## The question")
    w()
    w("An LLM's confidence report is calibrated to correctness under the procedure used to elicit it (a prompt, a scale, a "
      "letter legend). The registered question (`reports/registered_abstract_2026-09-29.md`) is whether a partition of "
      "question-answer examples, learned from a frozen representation before elicitation, adds calibration structure across "
      "procedures beyond recalibrating the report, and how finely to group under a limited label budget. It is evaluated "
      "with controlled reporting interventions, synthetic reliability models and held-out factual QA, against "
      "procedure-specific recalibration, multicalibration and a direct predictor on the same features, separating "
      "correctness forecasting from conformal prediction sets.")
    w()
    w("## What we found")
    w()
    w("- **Elicitation changes break calibration.** Reversing the letter legend or rewording the prompt reshuffles which "
      "questions look confident and breaks the coverage of conformal sets calibrated under the original procedure.")
    w("- **Groups help only with real group structure and plentiful labels.** On TruthfulQA (272 labels) no pre-declared test "
      "survives correction; on PopQA (about 2,400 labels) one of fifteen does, the gain is under 1% of the Brier score, and the "
      "best number of groups grows with the budget. Synthetic and semi-synthetic models show the same pattern.")
    w("- **A direct predictor on the same representation does better** in every setting tested, and groups do not shrink "
      "conformal sets.")
    w("- **Groups do not protect calibration across a procedure change**, because the forecast still reads the report. Only a "
      "predictor that ignores the report is invariant, at a cost in set size when the report is informative.")
    w("- **Transport phenomenon.** A monotone map between two procedures' reports, fitted on unlabelled questions, carries a "
      "calibration across procedures: exactly for a monotone change, partly otherwise, and on average rather than per group. "
      "With informative reports (PopQA) no label-free map is reliably exact; 60+60 labels under the new procedure match "
      "the target on average. No arm, labelled or label-free, keeps every frozen group at the target after a change.")
    w()
    w("## Caveats")
    w()
    w("- PopQA-Elicit, Controlled-Shifts, Real-Geometry and Failure-Modes were added after the TruthfulQA analysis returned no "
      "significant group effect; pooled over all 27 group-vs-recalibration tests nothing survives Holm correction (Section 3).")
    w("- The Granularity analysis reported in the paper is a review-driven re-analysis (tuned forecasters, corrected "
      "intervals), marked post hoc; the pre-set analysis is superseded.")
    w("- Models are 7-8B; candidates are fixed and label-blind, not the model's own answers; procedures are two legends and "
      "two wordings; groups are capped at 16.")
    w("- Simulated reviews used the same model family as the assistant that built the paper; they are role-separated, not "
      "independent.")
    w()
    w("## Where everything lives")
    w()
    table(["what", "where"], [
        ["Working draft (with \\todo notes naming each command)", "`paper/main.tex`"],
        ["Submission source / PDF", "`paper/aistats2027_submission.tex` (built by `python paper/build_submission.py`) / `paper/aistats2027_submission.pdf` (`cd paper && tectonic aistats2027_submission.tex`)"],
        ["Figures (PDF, PNG and plotted numbers)", "`paper/figs/`"],
        ["Registered abstract", "`reports/registered_abstract_2026-09-29.md`"],
        ["Dated protocol notes", "`reports/estimand_note_*.md`, `reports/calibration_toolbox_note_2026-10-03.md`, `reports/transport_controls_note_2026-10-03.md`"],
        ["Run outputs and manifest", "`runs/p1_*/`, `runs/run_all_manifest.json`"],
        ["Status page", "`reports/AISTATS_P1_STATUS_2026-10-03.md`"],
        ["Simulated reviews", "`reports/review/`, `reports/rereview/`, `reports/review_round3/`"],
        ["Code", "`experiments/` (one module per experiment; `run_all.py` runs everything), `mondrian/`, `models/`, `data/`"],
    ])


def postlock():
    w("## 11. Post-hoc analyses after the experiment lock (5 Oct)")
    w()
    w("Design: `reports/postlock_analyses_note_2026-10-05.md`. Code `analyses/`, outputs `runs_postlock/`; descriptive, outside the Holm family.")
    w()
    pe = J("runs_postlock/partition_efficiency.json")
    w("**Partition efficiency with a fixed score** (coverage / set size / full-set rate / fallback mass / worst-cell coverage):")
    w()
    rows = []
    for k, v in pe.items():
        if not isinstance(v, dict):
            continue
        for sc in ("R", "D"):
            for p in ("M1", "bins4", "bins8", "rep4", "rep8"):
                x = v[f"{sc}|{p}"]
                rows.append([k, sc, p] + [f"{y:.3f}" for y in x])
    table(["data|model", "score", "partition", "coverage", "set size", "full-set rate", "fallback", "worst cell"], rows)
    pb = J("runs_postlock/probe_baseline.json")
    w("**Correctness-probe baselines** (Brier difference vs R at the largest budget, ×10³, NB 95% CI):")
    w()
    rows = []
    for k, v in pb.items():
        if not isinstance(v, dict):
            continue
        n = max(v, key=int); c = v[n]["brier_vs_R"]
        rows.append([k, n] + [f"{c[a]['mean'] * 1e3:+.1f} {ci(c[a]['ci95'], 1, 1e3)}" for a in ("P_their", "P16", "V+P", "G4", "D")])
    table(["data|model", "labels", "P-full (repo spec)", "P16", "V+P", "G4", "D"], rows)


def main():
    overview(); headlines(); breaks(); granularity(); popqa_elicit(); synthetic(); transport(); controlled(); failure(); reporters(); theory(); postlock(); provenance()
    w("## Reviews")
    w()
    w("Simulated panels (same model family, role-separated, not independent): `reports/review/editorial_decision.md` (round 1), "
      "`reports/rereview/editorial_decision_round2.md` (round 2), `reports/review_round3/editorial_decision_round3.md` (round 3, "
      "with the list of applied revisions).")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, len(L), "lines")


if __name__ == "__main__":
    main()
