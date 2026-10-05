"""Run every P1 experiment under one source hash and record what ran.

  python -m experiments.run_all [--jobs 3] [--only NAME ...] [--popqa] [--check]

Stages run in order; jobs inside a stage run in parallel (each may itself use joblib).
The manifest runs/run_all_manifest.json records, per job, the command, the source hash
at launch, wall time, return code, log path and the sha256 of every expected output.
--check reruns nothing: it verifies that every job finished with return code 0 under
the current source hash and that its outputs still have the recorded hashes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from mondrian.audit import source_hash

MANIFEST = Path("runs/run_all_manifest.json")
MG = "1 2 3 4 5 6 8 11 14 18 23 30 39 51 67 87 112 146 190".split()
NG = "300 500 800 1300 2000 3200 5000 8000".split()


def x1(d, ff):
    out = f"runs/p1_x1_v2{'_ff' if ff else ''}_d{d}"
    cmd = ["experiments.x1_synthetic", "--protocol", "v2", *(["--method", "voronoi"] if ff else []), "--out", out,
           "--d", str(d), "--seeds", "20", "--n-fit", "4000", "--n-test", "12000", "--M-grid", *MG, "--n-cal-grid", *NG]
    return (f"Known-Law (X1) {'ff ' if ff else ''}d={d}", cmd, [f"{out}/gate.json", f"{out}/summary.json"])


# (name used in the paper, module + args, expected outputs)
STAGES = [
    [x1(1, False), x1(2, False), x1(1, True), x1(2, True),
     ("Answerability (X4)", ["experiments.x4_selection", "--out", "runs/p1_x4", "--seeds", "50"], ["runs/p1_x4/gate.json"]),
     ("Trained-Reporters/Legend-Swap (E1/E2)", ["experiments.e_llm", "--which", "e12", "--out", "runs/p1_e12", "--seeds", "20"],
      ["runs/p1_e12/e12_result.json"]),
     ("PopQA-Probe (E3)", ["experiments.e_llm", "--which", "e3", "--out", "runs/p1_e3", "--seeds", "20"], ["runs/p1_e3/e3_result.json"]),
     ("Reporters re-split (E1/E2 x)", ["experiments.e_llm", "--which", "e12x", "--out", "runs/p1_e12x"],
      ["runs/p1_e12x/e12x_result.json", "runs/p1_e12x/e1x_forecast_rows_q.json"]),
     ("PopQA-Probe re-split (E3 x)", ["experiments.e_llm", "--which", "e3x", "--out", "runs/p1_e3x"], ["runs/p1_e3x/e3x_result.json"]),
     ("Elicitation-Shift (E2-frozen)", ["experiments.e2_frozen", "--out", "runs/p1_e2_frozen"], ["runs/p1_e2_frozen/e2_frozen_result.json"]),
     ("Rank nulls", ["experiments.e_nulls"], ["runs/p1_nulls/nulls.json"]),
     ("Granularity pre-set (E5 v1)", ["experiments.e5_partition", "--out", "runs/p1_e5"], ["runs/p1_e5/e5_result.json"]),
     ("Diameter lemma check", ["experiments.check_diameter"], ["runs/p1_diameter/summary.json"])],
    [("Communities (X2)", ["experiments.x2_tabular", "--dataset", "communities", "--out", "runs/p1_x2"], ["runs/p1_x2/gate.json"]),
     ("Extras (top-k, balanced, argmin)", ["experiments.e_extras", "--out", "runs/p1_extras"], ["runs/p1_extras/extras_result.json"]),
     ("Granularity (E5)", ["experiments.e5v2", "--out", "runs/p1_e5v2"], ["runs/p1_e5v2/e5v2_result.json"]),
     ("Granularity extra (E5 TOST, one-cell)", ["experiments.e5v2_extra"], ["runs/p1_e5v2_extra/e5v2_extra.json"]),
     ("Synthetic-Reliability (E7)", ["experiments.e_synth", "--out", "runs/p1_synth_v2", "--dims", "4", "--clusters", "32"],
      ["runs/p1_synth_v2/synth_summary.json"]),
     ("Label-Free-Transfer (E8)", ["experiments.e8_transport"], ["runs/p1_e8/e8_result.json"]),
     ("Controlled-Shifts (E10)", ["experiments.e10_controlled"], ["runs/p1_e10/e10_result.json"]),
     ("Real-Geometry (E11)", ["experiments.e11_semisynth"], ["runs/p1_e11/e11_result.json"]),
     ("Failure-Modes TruthfulQA (E12)", ["experiments.e12_failure", "--data", "tqa", "--seeds", "100"], ["runs/p1_e12/e12_tqa.json"])],
    [("Figure: legend (Fig 1a, Fig 4)", ["experiments.fig_legend"], ["paper/figs/fig1a.pdf"]),
     ("Figure: transport plans", ["experiments.fig_transport"], ["paper/figs/fig_transport.pdf"]),
     ("Figures: main, Granularity, Synthetic", ["experiments.fig_e5"], ["paper/figs/fig_main.pdf", "paper/figs/fig_e5.pdf"]),
     ("Figure: reliability diagrams", ["experiments.fig_reliability"], ["paper/figs/fig_reliability.pdf"])],
]
POPQA = [
    [("PopQA-Elicit: Elicitation-Shift (E9a)", ["experiments.e9_popqa", "--part", "e2"], ["runs/p1_e9/e9a_e2frozen.json"]),
     ("PopQA-Elicit: Label-Free-Transfer (E9c)", ["experiments.e9_popqa", "--part", "e8"], ["runs/p1_e9/e9c_e8_llama3_8b_mistral_7b_qwen2_5_7b.json"]),
     ("Failure-Modes PopQA (E12)", ["experiments.e12_failure", "--data", "popqa", "--seeds", "50"], ["runs/p1_e12/e12_popqa.json"])],
    [(f"PopQA-Elicit: Granularity (E9b) {m}", ["experiments.e9_popqa", "--part", "e5", "--model", m], [f"runs/p1_e9/e9b_e5_{m}.json"])
     for m in ("llama3_8b", "mistral_7b", "qwen2_5_7b")],
]


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def run_job(job, src):
    name, cmd, outs = job
    log = Path("runs/run_all_logs") / f"{slug(name)}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([sys.executable, "-m", *cmd], stdout=fh, stderr=subprocess.STDOUT).returncode
    rec = {"command": "python -m " + " ".join(cmd), "source_sha256": src, "returncode": rc, "wall_seconds": round(time.time() - t0, 1),
           "log": str(log), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "outputs": {o: file_sha(o) for o in outs}}
    rec["outputs_present"] = all(rec["outputs"].values())
    print(f"[{'ok' if rc == 0 and rec['outputs_present'] else 'FAIL'}] {name} ({rec['wall_seconds']}s)", flush=True)
    return name, rec


def load():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def check(stages):
    man, src, bad = load(), source_hash(), 0
    for stage in stages:
        for name, cmd, outs in stage:
            r = man.get(name)
            if r is None:
                msg = "never run"
            elif r["returncode"] != 0:
                msg = f"returncode {r['returncode']}"
            elif r["source_sha256"] != src:
                msg = "stale source hash"
            elif any(file_sha(o) != h for o, h in r["outputs"].items()):
                msg = "output changed or missing since the run"
            else:
                msg = "ok"
            bad += msg != "ok"
            print(f"{msg:40s} {name}")
    print(f"source {src[:12]}; {bad} job(s) not current")
    return bad


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--jobs", type=int, default=3)
    pa.add_argument("--only", nargs="*", default=None, help="substring(s) of job names to run")
    pa.add_argument("--popqa", action="store_true", help="also run the PopQA-Elicit stages (needs the E9 reports)")
    pa.add_argument("--check", action="store_true")
    a = pa.parse_args()
    stages = STAGES + (POPQA if a.popqa else [])
    if a.check:
        sys.exit(1 if check(stages) else 0)
    src = source_hash()
    for stage in stages:
        jobs = [j for j in stage if a.only is None or any(s in j[0] for s in a.only)]
        if not jobs:
            continue
        with ThreadPoolExecutor(a.jobs) as ex:
            results = list(ex.map(lambda j: run_job(j, src), jobs))
        man = load(); man.update(dict(results))
        MANIFEST.write_text(json.dumps(man, indent=1))
        if source_hash() != src:
            raise RuntimeError("Source changed during the run; manifest entries are stale")
    sys.exit(check(stages) and 1)


if __name__ == "__main__":
    main()
