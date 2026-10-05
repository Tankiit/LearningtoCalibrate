"""Paired-seed Voronoi versus empirical OT power cells at matched M."""
import argparse
from pathlib import Path
import numpy as np
from mondrian.audit import require_gate,write_json,source_hash
from .common import validate_first,require_tests,summarize
from .x1_synthetic import run as synthetic_run,exponent_test,METRICS


def run(out="runs/p1_level2",previous_gate="runs/p1_x2/gate.json",seeds=range(8),
        M_grid=(1,2,4,8,16,32),n_cal_grid=(400,800,1600,3200),d=1):
    require_tests(out);require_gate(previous_gate,"x2")
    rows=[];exponents={}
    for rule in ("voronoi","power"):
        arm=synthetic_run(M_grid,n_cal_grid,d,seeds,rule=rule,out=out)
        rows+=arm;exponents[rule]=exponent_test(arm)
    lookup={(r["seed"],r["n_cal"],r["M"],r["rule"]):r for r in rows}
    differences=[]
    for r in rows:
        if r["rule"]=="voronoi":
            p=lookup[r["seed"],r["n_cal"],r["M"],"power"]
            differences.append({"seed":r["seed"],"n_cal":r["n_cal"],"M":r["M"],
                                **{m:p[m]-r[m] for m in METRICS}})
    write_json(Path(out)/"rows.json",rows)
    write_json(Path(out)/"summary.json",summarize(rows,["rule","n_cal","M"],METRICS))
    write_json(Path(out)/"paired_differences.json",summarize(differences,["n_cal","M"],METRICS))
    write_json(Path(out)/"exponents.json",exponents)
    write_json(Path(out)/"gate.json",{"stage":"level2","status":"PASS","paper_status":"HOLD",
        "source_sha256":source_hash(),"scope":"implementation diagnostic completed",
        "interpretation":"Fit balance does not imply population balance; inspect all residuals"})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/p1_level2")
    p.add_argument("--previous-gate",default="runs/p1_x2/gate.json")
    a=p.parse_args();validate_first(a.out);run(a.out,a.previous_gate)


if __name__=="__main__":
    main()
