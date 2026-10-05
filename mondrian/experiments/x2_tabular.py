"""Exemplar cells versus given-group and global conformal on tabular data."""
import argparse
from pathlib import Path
import time
import numpy as np
from data.tabular import load
from models.scores import ScoreModel
from models.partition import ExemplarPartition
from mondrian.audit import require_gate,write_json,manifest,source_hash,write_free_check,array_hash
from mondrian.splits import make_splits,freeze_splits,split_hash
from mondrian.conformal import per_cell_quantile,coverage_by_cell,functionals
from .common import validate_first,require_tests,summarize


def run(dataset="communities",path="data/raw/communities.data",out="runs/p1_x2",
        previous_gate="runs/p1_x1_v2_d1/gate.json",previous_stage="x1",seeds=range(8),
        M_grid=(2,4,8,16,32),floor=20,alpha=.1):
    # Order changed 2026-10-02 (estimand note, section 5): X2 follows X1 directly.
    require_tests(out);require_gate(previous_gate,previous_stage)
    X,y,groups,cluster,meta=load(dataset,path)
    rows=[]
    for seed in seeds:
        start=time.perf_counter()
        splits=make_splits(len(X),seed=seed,groups=cluster)
        freeze_splits(Path(out)/"splits"/f"seed{seed}.json",splits,meta["sha256"])
        f,c,t=(splits[k] for k in ("fit","cal","test"))
        score=ScoreModel(meta["task"],seed).fit(X[f],y[f])
        Zf,Zc,Zt=(score.features(X[a]) for a in (f,c,t))
        sc,st=score.score(X[c],y[c]),score.score(X[t],y[t])
        # Category vocabulary from fit only; unseen groups share a reserved cell.
        vocabulary={v:i for i,v in enumerate(sorted(set(groups[f])))}
        group_rule=lambda g: np.array([vocabulary.get(v,len(vocabulary)) for v in g])
        gc,gt=group_rule(groups[c]),group_rule(groups[t])
        arms=[("global",1,np.zeros(len(c),int),np.zeros(len(t),int),{}),
              ("given_group",len(vocabulary)+1,gc,gt,{"given_group":meta["given_group"]})]
        for M in M_grid:
            part=ExemplarPartition.fit(Zf,M,seed=seed)
            write_free_check(part,Zt[:10],Zt[10:30])
            arms.append(("exemplar",M,part(Zc),part(Zt),dict(part.metadata)))
        for arm,M,cc,ct,part_meta in arms:
            q,counts=per_cell_quantile(cc,sc,M,alpha,floor)
            cov,nt=coverage_by_cell(ct,st,q)
            metrics=functionals(cov,nt,alpha)
            # Given-group coverage is evaluated on identical groups across all arms.
            hit=st<=q[ct]
            gn=np.bincount(gt,minlength=len(vocabulary)+1)
            gcov=np.divide(np.bincount(gt,weights=hit,minlength=len(gn)),gn,
                           out=np.full(len(gn),np.nan),where=gn>0)
            group_metrics=functionals(gcov,gn,alpha)
            efficiency=score.efficiency(X[t],q[ct]);finite=np.isfinite(efficiency)
            rows.append({"seed":seed,"M":M,"arm":arm,**metrics,
                "subgroup_gap":group_metrics["concentrated"],"counts_cal":counts.tolist(),
                "counts_test":nt.tolist(),"given_group_counts":gn.tolist(),"given_group_coverages":gcov.tolist(),
                "n_below_floor":int(np.sum(counts<floor)),"test_mass_infinite":float(np.mean(np.isinf(q[ct]))),
                "mean_efficiency_finite_only":float(efficiency[finite].mean()) if finite.any() else None,
                "efficiency_infinite_fraction":float(np.mean(~finite)),
                "manifest":manifest(split_hashes={k:split_hash(v) for k,v in splits.items()},M=M,seed=seed,
                    floor=floor,rule=arm,n_below_floor=int(np.sum(counts<floor)),dataset=meta,
                    partition=part_meta,score_model=type(score.predictor).__name__,
                    fit_features_hash=array_hash(Zf),elapsed_seed_seconds=time.perf_counter()-start)})
    summary=summarize(rows,["arm","M"],["coverage","subgroup_gap","n_below_floor","test_mass_infinite"])
    exemplars=sorted([r for r in summary if r["arm"]=="exemplar"],key=lambda r:r["M"])
    best=int(np.argmin([r["subgroup_gap"]["mean"] for r in exemplars]))
    shape="interior minimum observed; uncertainty still required" if 0<best<len(exemplars)-1 else "no interior minimum resolved; do not claim a tradeoff"
    write_json(Path(out)/"rows.json",rows);write_json(Path(out)/"summary.json",summary)
    write_json(Path(out)/"gate.json",{"stage":"x2","status":"PASS","paper_status":"HOLD",
        "source_sha256":source_hash(),"scope":"implementation diagnostic completed","shape":shape})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["communities","acs_income"],default="communities")
    p.add_argument("--path",default="data/raw/communities.data")
    p.add_argument("--out",default="runs/p1_x2")
    p.add_argument("--previous-gate",default="runs/p1_x1_v2_d1/gate.json")
    p.add_argument("--previous-stage",default="x1")
    a=p.parse_args();validate_first(a.out)
    run(a.dataset,a.path,a.out,a.previous_gate,a.previous_stage)


if __name__=="__main__":
    main()
