"""Overlap diagnostic: pooled top-k neighborhoods have no Mondrian guarantee."""
import argparse
from pathlib import Path
import numpy as np
from data.synthetic import sample,coverage_cdf
from models.partition import ExemplarPartition
from mondrian.cells import assign_topk
from mondrian.conformal import per_cell_quantile
from mondrian.splits import make_splits,split_hash,freeze_splits
from mondrian.audit import require_gate,write_free_check,manifest,array_hash,write_json,source_hash
from .common import validate_first,require_tests,summarize


def pooled_quantiles(cal_members,scores,test_members,alpha,floor):
    """Deduplicate calibration ROWS when any exemplar membership overlaps."""
    thresholds=np.empty(len(test_members))
    counts=np.empty(len(test_members),int)
    for membership in np.unique(test_members,axis=0):
        test_mask=np.all(test_members==membership,axis=1)
        selected=np.any(np.isin(cal_members,membership),axis=1)
        q,n=per_cell_quantile(np.zeros(selected.sum(),int),scores[selected],1,alpha,floor)
        thresholds[test_mask]=q[0]
        counts[test_mask]=n[0]
    return thresholds,counts


def run(out="runs/p1_topk",x1_gate="runs/p1_x1/gate.json",seeds=range(8),M=16,ks=(1,2,4),floor=20,alpha=.1):
    require_tests(out)
    require_gate(x1_gate,"x1")
    rows=[]
    for seed in seeds:
        X,S=sample(5000,2,np.random.default_rng(np.random.SeedSequence([seed,101])))
        splits=make_splits(len(X),seed=seed)
        freeze_splits(Path(out)/"splits"/f"seed{seed}.json",splits,array_hash(np.column_stack([X,S])))
        part=ExemplarPartition.fit(X[splits["fit"]],M,seed=seed)
        for k in ks:
            rule=lambda x: assign_topk(x,part.exemplars,k)
            write_free_check(rule,X[splits["test"]][:10],X[splits["test"]][10:30])
            cal=rule(X[splits["cal"]]);test=rule(X[splits["test"]])
            q,counts=pooled_quantiles(cal,S[splits["cal"]],test,alpha,floor)
            probs=coverage_cdf(X[splits["test"]],q)
            rows.append({"seed":seed,"M":M,"k":k,"coverage":float(np.mean(S[splits["test"]]<=q)),
                "pointwise_msce":float(np.mean((probs-(1-alpha))**2)),
                "test_fraction_below_floor":float(np.mean(counts<floor)),
                "unique_pool_count":int(len(np.unique(test,axis=0))),
                "guarantee":"Mondrian only for k=1; no assertion for k>1",
                "manifest":manifest(split_hashes={a:split_hash(b) for a,b in splits.items()},M=M,seed=seed,
                    floor=floor,rule="overlap_pooled_topk",n_below_floor=int(np.sum(counts<floor)),
                    floor_count_unit="test queries",k=k,partition=dict(part.metadata))})
    write_json(Path(out)/"rows.json",rows)
    write_json(Path(out)/"summary.json",summarize(rows,["M","k"],["coverage","pointwise_msce","test_fraction_below_floor"]))
    # Completion permits next experiment; it does NOT release these numbers to paper.
    write_json(Path(out)/"gate.json",{"stage":"topk","status":"PASS","paper_status":"HOLD",
        "source_sha256":source_hash(),"scope":"implementation diagnostic completed"})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/p1_topk")
    p.add_argument("--x1-gate",default="runs/p1_x1/gate.json")
    args=p.parse_args();validate_first(args.out)
    run(out=args.out,x1_gate=args.x1_gate)


if __name__=="__main__":
    main()
