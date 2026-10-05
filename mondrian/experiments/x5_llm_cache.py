"""Optional isolated correctness-prediction study; no P2 numbers enter P1."""
import argparse
from pathlib import Path
import numpy as np
from data.llm_cache import load_cache,one_row_per_question
from models.scores import ScoreModel
from models.partition import ExemplarPartition
from mondrian.splits import make_splits,freeze_splits,split_hash
from mondrian.conformal import per_cell_quantile,coverage_by_cell,functionals
from mondrian.audit import require_gate,manifest,write_json,array_hash,source_hash
from .common import validate_first,require_tests


def run(path,model,dataset,layer_pool,out="runs/p1_x5",previous_gate="runs/p1_level2/gate.json",seeds=range(8),M=16,floor=20,alpha=.1):
    require_tests(out);require_gate(previous_gate,"level2")
    X,y,qid=load_cache(path,model,dataset,layer_pool)
    rows=[]
    for seed in seeds:
        Z,v,ids,selected=one_row_per_question(X,y,qid,seed)
        splits=make_splits(len(Z),seed=seed,groups=ids)
        freeze_splits(Path(out)/"splits"/f"seed{seed}.json",splits,array_hash(selected))
        f,c,t=(splits[k] for k in ("fit","cal","test"))
        score=ScoreModel("classification",seed).fit(Z[f],v[f])
        part=ExemplarPartition.fit(score.features(Z[f]),M,seed=seed)
        cc,ct=part(score.features(Z[c])),part(score.features(Z[t]))
        q,counts=per_cell_quantile(cc,score.score(Z[c],v[c]),M,alpha,floor)
        cov,nt=coverage_by_cell(ct,score.score(Z[t],v[t]),q)
        rows.append({"seed":seed,"M":M,**functionals(cov,nt,alpha),"counts_cal":counts.tolist(),
            "counts_test":nt.tolist(),"selected_rows":selected.tolist(),"n_below_floor":int(np.sum(counts<floor)),
            "manifest":manifest(split_hashes={k:split_hash(i) for k,i in splits.items()},M=M,seed=seed,
                floor=floor,rule="voronoi",n_below_floor=int(np.sum(counts<floor)),model=model,
                dataset=dataset,layer_pool=layer_pool,scope="optional_correctness_prediction_P1_HOLD")})
    write_json(Path(out)/"rows.json",rows)
    write_json(Path(out)/"gate.json",{"stage":"x5","status":"HOLD","source_sha256":source_hash(),
        "reason":"Optional cache evidence requires separate scientific review; no P1 release"})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("path","model","dataset","layer-pool"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--out",default="runs/p1_x5")
    p.add_argument("--previous-gate",default="runs/p1_level2/gate.json")
    a=p.parse_args();validate_first(a.out)
    run(a.path,a.model,a.dataset,a.layer_pool,a.out,a.previous_gate)


if __name__=="__main__":
    main()
