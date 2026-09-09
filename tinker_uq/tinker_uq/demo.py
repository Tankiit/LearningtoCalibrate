"""Offline plumbing demo. Scores are simulated, not Tinker/model results."""
from pathlib import Path
import numpy as np
from .data import write_jsonl, write_json


def generate(out, seed=0, n=250):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    if n<2: raise ValueError('Use at least two questions per split')
    rng=np.random.default_rng(seed)
    data=[]; predictions=[]
    for split in ('train','val','cal','test'):
        for i in range(n):
            difficulty=float(rng.uniform(.1,.9))
            answer_effect=float(rng.normal(0,.8))
            true_p=float(1/(1+np.exp(-(np.log(difficulty/(1-difficulty))+answer_effect))))
            y=int(rng.random()<true_p)
            rid=f'{split}-{i}'
            data.append(dict(row_id=rid,question_id=rid,question=f'Demo question {rid}',answer='Synthetic candidate',correct=y,split=split))
            for arm in ('q','qa'):
                signal=difficulty if arm=='q' else true_p
                for stage in ('base','trained'):
                    p=float(np.clip(signal if stage=='trained' else .5+.5*(signal-.5),.001,.999))
                    for legend in ('normal','reversed'):
                        score=p if legend=='normal' else .9*p+.05
                        predictions.append(dict(row_id=rid,question_id=rid,split=split,correct=y,
                                                arm=arm,stage=stage,legend=legend,p_correct=score,
                                                provenance='SIMULATED_OFFLINE_DEMO'))
    write_jsonl(out/'cached_answers.jsonl',data)
    write_jsonl(out/'predictions.jsonl',predictions)
    write_json(out/'manifest.json',{'provenance':'SIMULATED_OFFLINE_DEMO','seed':seed,'questions_per_split':n})
    print(f'Created simulated inputs/outputs in {out}; no Tinker API calls.')
