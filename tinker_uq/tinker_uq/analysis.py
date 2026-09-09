"""ICLR diagnostics. Positive Brier improvement means qa beats q."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .data import read_predictions, write_json
from .training import summarize


def safe_spearman(a,b):
    if len(a)<2 or len(set(a))<2 or len(set(b))<2:
        return None
    return float(spearmanr(a,b).statistic)


def tie_averaged_aurc(rs):
    """Expected AURC over uniform ordering within equal-confidence groups."""
    groups=defaultdict(list)
    for r in rs:
        groups[r['p_correct']].append(1-r['correct'])
    count=0; errors=0.; area=0.
    for p in sorted(groups,reverse=True):
        group=groups[p]; rate=sum(group)/len(group)
        for j in range(1,len(group)+1):
            area+=(errors+j*rate)/(count+j)
        count+=len(group); errors+=sum(group)
    return area/count


def brier_gain(q,qa,seed=0,draws=1000):
    if set(q)!=set(qa):
        raise ValueError('Mismatched q/qa row IDs')
    groups=defaultdict(list)
    for rid in sorted(q):
        a,b=q[rid],qa[rid]
        groups[a['question_id']].append((a['p_correct']-a['correct'])**2-(b['p_correct']-b['correct'])**2)
    values=np.array([np.mean(v) for _,v in sorted(groups.items())])
    rng=np.random.default_rng(seed)
    boot=[float(np.mean(rng.choice(values,size=len(values),replace=True))) for _ in range(draws)]
    return {'question_mean_brier_gain_qa_over_q':float(np.mean(values)),
            'question_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'questions':len(values),'bootstrap_draws':draws}


def run(args):
    rows=read_predictions(args.predictions)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    metrics=summarize(args.predictions)
    groups=defaultdict(list)
    for r in rows:
        groups[(r['arm'],r['stage'],r['legend'],r['split'])].append(r)
    for key,rs in groups.items():
        metrics['/'.join(key)]['tie_averaged_aurc']=tie_averaged_aurc(rs)
    comparisons={}; legends={}
    for split in ('val','test'):
        for stage in ('base','trained'):
            q={r['row_id']:r for r in groups.get(('q',stage,'normal',split),[])}
            qa={r['row_id']:r for r in groups.get(('qa',stage,'normal',split),[])}
            if q and qa:
                comparisons[f'{stage}/{split}']=brier_gain(q,qa,args.seed,args.bootstrap)
            for arm in ('q','qa'):
                a={r['row_id']:r for r in groups.get((arm,stage,'normal',split),[])}
                b={r['row_id']:r for r in groups.get((arm,stage,'reversed',split),[])}
                if not a or not b: continue
                if set(a)!=set(b): raise ValueError('Mismatched legend row IDs')
                ids=sorted(a); av=[a[i]['p_correct'] for i in ids]; bv=[b[i]['p_correct'] for i in ids]
                item={'spearman':safe_spearman(av,bv),
                      'mean_absolute_probability_change':float(np.mean(np.abs(np.array(av)-bv))),
                      'retention':[]}
                for fraction in (.5,.8,.9):
                    n=max(1,int(np.ceil(len(ids)*fraction)))
                    # Fixed row-id tie rule, disclosed in README; no correctness used.
                    aa=set(sorted(ids,key=lambda i:(-a[i]['p_correct'],i))[:n])
                    bb=set(sorted(ids,key=lambda i:(-b[i]['p_correct'],i))[:n])
                    item['retention'].append({'fraction':fraction,'n_retained':n,
                        'intersection_fraction':len(aa&bb)/n,
                        'normal_error':sum(1-a[i]['correct'] for i in aa)/n,
                        'reversed_error':sum(1-b[i]['correct'] for i in bb)/n})
                legends[f'{arm}/{stage}/{split}']=item
    write_json(out/'metrics.json',metrics)
    write_json(out/'answer_gain.json',comparisons)
    write_json(out/'legend_transfer.json',legends)
    print(f'Saved ICLR diagnostics to {out}')
