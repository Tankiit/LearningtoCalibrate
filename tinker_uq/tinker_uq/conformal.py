"""Finite-sample Mondrian classification with partitions fixed before calibration.

For z in {0,1}, R(x,z)=1-p_anchor(z|x). Quantile is the
ceil((n_cell+1)*(1-alpha))-th order statistic, with infinity if rank>n_cell.
The implementation uses inclusive <= membership and preserves ties.
"""
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from .data import read_predictions, join_signals, one_per_question, write_json, write_jsonl, digest


def quantile(scores, alpha):
    if not 0<alpha<1:
        raise ValueError('alpha must lie strictly between 0 and 1')
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Scores must be a finite vector')
    rank=math.ceil((len(values)+1)*(1-alpha))
    if rank>len(values):
        return math.inf
    return float(np.partition(values,rank-1)[rank-1])


def partition_edges(values, k):
    if k<1 or type(k) is not int:
        raise ValueError('K must be a positive integer')
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Partition fitting requires finite pre-calibration scores')
    unique, counts=np.unique(values, return_counts=True)
    if len(unique)<2 or k==1:
        return []
    # Boundaries between observed values; equal scores are never split.
    # Select the observed cumulative mass closest to each requested quantile.
    cumulative=np.cumsum(counts)[:-1]/len(values)
    indices=sorted({int(np.argmin(abs(cumulative-j/k))) for j in range(1,k)})
    # A boundary equals the upper value; searchsorted(side=right) sends it right.
    # This avoids floating-point midpoint collapse for adjacent float values.
    return [float(unique[i+1]) for i in indices]


def cell_id(value, edges):
    return int(np.searchsorted(edges,value,side='right'))


def fit_model(fit_rows, cal_rows, signal, k, alpha):
    if not fit_rows or not cal_rows:
        raise ValueError('Nonempty partition and calibration data required')
    if {r['question_id'] for r in fit_rows}&{r['question_id'] for r in cal_rows}:
        raise ValueError('Partition/calibration question leakage')
    if signal not in ('q','qa'):
        raise ValueError('Unknown signal')
    edges=partition_edges([r[signal] for r in fit_rows],k)
    scores=defaultdict(list)
    for r in cal_rows:
        p=r['anchor_p']
        score=1-p if r['correct'] else p
        scores[cell_id(r[signal],edges)].append(score)
    thresholds=[]
    for i in range(len(edges)+1):
        qhat=quantile(scores[i],alpha)
        thresholds.append({'cell':i,'n_cal':len(scores[i]),
                           'qhat':None if math.isinf(qhat) else qhat,
                           'full_set_fallback':math.isinf(qhat)})
    return {'signal':signal,'requested_k':k,'effective_k':len(edges)+1,
            'alpha':alpha,'edges':edges,'cells':thresholds}


def prediction_set(row, model):
    cell=cell_id(row[model['signal']],model['edges'])
    qhat=model['cells'][cell]['qhat']
    qhat=math.inf if qhat is None else qhat
    p=row['anchor_p']
    return cell,[z for z,score in ((0,p),(1,1-p)) if score<=qhat]


def wilson(successes,n):
    if n==0:
        return [None,None]
    z=1.959963984540054
    p=successes/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-radius),min(1,center+radius)]


def evaluate_model(rows,model):
    if not rows:
        raise ValueError('Empty test set')
    predictions=[]
    for r in rows:
        cell,labels=prediction_set(r,model)
        predictions.append(dict(row_id=r['row_id'],question_id=r['question_id'],cell=cell,
                                correct=r['correct'],labels=labels,size=len(labels),covered=r['correct'] in labels))
    def stats(rs):
        n=len(rs)
        hits=sum(r['covered'] for r in rs)
        return {'n_test':n,'coverage':hits/n if n else None,
                'coverage_wilson95':wilson(hits,n),
                'mean_set_size':sum(r['size'] for r in rs)/n if n else None,
                'empty_rate':sum(r['size']==0 for r in rs)/n if n else None,
                'singleton_rate':sum(r['size']==1 for r in rs)/n if n else None,
                'full_set_rate':sum(r['size']==2 for r in rs)/n if n else None}
    cells=[dict(c,**stats([r for r in predictions if r['cell']==c['cell']])) for c in model['cells']]
    nonempty=[c['coverage'] for c in cells if c['n_test']]
    summary=dict(signal=model['signal'],requested_k=model['requested_k'],effective_k=model['effective_k'],
                 alpha=model['alpha'],**stats(predictions),
                 worst_observed_cell_coverage=min(nonempty),
                 full_set_fallback_cells=sum(c['full_set_fallback'] for c in cells),
                 min_calibration_cell_count=min(c['n_cal'] for c in cells))
    return summary,cells,predictions


def fit(args):
    # Test values are validated by the loader, but never used in partitions,
    # threshold fitting, signal selection, or K selection.
    rows=read_predictions(args.predictions)
    joined=join_signals(rows,args.stage,args.legend,args.anchor)
    selected=one_per_question(joined,args.candidate_policy,args.seed)
    fit_rows=[r for r in selected if r['split']==args.partition_split]
    cal_rows=[r for r in selected if r['split']=='cal']
    ks=sorted(set(args.k))
    if not ks or any(k<1 for k in ks):
        raise ValueError('K values must be positive')
    models=[fit_model(fit_rows,cal_rows,signal,k,args.alpha) for signal in ('q','qa') for k in ks]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    state={'version':1,'stage':args.stage,'legend':args.legend,'anchor':args.anchor,
           'partition_split':args.partition_split,'candidate_policy':args.candidate_policy,'seed':args.seed,
           'source_sha256':digest(args.predictions),
           'fit_question_ids':[r['question_id'] for r in fit_rows],
           'cal_question_ids':[r['question_id'] for r in cal_rows],
           'selected_row_ids':[r['row_id'] for r in selected],
           'models':models,
           'interpretation':'Fixed-K comparisons; no test-based selection. Null qhat means infinity.'}
    write_json(out/'calibration.json',state)
    print(f'Saved {len(models)} fixed calibrators to {out}; test metrics not computed.')


def evaluate(args):
    import json
    state=json.loads(Path(args.calibration).read_text())
    if digest(args.predictions)!=state['source_sha256']:
        raise ValueError('Prediction file changed since fitting. Use the exact frozen export.')
    rows=join_signals(read_predictions(args.predictions),state['stage'],state['legend'],state['anchor'])
    selected=one_per_question(rows,state['candidate_policy'],state['seed'])
    test=[r for r in selected if r['split']=='test']
    if {r['question_id'] for r in test}&set(state['fit_question_ids']+state['cal_question_ids']):
        raise ValueError('Test leakage')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    summaries=[]; cells=[]; predictions=[]
    for model in state['models']:
        summary,cell_stats,sets=evaluate_model(test,model)
        summaries.append(summary)
        tag={k:model[k] for k in ('signal','requested_k','effective_k','alpha')}
        cells.extend(dict(tag,**x) for x in cell_stats)
        predictions.extend(dict(tag,**x) for x in sets)
    write_json(out/'summary.json',summaries)
    write_jsonl(out/'cells.jsonl',cells)
    write_jsonl(out/'sets.jsonl',predictions)
    import csv
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(f'Saved test coverage and set-size comparisons to {out}. No best K selected.')
