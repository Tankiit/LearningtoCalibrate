"""Correctness metrics, paired question uncertainty, and consequential decisions."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from .data import load_jsonl,write_json
from .reporting import mapping_items,decode

METRICS=('brier','auroc','aurc')

def metric_vector(y,p):
    y=np.asarray(y); p=np.asarray(p); n=len(y); positives=y.sum()
    auc=float((rankdata(p)[y==1].sum()-positives*(positives+1)/2)/(positives*(n-positives))) if 0<positives<n else np.nan
    order=np.argsort(-p,kind='stable'); s=p[order]; errors=1-y[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]; sizes=np.diff(np.r_[starts,n])
    expected=np.repeat(np.add.reduceat(errors,starts)/sizes,sizes)
    return np.array([np.mean((p-y)**2),auc,np.mean(np.cumsum(expected)/np.arange(1,n+1))])

def csv_save(path,rows):
    if not rows: return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def retention(y,a,b,ids,fraction):
    n=max(1,int(np.ceil(len(y)*fraction)))
    aa=set(np.lexsort((np.asarray(ids),-a))[:n]); bb=set(np.lexsort((np.asarray(ids),-b))[:n])
    return dict(retention=fraction,n_retained=n,intersection_fraction=len(aa&bb)/n,
        jaccard=len(aa&bb)/len(aa|bb),normal_risk=float(np.mean(1-y[list(aa)])),
        reversed_risk=float(np.mean(1-y[list(bb)])),risk_delta=float(np.mean(1-y[list(bb)])-np.mean(1-y[list(aa)])))

def pair_stability(y,a,b,ids):
    d=np.abs(a-b)
    return dict(mean_abs_shift=float(d.mean()),p95_abs_shift=float(np.quantile(d,.95)),
        spearman=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None)

def verified_group(path,refs,mapping):
    rows=load_jsonl(path); d={r['row_id']:r for r in rows}
    if len(d)!=len(rows) or set(d)!=set(refs): raise ValueError(f'IDs mismatch: {path}')
    for rid,r in d.items():
        ref=refs[rid]
        if any(r[k]!=ref[k] for k in ('question_id','correct','split')): raise ValueError('Label/split mismatch')
        if r['mapping']!=mapping: raise ValueError('Mapping mismatch')
        decoded=decode(r['label_logprobs'],mapping)
        for key in ('decoded_mean','allowed_alphabet_mass','allowed_label_logmass'):
            if abs(decoded[key]-r[key])>1e-10: raise ValueError(f'Decoding mismatch {key}')
        if not np.allclose(decoded['probability_vector'],r['probability_vector'],atol=1e-12): raise ValueError('Vector mismatch')
        if abs(r['p_correct']-r['decoded_mean'])>1e-12: raise ValueError('Score alias mismatch')
    return d

def analyze(out,draws=2000):
    from .matched_experiment import METHODS
    out=Path(out); maps=json.loads((out/'mappings.json').read_text()); names=list(maps)
    refs={r['row_id']:r for r in load_jsonl(out/'data.jsonl') if r['split']=='test'}
    ids=sorted(refs); y=np.array([refs[i]['correct'] for i in ids]); methods=['base']+list(METHODS)
    metrics=[]; stability=[]; decisions=[]; codebook=[]; scores={}; boot_metrics={}
    rng=np.random.default_rng(20260914); bootstrap=rng.integers(0,len(ids),size=(draws,len(ids)))
    for method in methods:
        groups=[verified_group(out/method/f'{name}.jsonl',refs,maps[name]) for name in names]
        matrix=np.array([[g[i]['decoded_mean'] for i in ids] for g in groups]); scores[method]=matrix
        evaluated={name:matrix[j] for j,name in enumerate(names)}
        evaluated['normal_reversed_average']=matrix[:2].mean(0)
        observed={name:metric_vector(y,p) for name,p in evaluated.items()}
        # Average metrics across held-out maps is distinct from scoring an averaged prediction.
        observed['heldout_mapping_mean']=np.mean([observed[n] for n in names[2:]],axis=0)
        boots={name:np.array([metric_vector(y[ix],p[ix]) for ix in bootstrap]) for name,p in evaluated.items()}
        boots['heldout_mapping_mean']=np.mean([boots[n] for n in names[2:]],axis=0); boot_metrics[method]=boots
        for name,v in observed.items():
            r=dict(method=method,mapping=name,questions=len(ids))
            for k,m in enumerate(METRICS):
                r[m]=float(v[k]); lo,hi=np.nanquantile(boots[name][:,k],[.025,.975]); r[m+'_lo']=float(lo); r[m+'_hi']=float(hi)
            metrics.append(r)
        d=np.abs(matrix[2:]-matrix[0]); pairs=np.array([np.abs(matrix[a]-matrix[b]) for a in range(2,10) for b in range(a+1,10)])
        st=dict(method=method,**pair_stability(y,matrix[0],matrix[1],ids),
            heldout_vs_normal_mean_abs=float(d.mean()),heldout_vs_normal_p95_abs=float(np.quantile(d,.95)),
            heldout_pairwise_mean_abs=float(pairs.mean()),heldout_pairwise_p95_abs=float(np.quantile(pairs,.95)),
            heldout_per_question_range_mean=float(np.ptp(matrix[2:],axis=0).mean()),
            heldout_per_question_range_p95=float(np.quantile(np.ptp(matrix[2:],axis=0),.95)),
            allowed_mass_mean=float(np.mean([g[i]['allowed_alphabet_mass'] for g in groups for i in ids])))
        for label,values in [('heldout_pairwise_mean_abs',pairs),('heldout_vs_normal_mean_abs',d)]:
            ci=np.quantile([values[:,ix].mean() for ix in bootstrap],[.025,.975]); st[label+'_lo']=float(ci[0]); st[label+'_hi']=float(ci[1])
        stability.append(st)
        for fraction in (.2,.5,.8,.9):
            for j,name in enumerate(names[1:],1):
                r=dict(method=method,comparison='normal_vs_'+name,**retention(y,matrix[0],matrix[j],ids,fraction))
                decisions.append(r)
        cb=json.loads((out/method/'codebook.json').read_text())
        if len(cb)!=110: raise ValueError('Codebook diagnostic incomplete')
        for family in ('normal','reversed','heldout'):
            rs=[r for r in cb if r['mapping_name']==family or (family=='heldout' and r['mapping_name'].startswith('heldout'))]
            codebook.append(dict(method=method,family=family,n=len(rs),argmax_accuracy=float(np.mean([r['codebook_correct'] for r in rs])),
                expected_letter_probability=float(np.mean([r['expected_letter_probability'] for r in rs])),
                mean_numerical_error=float(np.mean([r['numerical_absolute_error'] for r in rs])),
                allowed_mass_mean=float(np.mean([r['allowed_alphabet_mass'] for r in rs]))))
    contrasts=[('both_token_brier','normal_token_brier'),('both_mean_brier','both_token_brier'),
        ('mean_consistency','both_mean_brier'),('both_mean_brier','aligned_distill_supervised'),
        ('mean_consistency','aligned_distill_supervised'),('aligned_distill_supervised','aligned_distill')]
    paired=[]
    lookup={(r['method'],r['mapping']):r for r in metrics}
    for a,b in contrasts:
        for name in ('normal','reversed','normal_reversed_average','heldout_mapping_mean'):
            r=dict(comparison=a+'_minus_'+b,mapping=name)
            for k,m in enumerate(METRICS):
                delta=boot_metrics[a][name][:,k]-boot_metrics[b][name][:,k]; lo,hi=np.nanquantile(delta,[.025,.975])
                r[m]=lookup[a,name][m]-lookup[b,name][m];r[m+'_lo']=float(lo);r[m+'_hi']=float(hi)
            paired.append(r)
    analysis=out/'analysis'; analysis.mkdir(exist_ok=True)
    for name,rs in [('metrics',metrics),('stability',stability),('retention',decisions),('codebook',codebook),('paired_differences',paired)]: csv_save(analysis/f'{name}.csv',rs)
    write_json(analysis/'protocol.json',dict(bootstrap_draws=draws,bootstrap_seed=20260914,unit='question; paired across all methods and mappings',
        overlap='intersection size / ceil(retention*n); Jaccard is intersection/union',
        risk='mean(1-correct) in retained set; row ID breaks score ties without labels',
        numerical_disagreement='absolute decoded-mean difference; pairwise heldout uses all 28 pairs of the eight saved maps',
        upper_tail='95th percentile over question/mapping-pair absolute differences',
        scope='single checkpoint per method; finite saved mapping family; no permutation-population confidence claim'))
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
    for ax,method in zip(axes.flat,methods):
        a=scores[method]; ax.scatter(a[0],a[1],c=y,cmap='coolwarm_r',s=10,alpha=.6,vmin=0,vmax=1)
        ax.plot([0,1],[0,1],color='0.65',lw=.7); ax.set(title=method.replace('_',' '),xlim=(0,1),ylim=(0,1))
        ax.spines[['top','right']].set_visible(False)
    axes.flat[-1].set_visible(False); fig.supxlabel('Normal decoded correctness probability'); fig.supylabel('Reversed decoded correctness probability')
    fig.tight_layout();fig.savefig(analysis/'normal_reversed_scatter.png',dpi=180);plt.close(fig)
    text='# Matched letter11 objective pilot\n\n128 training and 128 test questions; single seed; frozen mappings and hyperparameters. Binary evidence remains in `full_llama31_8b_qwen3_8b/qa_average_verified`.\n\n'
    text+='| Method | Evaluation | Brier | AUROC | AURC |\n|---|---|---:|---:|---:|\n'
    for r in metrics:
        if r['mapping'] not in ('normal','reversed','normal_reversed_average','heldout_mapping_mean'):continue
        text+=f"| {r['method']} | {r['mapping']} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['aurc']:.4f} |\n"
    text+='\nAll individual and paired question-bootstrap intervals are in the CSVs. Held-out mapping mean averages metrics across the eight fixed mappings; it does not average predictions before scoring. No generalization to all 11! permutations is claimed.\n'
    (analysis/'REPORT.md').write_text(text)
    print(f'Analysis complete: {analysis}',flush=True)
