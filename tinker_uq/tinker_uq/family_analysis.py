"""Finite reporting families: averages, elementary disagreement, and richer shape controls."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .data import load_jsonl,write_json,write_jsonl
from .reporting import align_to_values,digest
from .matched_analysis import verified_group,metric_vector,csv_save,METRICS

RSUQ=Path('/Users/tanmoy/research/Credal_Sets/Random_Sets/rsuq/rsuq')

def rsuq_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def family_summary(r,values):
    """r: question x mapping x ordered numerical value; hull vertices retained."""
    if r.ndim!=3 or not np.allclose(r.sum(-1),1) or np.any(r<0):raise ValueError('Invalid aligned distribution family')
    means=r@values; mixture=r.mean(1); low=means.min(1);high=means.max(1)
    entropy=rsuq_module('rsuq_signals',RSUQ/'core/signals.py').pignistic_entropy
    import torch
    hm=entropy(torch.tensor(mixture)).numpy(); hg=entropy(torch.tensor(r)).numpy().mean(1)
    cdf=np.cumsum(r,axis=-1)[...,:-1];cdf_width=np.ptp(cdf,axis=1)
    tv=np.max([np.abs(r[:,i]-r[:,j]).sum(-1)/2 for i in range(r.shape[1]) for j in range(i+1,r.shape[1])],axis=0)
    return dict(means=means,mixture=mixture,mean=means.mean(1),lower=low,upper=high,range=high-low,
        variance=means.var(1),mixture_entropy=hm,js_disagreement=hm-hg,tv_diameter=tv,
        cdf_width_area=cdf_width@np.diff(values),cdf_width=cdf_width)


def features(s):
    mean=s['mean'][:,None]
    elementary=np.column_stack([s[k] for k in ('mean','range','variance','lower','upper')])
    return {'fitted_mean':mean,'fitted_mixture':s['mixture'],
        'fitted_mean_range_variance':np.column_stack([s[k] for k in ('mean','range','variance')]),
        'fitted_elementary_extrema':elementary,
        'fitted_mixture_plus_elementary':np.column_stack([elementary,s['mixture']]),
        'fitted_family_shape':np.column_stack([elementary,s['mixture'],s['mixture_entropy'],s['js_disagreement'],s['tv_diameter'],s['cdf_width_area'],s['cdf_width']])}


def predictions(s,y,fit,test,values):
    out={'normal_report':s['means'][:,0],'two_orientation_average':s['means'][:,:2].mean(1),'average_confidence':s['mean'],'mixture_expectation':s['mixture']@values,
         'mean_minus_range':np.clip(s['mean']-s['range'],0,1),
         'mean_minus_sd':np.clip(s['mean']-np.sqrt(s['variance']),0,1),'family_lower_mean':s['lower']}
    for name,x in features(s).items():
        if len(set(y[fit]))<2:raise ValueError('Fit half lacks one correctness class')
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=0))
        model.fit(x[fit],y[fit]);out[name]=model.predict_proba(x)[:,1]
    return out


def top(p,ids,f):return np.lexsort((np.asarray(ids),-p))[:max(1,int(np.ceil(len(p)*f)))]


def example(out):
    v=np.array([.6,1.]);stable=np.array([[[.5,.5],[.5,.5]]]);unstable=np.array([[[1.,0.],[0.,1.]]])
    a=family_summary(stable,v);b=family_summary(unstable,v)
    assert np.array_equal(a['mixture'],b['mixture']) and np.allclose(a['mean'],b['mean'])
    write_json(out/'information_loss_example.json',dict(values=v.tolist(),threshold=.75,
        stable=dict(family=stable[0].tolist(),mixture=a['mixture'][0].tolist(),mean=float(a['mean'][0]),interval=[float(a['lower'][0]),float(a['upper'][0])],all_accept=True),
        unstable=dict(family=unstable[0].tolist(),mixture=b['mixture'][0].tolist(),mean=float(b['mean'][0]),interval=[float(b['lower'][0]),float(b['upper'][0])],all_accept=False),
        interpretation='Identical mixtures erase the generating disagreement. Min/max means exactly determine all linear confidence-threshold decisions; no CI interpretation.'))


def analyze(out,draws=2000):
    out=Path(out);folder=out/'analysis';folder.mkdir(exist_ok=True)
    example(folder)
    manifest=json.loads((out/'manifest.json').read_text());maps=json.loads((out/'mappings.json').read_text())
    refs={r['row_id']:r for r in load_jsonl(out/'data.jsonl') if r['split']=='val'};ids=sorted(refs)
    y=np.array([refs[i]['correct'] for i in ids]);ordering=sorted(range(len(ids)),key=lambda j:digest(['selection-fit-v1',refs[ids[j]]['question_id']]))
    fit=np.array(ordering[:len(ids)//2]);test=np.array(ordering[len(ids)//2:]);testids=[ids[j] for j in test]
    rng=np.random.default_rng(20260915);boot=rng.integers(0,len(test),size=(draws,len(test)))
    metrics=[];risk=[];stability=[];numerical=[];codebook=[];paired=[];families={};saved_predictions=[]
    rs_deferral=rsuq_module('rsuq_deferral',RSUQ/'deferral.py')
    for scale,family in maps.items():
        names=manifest['initial_mapping_names'][scale]
        groups=[verified_group(out/scale/f'{name}.jsonl',refs,family[name]) for name in names]
        values=np.array(sorted(family['normal'].values()))
        r=np.array([[align_to_values(g[i]['probability_vector'],g[i]['mapping']) for g in groups] for i in ids])
        families[scale]=r;s=family_summary(r,values);s2=family_summary(r[:,:2],values)
        pred=predictions(s,y,fit,test,values);pred2=predictions(s2,y,fit,test,values)
        # Algebraic guard: the family threshold rule equals the elementary interval rule.
        for threshold in manifest['thresholds']:
            assert np.array_equal(np.all(s['means']>=threshold,axis=1),s['lower']>=threshold)
            stability.append(dict(scale=scale,kind='threshold',threshold=threshold,
                average_decision_flip=float(np.mean((s['mean'][test]>=threshold)!=(s2['mean'][test]>=threshold))),
                initially_all_accept_fraction=float(np.mean(s2['lower'][test]>=threshold)),
                previously_supported_accept_withdrawn_fraction=float(np.mean((s2['lower'][test]>=threshold)&(s['lower'][test]<threshold))),
                family_ambiguous_initial=float(np.mean((s2['lower'][test]<threshold)&(s2['upper'][test]>=threshold))),
                family_ambiguous_final=float(np.mean((s['lower'][test]<threshold)&(s['upper'][test]>=threshold))),
                evaluated_legend_disagreement=float(np.mean(np.ptp((s['means'][test]>=threshold).astype(int),axis=1)>0))))
        mean_d=np.abs(s['means'][:,:,None]-s['means'][:,None,:])[:,np.triu_indices(6,1)[0],np.triu_indices(6,1)[1]]
        numerical.append(dict(scale=scale,n_questions=len(ids),reports_per_question=6,mean_pairwise_disagreement=float(mean_d.mean()),p95_pairwise_disagreement=float(np.quantile(mean_d,.95)),
            mean_range=float(s['range'].mean()),p95_range=float(np.quantile(s['range'],.95)),exhaustive=scale=='coarse',
            alphabet_mass_mean=float(np.mean([g[i]['allowed_alphabet_mass'] for g in groups for i in ids]))))
        for name,p in pred.items():
            yy=y[test];pp=p[test];vals=metric_vector(yy,pp);bv=np.array([metric_vector(yy[ix],pp[ix]) for ix in boot])
            rec=dict(scale=scale,approach=name,n_fit=len(fit),n_eval=len(test),deployment_report_prompts=1 if name=='normal_report' else 2 if name=='two_orientation_average' else 6)
            for k,m in enumerate(METRICS):
                rec[m]=float(vals[k]);lo,hi=np.nanquantile(bv[:,k],[.025,.975]);rec[m+'_lo']=float(lo);rec[m+'_hi']=float(hi)
            rec['rsuq_grid_aurc']=float(rs_deferral.risk_coverage(-pp,1-yy)[2]);metrics.append(rec)
            for frac in manifest['retentions']:
                chosen=top(pp,testids,frac);old=top(pred2[name][test],testids,frac)
                rb=[]
                for ix in boot:
                    selected=top(pp[ix],[f'{testids[j]}:{k}' for k,j in enumerate(ix)],frac);rb.append(float(np.mean(1-yy[ix][selected])))
                lo,hi=np.quantile(rb,[.025,.975]);risk.append(dict(scale=scale,approach=name,retention=frac,n_retained=len(chosen),risk=float(np.mean(1-yy[chosen])),risk_lo=float(lo),risk_hi=float(hi),
                    first2_vs_all6_overlap=len(set(chosen)&set(old))/len(chosen)))
            for j in test:saved_predictions.append(dict(scale=scale,approach=name,row_id=ids[j],question_id=refs[ids[j]]['question_id'],correct=int(y[j]),score=float(p[j]),first_two_score=float(pred2[name][j])))
        for a,b in [('family_lower_mean','average_confidence'),('family_lower_mean','mean_minus_range'),('fitted_family_shape','fitted_elementary_extrema'),('fitted_family_shape','fitted_mean_range_variance'),('fitted_mixture','fitted_mean'),('fitted_family_shape','fitted_mixture'),('fitted_family_shape','fitted_mixture_plus_elementary')]:
            pa,pb=pred[a][test],pred[b][test];yy=y[test]
            observed=metric_vector(yy,pa)-metric_vector(yy,pb)
            bs=np.array([metric_vector(yy[ix],pa[ix])-metric_vector(yy[ix],pb[ix]) for ix in boot])
            rec=dict(scale=scale,comparison=a+'_minus_'+b)
            for k,m in enumerate(METRICS):
                rec[m]=float(observed[k]);lo,hi=np.nanquantile(bs[:,k],[.025,.975]);rec[m+'_lo']=float(lo);rec[m+'_hi']=float(hi)
            paired.append(rec)
        for j,rid in enumerate(ids):
            row=dict(row_id=rid,question_id=refs[rid]['question_id'],correct=int(y[j]),split='val',selection_fold='fit' if j in set(fit) else 'eval',scale=scale,
                values=values.tolist(),mapping_names=names,aligned_family=r[j].tolist(),mixture=s['mixture'][j].tolist(),mean=float(s['mean'][j]),lower=float(s['lower'][j]),upper=float(s['upper'][j]),range=float(s['range'][j]),variance=float(s['variance'][j]))
            saved_predictions.append(dict(approach='family_artifact',**row))
        cb=json.loads((out/scale/'codebook.json').read_text())
        if len(cb)!=6*len(values):raise ValueError('Missing codebook controls')
        codebook.append(dict(scale=scale,n=len(cb),allowed_mass_mean=float(np.mean([q['allowed_alphabet_mass'] for q in cb])),unconditional_correct_sequence_probability=float(np.mean([q['allowed_alphabet_mass']*q['expected_letter_probability'] for q in cb])),conditional_argmax_accuracy=float(np.mean([q['codebook_correct'] for q in cb])),expected_letter_probability=float(np.mean([q['expected_letter_probability'] for q in cb])),mean_numerical_error=float(np.mean([q['numerical_absolute_error'] for q in cb]))))
    for name,rows in [('metrics',metrics),('retained_risk',risk),('decision_stability',stability),('numerical_disagreement',numerical),('codebook',codebook),('paired_differences',paired)]:csv_save(folder/f'{name}.csv',rows)
    write_jsonl(folder/'family_and_selection_predictions.jsonl',saved_predictions)
    write_json(folder/'analysis_protocol.json',dict(fit_ids=[ids[i] for i in fit],eval_ids=testids,bootstrap=draws,
        bootstrap_scope='paired evaluation-question percentile bootstrap; fitted selectors held fixed; does not include fit/checkpoint variation',
        rsuq_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RSUQ/'core/signals.py',RSUQ/'deferral.py')},
        reuse='rsuq pignistic_entropy for Shannon entropy of aligned mixtures/reports; risk_coverage as explicitly secondary grid-integrated AURC',
        no_misidentification='rsuq cluster-size credal_width is not this finite legend-family width and is not used',
        family_decision='all means >= threshold iff lower mean >= threshold; exactly an elementary-extrema decision',
        mixture_expectation='identical to averaged confidence; mixture shape can add value only with an explicitly defined downstream rule',
        learned_baselines='same fit/eval questions, standard scaling and logistic regression C=1; no test tuning; richer family includes elementary features',
        sequential_stability='first two maps versus all six from the same acquisition; no seventh coarse permutation exists',
        scoring='Brier of heuristic scores is descriptive; min/penalized scores are selection scores, not calibrated probability claims'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,scale in zip(axes,('coarse','fine')):
        s=family_summary(families[scale],np.array(sorted(maps[scale]['normal'].values())))
        order=np.argsort(s['mean'][test]);xx=np.arange(len(test));ix=test[order]
        ax.vlines(xx,s['lower'][ix],s['upper'][ix],color='0.6',lw=1);ax.scatter(xx,s['mean'][ix],c=y[ix],cmap='coolwarm_r',s=12,vmin=0,vmax=1)
        ax.axhline(.75,color='0.3',ls='--',lw=.8);ax.set(title=scale,xlabel='Evaluation questions ordered by mean',ylim=(0,1));ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean and range across six legends')
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([],[],marker='o',ls='',color=plt.cm.coolwarm_r(1.),label='Correct answer'),Line2D([],[],marker='o',ls='',color=plt.cm.coolwarm_r(0.),label='Incorrect answer')],loc='lower right',frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(folder/'legend_intervals.png',dpi=180);plt.close(fig)
    text='# Fixed-checkpoint scale and reporting-family comparison\n\nSame existing QA checkpoint, fixed Llama answers and six calls per validation question per scale. Three-level average enumerates all permutations; eleven-level average uses identity, reversal and four frozen random mappings.\n\n'
    text+='## Information lost by averaging\n\nBoth illustrative teachers average to 0.5 mass at 0.6 and 0.5 at 1, with mean 0.8. Stable family means are [0.8,0.8]; unstable means are [0.6,1]. At 0.75 the first unanimously accepts and the second disagrees. Perfect mixture imitation cannot recover the generating disagreement. These are evaluated-legend sensitivity intervals, not confidence intervals for correctness.\n\n'
    text+=f'## Evaluation\n\n{len(ids)} validation questions; {len(fit)} fixed questions fit every supervised selector and the other {len(test)} evaluate it. All methods share the same calls. Brier/AURC confidence intervals condition on the fitted selectors.\n\n| Scale | Approach | Brier | AUROC | AURC |\n|---|---|---:|---:|---:|\n'
    for r in metrics:text+=f"| {r['scale']} | {r['approach']} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['aurc']:.4f} |\n"
    text+='\nFull paired intervals, matched-coverage risks, set overlap after four additional legends, numerical disagreement and codebook controls are in the adjacent CSVs. For linear threshold decisions the hull and its mean interval are exactly equivalent; improved lower-bound selection alone cannot establish a richer random-set geometry benefit. Learned family-shape features must improve over the elementary-extrema baseline to support an additional-information claim. This is a small single-checkpoint validation experiment, not a general scale-failure result.\n'
    (folder/'REPORT.md').write_text(text);print(f'Family analysis: {folder}',flush=True)
