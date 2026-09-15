"""Separate teacher quality, student imitation, and held-out legend sensitivity."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .data import load_jsonl,write_json
from .reporting import align_to_values
from .matched_analysis import verified_group,metric_vector,csv_save,retention,METRICS


def analyze(out,draws=2000,scales=None,plan_filename="student_plan.json"):
    out=Path(out);p=json.loads((out/plan_filename).read_text());m=json.loads((out/'manifest.json').read_text());maps=json.loads((out/'mappings.json').read_text())
    if scales is not None:p['methods']={k:v for k,v in p['methods'].items() if k in scales}
    refs={r['row_id']:r for r in load_jsonl(out/'data.jsonl') if r['split']=='test'};ids=sorted(refs);y=np.array([refs[i]['correct'] for i in ids]);n=len(ids)
    rng=np.random.default_rng(20260915);ixs=rng.integers(0,n,size=(draws,n))
    metricrows=[];imitation=[];sensitivity=[];decisions=[];paired=[];codebook=[];budgets=[];saved={};boots={};raw={}
    for scale,methods in p['methods'].items():
        family=maps[scale];names=list(family);values=np.array(sorted(family['normal'].values()));folder=out/'students'/scale
        distributions={};allmeans={}
        for method in ['teacher']+methods:
            gs=[verified_group(folder/method/f'{name}.jsonl',refs,family[name]) for name in names]
            rr=np.array([[align_to_values(g[i]['probability_vector'],g[i]['mapping']) for g in gs] for i in ids])
            distributions[method]=rr;means=rr@values;allmeans[method]=means;raw[scale,method]=means
            # Shared comparison bank excludes normal/reversed for every method;
            # arm-specific exposure is recorded separately below.
            pools={'normal':[0],'reversed':[1],'common_withheld_bank':list(range(2,len(names))),'first_six':list(range(6))}
            pools.update({'legend_'+name:[j] for j,name in enumerate(names)})
            if scale=='fine':pools['fresh_beyond_teacher']=list(range(6,10))
            for name,indices in pools.items():
                permap=np.array([metric_vector(y,means[:,j]) for j in indices]);obs=permap.mean(0)
                b=np.array([np.mean([metric_vector(y[ix],means[ix,j]) for j in indices],axis=0) for ix in ixs]);boots[scale,method,name]=b;saved[scale,method,name]=obs
                r=dict(scale=scale,method=method,evaluation=name,n_questions=n,n_mappings=len(indices))
                for k,metric in enumerate(METRICS):
                    r[metric]=float(obs[k]);lo,hi=np.nanquantile(b[:,k],[.025,.975]);r[metric+'_lo']=float(lo);r[metric+'_hi']=float(hi)
                metricrows.append(r)
            for name,score in [('normal_reversed_average',means[:,:2].mean(1)),('six_report_average',means[:,:6].mean(1))]:
                obs=metric_vector(y,score);b=np.array([metric_vector(y[ix],score[ix]) for ix in ixs]);boots[scale,method,name]=b;saved[scale,method,name]=obs
                r=dict(scale=scale,method=method,evaluation=name,n_questions=n,n_mappings=2 if name.startswith('normal') else 6)
                for k,metric in enumerate(METRICS):
                    r[metric]=float(obs[k]);lo,hi=np.nanquantile(b[:,k],[.025,.975]);r[metric+'_lo']=float(lo);r[metric+'_hi']=float(hi)
                metricrows.append(r)
            for pool,indices in [('first_six',list(range(6))),('common_withheld_bank',list(range(2,len(names))))]:
                diffs=np.array([np.abs(means[:,a]-means[:,b]) for ia,a in enumerate(indices) for b in indices[ia+1:]])
                qs=np.ptp(means[:,indices],axis=1)
                r=dict(scale=scale,method=method,evaluation=pool,mean_abs_disagreement=float(diffs.mean()),p95_abs_disagreement=float(np.quantile(diffs,.95)),mean_range=float(qs.mean()),p95_range=float(np.quantile(qs,.95)))
                for key,calc in [('mean_abs_disagreement',lambda ix:diffs[:,ix].mean()),('p95_abs_disagreement',lambda ix:np.quantile(diffs[:,ix],.95))]:
                    lo,hi=np.quantile([calc(ix) for ix in ixs],[.025,.975]);r[key+'_lo']=float(lo);r[key+'_hi']=float(hi)
                r['normal_reversed_spearman']=float(spearmanr(means[:,0],means[:,1]).statistic) if np.std(means[:,0])>0 and np.std(means[:,1])>0 else None
                r['normal_reversed_threshold_flip_075']=float(np.mean((means[:,0]>=.75)!=(means[:,1]>=.75)))
                r['family_ambiguous_075']=float(np.mean((means[:,indices].min(1)<.75)&(means[:,indices].max(1)>=.75)))
                sensitivity.append(r)
            for fraction in (.2,.5,.8,.9):
                for j,name in enumerate(names[1:],1):
                    decisions.append(dict(scale=scale,method=method,comparison='normal_vs_'+name,**retention(y,means[:,0],means[:,j],ids,fraction)))
                for approach,score in [('six_report_average',means[:,:6].mean(1)),('family_lower',means[:,:6].min(1))]:
                    chosen=np.lexsort((np.asarray(ids),-score))[:int(np.ceil(n*fraction))]
                    decisions.append(dict(scale=scale,method=method,comparison=approach,retention=fraction,n_retained=len(chosen),intersection_fraction=None,jaccard=None,
                        normal_risk=float(np.mean(1-y[chosen])),reversed_risk=None,risk_delta=None))
            if method!='teacher':
                cb=json.loads((folder/method/'codebook.json').read_text())
                if len(cb)!=len(names)*len(values):raise ValueError('Incomplete codebook scores')
                trained_names=['normal'] if method=='normal_token_brier' else ['normal','reversed']
                for kind,ns in [('trained',trained_names),('withheld',[x for x in names if x not in trained_names])]:
                    selected=[r for r in cb if r['mapping_name'] in ns]
                    codebook.append(dict(scale=scale,method=method,evaluation=kind,n=len(selected),conditional_argmax_accuracy=float(np.mean([r['codebook_correct'] for r in selected])),allowed_mass_mean=float(np.mean([r.get('allowed_alphabet_mass',1.) for r in selected])),unconditional_correct_sequence_probability=float(np.mean([r.get('allowed_alphabet_mass',1.)*r['expected_letter_probability'] for r in selected])),expected_letter_probability=float(np.mean([r['expected_letter_probability'] for r in selected])),numerical_error=float(np.mean([r['numerical_absolute_error'] for r in selected]))))
                logs=load_jsonl(folder/method/'training.jsonl')
                if len(logs)!=16:raise ValueError('Unmatched optimization budget')
                budgets.append(dict(scale=scale,method=method,steps=len(logs),questions=sum(len(r['question_ids']) for r in logs),sequences=sum(r['sequences'] for r in logs),input_tokens=sum(r['input_tokens'] for r in logs),question_order_sha256=__import__('hashlib').sha256(json.dumps([r['question_ids'] for r in logs]).encode()).hexdigest()))
        teacher=distributions['teacher'][:,:6].mean(1)
        # Imitation is measured against the frozen mixture for each test question.
        for method in methods:
            rr=distributions[method];kl=(teacher[:,None,:]*(np.log(np.maximum(teacher[:,None,:],1e-30))-np.log(np.maximum(rr,1e-30)))).sum(-1)
            tv=np.abs(rr-teacher[:,None,:]).sum(-1)/2;meanerr=np.abs((rr@values)-(teacher@values)[:,None])
            trained_indices=[0] if method=='normal_token_brier' else [0,1]
            for kind,index in [('trained',trained_indices),('withheld',[j for j in range(len(names)) if j not in trained_indices])]:
                imitation.append(dict(scale=scale,method=method,evaluation=kind,mean_kl=float(kl[:,index].mean()),mean_tv=float(tv[:,index].mean()),mean_abs_teacher_mean_error=float(meanerr[:,index].mean())))
        contrasts=[('aligned_distill_supervised','aligned_distill'),('both_mean_brier','aligned_distill_supervised')]
        if scale=='coarse' and 'both_token_brier' in methods:contrasts += [('both_token_brier','normal_token_brier'),('both_mean_brier','both_token_brier')]
        if scale=='fine':contrasts += [('both_token_brier','normal_token_brier'),('both_mean_brier','both_token_brier'),('mean_consistency','both_mean_brier'),('mean_consistency','aligned_distill_supervised')]
        for a,b in contrasts:
            for group in ('normal','reversed','normal_reversed_average','six_report_average','common_withheld_bank'):
                obs=saved[scale,a,group]-saved[scale,b,group];bb=boots[scale,a,group]-boots[scale,b,group];r=dict(scale=scale,comparison=a+'_minus_'+b,evaluation=group)
                for k,metric in enumerate(METRICS):
                    r[metric]=float(obs[k]);lo,hi=np.nanquantile(bb[:,k],[.025,.975]);r[metric+'_lo']=float(lo);r[metric+'_hi']=float(hi)
                paired.append(r)
    if len({r['question_order_sha256'] for r in budgets})!=1:raise ValueError('Training question orders differ')
    scale_comparison=[]
    for method in (['teacher']+p['methods']['coarse'] if {'coarse','fine'} <= set(p['methods']) else []):
        def differences(scale):
            a=raw[scale,method][:,:6];return np.array([np.abs(a[:,i]-a[:,j]) for i in range(6) for j in range(i+1,6)])
        fine,coarse=differences('fine'),differences('coarse')
        row=dict(method=method,comparison='fine_minus_coarse_six_call_disagreement')
        for name,fn in [('mean',np.mean),('p95',lambda x:np.quantile(x,.95))]:
            row[name]=float(fn(fine)-fn(coarse));lo,hi=np.quantile([fn(fine[:,ix])-fn(coarse[:,ix]) for ix in ixs],[.025,.975]);row[name+'_lo']=float(lo);row[name+'_hi']=float(hi)
        scale_comparison.append(row)
    analysis=out/('students/analysis' if scales is None else 'students/analysis_'+'_'.join(scales));analysis.mkdir(exist_ok=True)
    with (analysis/'verified_test_predictions.jsonl').open('w') as stream:
        for scale,methods in p['methods'].items():
            for method in ['teacher']+methods:
                for name in maps[scale]:
                    for r in load_jsonl(out/'students'/scale/method/f'{name}.jsonl'):
                        stream.write(json.dumps(dict(r,scale=scale,method=method))+'\n')
    for name,rows in [('metrics',metricrows),('imitation',imitation),('sensitivity',sensitivity),('retained_risk',decisions),('paired_differences',paired),('codebook',codebook),('budget_audit',budgets),('scale_comparison',scale_comparison)]:csv_save(analysis/f'{name}.csv',rows)
    write_json(analysis/'protocol.json',dict(scales=list(p['methods']),primary_readout='decoded expectation, not actual emitted confidence',test_status='Exploratory pilot: coarse correctness-only test results informed the subsequent research direction; no objective selection on these test questions.',teacher_exposure='All six coarse mappings used in teacher targets; normal-only token Brier uses normal; other students use normal/reversed inputs. No globally unseen coarse mapping. The common_withheld_bank excludes both normal and reversal for every arm.',bootstrap_draws=draws,unit='paired test question',seed=20260915,
        teacher_quality='Brier/AUROC/AURC of frozen six-report mixture expectation',student_imitation='KL/TV and absolute mean error to that value-aligned mixture',
        invariance='numerical mean and p95 pairwise disagreement among withheld student mappings',
        scale_comparison='same questions/checkpoint/call count; six coarse exhaustive vs six fine approximate; finite family only',
        supervision='all students inherit same correctness-trained checkpoint; supervised distillation adds same mean-Brier loss as correctness-only control'))
    text='# Matched scale and objective student pilot\n\nAll students start from the same existing QA checkpoint, with fresh optimizers and matched 128-question/16-update budgets. Pure distillation has no additional label term; supervised distillation and correctness-only training receive identical correctness targets. Teacher averages use six reports (coarse exhaustive, fine approximate).\n\n| Scale | Method | Evaluation | Brier | AUROC | AURC |\n|---|---|---|---:|---:|---:|\n'
    for r in metricrows:
        if r['evaluation'] not in ('six_report_average','common_withheld_bank'):continue
        text+=f"| {r['scale']} | {r['method']} | {r['evaluation']} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['aurc']:.4f} |\n"
    text+='\nThese are decoded-expectation results, not emitted numerical confidence. The coarse correctness-only test summary informed the research direction; this is an exploratory pilot, not independent confirmation. Four coarse mappings are unseen by every student; the normal-only token-Brier student also has not trained on reversal. All six enter the coarse teacher targets. The common withheld bank excludes normal and reversal for matched comparisons.\n'
    text+='\nTeacher correctness information, student imitation and fresh legend sensitivity are separate outputs. All intervals condition on the fitted checkpoint and the saved mappings. No larger-alphabet failure claim follows from alphabet size alone. See paired differences and scale comparison before attributing improvements to the objective.\n'
    (analysis/'REPORT.md').write_text(text);print(f'Student analysis complete: {analysis}',flush=True)
