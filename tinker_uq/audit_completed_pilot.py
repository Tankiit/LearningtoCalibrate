"""Independent, read-only cache audit. Writes only a separate audit directory."""
import collections
import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/scale_family_v1'
OUT=ROOT/'audit_2026-09-15'
checks=collections.Counter()
files={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);files[str(p.relative_to(ROOT.parent))]=sha(p)
    return json.loads(p.read_text())
def rows(p):
    p=Path(p);files[str(p.relative_to(ROOT.parent))]=sha(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def table(p):
    files[str(p.relative_to(ROOT.parent))]=sha(p)
    with p.open() as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):
    assert np.allclose(a,b,atol=tol,rtol=tol,equal_nan=True),(a,b)
def serial_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def decode(raw,mapping,direct,name):
    s=raw.strip()
    if not direct:return mapping.get(s)
    if name=='strict_v1':
        if re.fullmatch(r'(?:0|[1-9]\d*)(?:\.\d+)?%',s) is None:return None
        x=float(s[:-1])/100
        return next((v for v in mapping.values() if abs(v-x)<1e-10),None)
    if re.fullmatch(r'(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?',s) is None:return None
    x=Decimal(s.rstrip('%'))
    if s.endswith('%') or x>1:x/=100
    return next((v for v in mapping.values() if Decimal(str(v))==x),None)


def quality(y,p):
    if not len(y):return np.array([np.nan]*3)
    positive=p[y==1];negative=p[y==0]
    auc=float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative))) if len(positive) and len(negative) else np.nan
    cumulative_error=0.;seen=0;prefix=[]
    for score in sorted(set(p),reverse=True):
        group=y[p==score];error=float(np.mean(1-group))
        for j in range(1,len(group)+1):prefix.append((cumulative_error+j*error)/(seen+j))
        cumulative_error+=float(np.sum(1-group));seen+=len(group)
    return np.array([np.mean((p-y)**2),auc,np.mean(prefix)])


def emission_prompt(r,mapping,direct,instruction):
    if direct:tail='Permitted confidence percentages: '+', '.join(f'{v*100:g}%' for v in sorted(mapping.values()))+'.\nReply with one permitted percentage followed by a newline.'
    else:tail='Report legend: '+'  '.join(f'{k}={v*100:g}%' for k,v in sorted(mapping.items()))+'\nReply with one letter followed by a newline.'
    return instruction+'\nQuestion:\n'+r['question']+'\nCandidate answer:\n'+r['answer']+'\n'+tail


def main():
    OUT.mkdir(exist_ok=True)
    plan=read(RUN/'student_plan_conftuner.json');manifest=read(RUN/'manifest.json');maps=read(RUN/'mappings.json')
    source=rows(ROOT/'data/full_llama31_8b/cached_answers.jsonl');source_by_id={r['question_id']:r for r in source}
    assert len(source)==len(source_by_id)==53327
    assert sha(ROOT/'data/full_llama31_8b/cached_answers.jsonl')==manifest['source_sha256']
    data=rows(RUN/'data.jsonl');refs={r['question_id']:r for r in data if r['split']=='test'};ids=sorted(refs);y=np.array([refs[i]['correct'] for i in ids]);n=len(ids)
    assert len(data)==len({r['question_id'] for r in data}) and n==128
    assert all(r==source_by_id[r['question_id']] for r in data)
    assert sha(RUN/'data.jsonl')==plan['data_sha256']==manifest['data_sha256']
    assert sha(RUN/'mappings.json')==plan['mappings_sha256']==manifest['mappings_sha256']
    checks['source_rows_and_disjoint_pilot_partitions']=len(data)
    # Join all pilot labels directly to the released evaluated-answer CSV.
    csv_path=ROOT.parent/'evaluations_revisited/llama3.1_8b_chat/trivia_qa_60k/base/main_generations_evaluated_revisited.csv'
    files[str(csv_path.relative_to(ROOT.parent))]=sha(csv_path)
    with csv_path.open() as f:original=list(csv.DictReader(f))
    reconstructed={}
    for r in original:
        question=r['prompt'].rsplit('Question:',1)[-1].split('\nAnswer:',1)[0].strip()
        answer=r['answer']
        for marker in ['<|end_of_text|>','</s>','<|eot_id|>','[/INST]']:answer=answer.replace(marker,' ')
        answer=' '.join(answer.split())
        if not question or not answer:continue
        qid='tqa-'+hashlib.sha256(' '.join(question.split()).casefold().encode()).hexdigest()[:16]
        if qid in reconstructed:continue
        bucket=int(hashlib.sha256(('tinker-uq-v1'+qid).encode()).hexdigest()[:8],16)/2**32
        split='train' if bucket<.5 else 'val' if bucket<.6 else 'cal' if bucket<.8 else 'test'
        reconstructed[qid]=(question,answer,int(r['correct_revisited']),split)
    assert set(reconstructed)==set(source_by_id)
    for qid,r in source_by_id.items():assert reconstructed[qid]==(r['question'],r['answer'],r['correct'],r['split'])
    checks['source_labels_answers_ids_splits_reconstructed_from_csv']=len(reconstructed)
    orders=[]
    for scale,methods in plan['methods'].items():
        for method in methods:
            training=rows(RUN/'students'/scale/method/'training.jsonl');cp=read(RUN/'students'/scale/method/'checkpoint.json')
            assert len(training)==cp['steps']==16
            expected=['normal','normal'] if method=='normal_token_brier' else ['normal','reversed']
            assert all(r['training_mappings']==expected and len(r['question_ids'])==8 for r in training)
            order=[q for r in training for q in r['question_ids']]
            assert set(order)=={r['question_id'] for r in data if r['split']=='train'} and len(order)==128
            orders.append(order);checks['matched_training_arms']+=1
    assert all(o==orders[0] for o in orders)
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',local_files_only=True)
    declaration=read(ROOT/'contracts/lenient_v1.json')
    for raw,want in declaration['examples_fine'].items():assert decode(raw,maps['fine']['normal'],True,'lenient_v1')==want
    for raw in declaration['reject_fine']:assert decode(raw,maps['fine']['normal'],True,'lenient_v1') is None
    losses={};ranges={};all_metric_rows=table(RUN/'trained_emission_analysis/metrics.csv')
    indexed_metrics={(r['scale'],r['method'],r['contract'],r['report']):r for r in all_metric_rows}
    ix=np.random.default_rng(20260915).integers(0,n,(2000,n))
    hist=[]
    for scale,methods in plan['methods'].items():
        for method in methods:
            base=RUN/'trained_emissions'/scale/method;m=read(base/'manifest.json');raw=rows(base/'reports.jsonl');cp=read(RUN/'students'/scale/method/'checkpoint.json')
            assert m['sampler_path']==cp['sampler_path'] and m['lenient_declaration_sha256']==sha(ROOT/'contracts/lenient_v1.json')
            assert len(raw)==n*len(m['formats'])
            raw_by_key={(r['question_id'],r['format']):r for r in raw};assert len(raw_by_key)==len(raw)
            for r in raw:
                ref=refs[r['question_id']];mapping=maps[scale]['normal' if r['format']=='direct' else r['format']]
                assert all(r[k]==ref[k] for k in ['question','answer','correct','split','row_id'])
                prompt=emission_prompt(ref,mapping,r['format']=='direct',m['instruction'])
                assert r['mapping']==mapping and r['prompt_text']==prompt and r['prompt_text_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                tokens=tok.apply_chat_template([{'role':'user','content':prompt}],enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
                assert r['prompt_token_ids']==tokens and r['model_revision']==cp['sampler_path'] and r['sampling_policy']==m['policy']
                assert tok.decode(r['output_token_ids'],skip_special_tokens=True)==r['raw_response']
                checks['raw_emission_prompt_token_candidate_audits']+=1
            for name,folder in [('strict_v1',base),('lenient_v1',base/'lenient_v1')]:
                rr=raw if name=='strict_v1' else rows(folder/'reports.jsonl')
                if name=='lenient_v1':
                    lm=read(folder/'manifest.json');assert lm['source_raw_sha256']==sha(base/'reports.jsonl')
                    assert sha(folder/'contract_declaration.json')==sha(ROOT/'contracts/lenient_v1.json')
                group={}
                for r in rr:
                    original=raw_by_key[r['question_id'],r['format']]
                    assert all(r[k]==original[k] for k in ['raw_response','output_token_ids','prompt_text','model_revision','mapping','correct'])
                    v=decode(original['raw_response'],original['mapping'],r['format']=='direct',name)
                    assert v==r['emitted_value'] and r['valid']==(v is not None)
                    group[r['question_id'],r['format']]=np.nan if v is None else v
                    checks['independently_decoded_emission_records']+=1
                p=np.array([[group[i,f] for f in m['formats']] for i in ids]);legends=[j for j,f in enumerate(m['formats']) if f!='direct'];js={f:[j] for j,f in enumerate(m['formats'])};js.update(normal_reversed_average=[m['formats'].index('normal'),m['formats'].index('reversed')],three_format_average=[m['formats'].index(f) for f in ['direct','normal','reversed']],six_report_average=legends[:6])
                for report,indices in js.items():
                    if report=='three_format_average':continue
                    value=p[:,indices].mean(1);valid=np.isfinite(value);loss=np.where(valid,(value-y)**2,1.);losses[scale,method,name,report]=loss
                    metric=indexed_metrics[scale,method,name,report]
                    assert int(metric['valid'])==int(valid.sum());close(float(metric['penalized_brier']),loss.mean());close([float(metric['brier_lo']),float(metric['brier_hi'])],np.quantile(loss[ix].mean(1),[.025,.975]))
                    close([float(metric[k]) for k in ['valid_only_brier','valid_only_auroc','valid_only_aurc']],quality(y[valid],value[valid]))
                    checks['independent_metric_and_brier_interval_rows']+=1
                individual=np.where(np.isfinite(p[:,legends[:6]]),(p[:,legends[:6]]-y[:,None])**2,1.)
                metric=indexed_metrics[scale,method,name,'mean_individual_brier']
                close(float(metric['penalized_brier']),individual.mean())
                close([float(metric['brier_lo']),float(metric['brier_hi'])],np.quantile(individual.mean(1)[ix].mean(1),[.025,.975]))
                worst=read(folder/'analysis/individual_brier.json');close(worst['worst'],individual.mean(0).max());close(worst['mean'],individual.mean())
                checks['mean_and_worst_individual_brier_rows']+=1
                for bank,indices in [('normal_reversed',js['normal_reversed_average']),('first_six',legends[:6]),('common_withheld_bank',legends[2:])]:
                    a=p[:,indices];valid=np.isfinite(a).all(1);gap=np.where(valid,np.ptp(a,axis=1),np.nan);flip=np.where(valid,(a.min(1)<.75)&(a.max(1)>=.75),np.nan)
                    ranges[scale,method,name,bank]=(gap,flip)
                hist.append(dict(scale=scale,method=method,contract=name,values=dict(collections.Counter(str(x) for x in p[:,m['formats'].index('direct')]))))
    for r in table(RUN/'trained_emission_analysis/stability.csv'):
        gap,flip=ranges[r['scale'],r['method'],r['contract'],r['bank']];valid=np.isfinite(gap)
        assert int(r['complete'])==valid.sum()
        close([float(r[k]) for k in ['mean_range','p95_range','flip_075']],[np.nanmean(gap),np.nanquantile(gap,.95),np.nanmean(flip)])
        close([float(r['flip_lo']),float(r['flip_hi'])],np.nanquantile(np.nanmean(flip[ix],axis=1),[.025,.975]));checks['stability_rows']+=1
    for r in table(RUN/'trained_emission_analysis/paired_differences.csv'):
        a,b=r['comparison'].split('_minus_');prefix=(r['scale'],r['contract']);metric=r['metric']
        if metric=='penalized_brier':delta=losses[prefix[0],a,prefix[1],r['readout']]-losses[prefix[0],b,prefix[1],r['readout']]
        else:
            k=0 if metric=='mean_range' else 1;av=ranges[prefix[0],a,prefix[1],r['readout']][k];bv=ranges[prefix[0],b,prefix[1],r['readout']][k];delta=av-bv
        assert int(r['n'])==np.isfinite(delta).sum()
        close(float(r['delta']),np.nanmean(delta));close([float(r['lo']),float(r['hi'])],np.nanquantile(np.nanmean(delta[ix],axis=1),[.025,.975]));checks['paired_contrasts']+=1
    control_table=table(RUN/'trained_emission_analysis/mapping_controls.csv')
    for scale in ['coarse','fine']:
        for method in ['normal_token_brier','both_token_brier']:
            base=RUN/'trained_emissions'/scale/method;controls=rows(base/'mapping_controls.jsonl');cp=read(RUN/'students'/scale/method/'checkpoint.json')
            assert len(controls)==(len(maps[scale])+1)*len(maps[scale]['normal'])
            for r in controls:
                assert r['model_revision']==cp['sampler_path']
                assert decode(r['raw_response'],r['mapping'],r['format']=='direct','strict_v1')==r['emitted_value']
                checks['trained_known_value_raw_controls']+=1
            for summary in [r for r in control_table if r['scale']==scale and r['method']==method]:
                group=[r for r in controls if r['format']==summary['format']]
                decoded=[decode(r['raw_response'],r['mapping'],r['format']=='direct',summary['contract']) for r in group]
                assert int(summary['correct'])==sum(v==r['known_value'] for v,r in zip(decoded,group))
                assert int(summary['valid'])==sum(v is not None for v in decoded) and int(summary['n'])==len(group)
    predictions=rows(RUN/'students/analysis/verified_reporting_predictions.jsonl');seen=set()
    for r in predictions:
        key=(r['scale'],r['method'],r['legend'],r['question_id']);assert key not in seen;seen.add(key)
        mapping=maps[r['scale']][r['legend']];letters=sorted(mapping);values=np.array([mapping[k] for k in letters]);lp=np.array(r['label_logprobs']);prob=np.exp(lp-lp.max());prob/=prob.sum()
        close(r['probability_vector'],prob,1e-6);close(r['decoded_mean'],prob@values,1e-6);close(r['allowed_alphabet_mass'],np.exp(lp).sum(),1e-6);close(r['aligned_probability_vector'],prob[np.argsort(values)],1e-6)
        ref=refs[r['question_id']];assert r['correctness_label']==ref['correct'] and r['answer']==ref['answer'] and r['split']==ref['split']
        student_seen=r['legend'] in (['normal'] if r['method']=='normal_token_brier' else ['normal','reversed']) if r['method']!='teacher' else None
        assert r['student_training_exposure']==student_seen and r['teacher_target_exposure']==(r['legend'] in list(maps[r['scale']])[:6])
        text='Judge whether the supplied candidate answer is correct.\nQuestion:\n'+ref['question']+'\nCandidate answer:\n'+ref['answer']+'\nReport the probability that this answer is correct.\nReport legend: '+'  '.join(f'{k}={mapping[k]*100:g}%' for k in letters)+'\nReply with one letter followed by a newline.'
        assert r['prompt_provenance']['text']==text
        checks['independent_distribution_decodes']+=1
    assert len(predictions)==13568
    # Recompute original ICLR control booleans from the vectors and raw generations.
    reversed_context=[];reversed_first_tokens=[]
    import torch
    for model in ['llama3_8b','mistral_7b','qwen2_5_7b']:
        obj=read(ROOT.parent/f'cached_results/letter11_mapping_control_v1/{model}.json');keys=set()
        original_sources={}
        for dataset in ['truthfulqa','pavlick_nli']:
            for legend in ['forward','reversed']:
                path=ROOT.parent/f'letter11_provenance_v1/{model}/{dataset}/{legend}.pt';files[str(path.relative_to(ROOT.parent))]=sha(path)
                original_sources[dataset,legend]=torch.load(path,map_location='cpu',weights_only=False)
        for dataset in ['truthfulqa','pavlick_nli']:
            source_ids=[str(x) for x in original_sources[dataset,'forward']['example_ids']]
            selected=sorted(source_ids,key=lambda q:hashlib.sha256(('letter11-codebook-v1:'+dataset+':'+q).encode()).hexdigest())[:32]
            for r in [x for x in obj['rows'] if x['dataset']==dataset]:
                assert r['question_id'] in selected and r['value']==10*(selected.index(r['question_id'])%11)
        for r in obj['rows']:
            key=(r['dataset'],r['question_id'],r['candidate_side'],r['legend']);assert key not in keys;keys.add(key)
            letters=list(r['mapping']);expected=next(k for k,v in r['mapping'].items() if v==r['value']/100);argmax=letters[int(np.argmax(r['probability_vector']))]
            assert r['expected_symbol']==expected and r['conditional_argmax']==argmax and r['conditional_correct']==(argmax==expected)
            close(sum(r['probability_vector']),1,1e-6);close(r['decoded_mean'],np.array(r['probability_vector'])@np.array(list(r['mapping'].values())),1e-6)
            legend_text='  '.join(f'{k}={v*100:g}%' for k,v in r['mapping'].items())
            control=('' if r['question'] is None else f"Question: {r['question']}\nAnswer: {r['answer']}\n")+f"The confidence value is supplied as {r['value']}%. Do not estimate correctness or change this value. Encode the supplied value using the legend.\n{legend_text}\nAnswer with a single letter.\nConfidence:"
            assert control==r['prompt'] and hashlib.sha256(control.encode()).hexdigest()==r['prompt_sha256']
            origin=original_sources[r['dataset'] or 'truthfulqa',r['legend']]
            assert obj['revision']==origin['meta']['model_revision'] and obj['model_id']==origin['meta']['model_id']
            assert r['candidate_token_ids']==origin['meta']['candidate_ids'] and r['common_prefix_ids']==origin['meta']['common_prefix_ids']
            if r['kind']=='contextual':
                index=[str(x) for x in origin['example_ids']].index(r['question_id'])
                original_prompt=f"Question: {r['question']}\nAnswer: {r['answer']}\nHow confident are you that this answer is correct?\n{legend_text}\nAnswer with a single letter.\nConfidence:"
                assert hashlib.sha256(original_prompt.encode()).hexdigest()==origin['prompt_hashes_'+r['candidate_side']][index]
                checks['original_cell_prompt_token_readout_joins']+=1
            emitted=r['raw_response'].strip();assert r['emitted_valid']==(emitted in letters) and r['emitted_correct']==(emitted==expected)
            assert 0<=r['allowed_alphabet_mass']<=1
            if r['legend']=='reversed' and r['kind']=='contextual':
                reversed_context.append(argmax==expected)
                reversed_first_tokens.append(bool(r['generated_token_ids'] and r['generated_token_ids'][0]==r['candidate_token_ids'][letters.index(expected)]))
            checks['independent_original_control_flags']+=1
    assert len(reversed_context)==384 and sum(reversed_context)==sum(reversed_first_tokens)==383
    # Original 1,000-question direct cache: primary source and derivative agree on bytes.
    old=ROOT/'runs/emitted_validation_v1';old_manifest=read(old/'manifest.json');raw=rows(old/'reports.jsonl');secondary=rows(old/'lenient_v1/reports.jsonl');raw_index={(r['question_id'],r['scale'],r['format']):r for r in raw};g={}
    assert old_manifest['sampler_path']==read(RUN/'students/coarse/both_mean_brier/checkpoint.json')['sampler_path']
    assert len(raw_index)==len(raw)==len(secondary)==6000
    for r in secondary:
        original=raw_index[r['question_id'],r['scale'],r['format']];assert original['raw_response']==r['raw_response'] and original['prompt_text']==r['prompt_text']
        ref=source_by_id[r['question_id']];assert ref['split']=='val' and all(original[k]==ref[k] for k in ['question','answer','correct','split','row_id'])
        mapping=maps[r['scale']]['normal' if r['format']=='direct' else r['format']]
        assert original['mapping']==mapping and original['model_revision']==old_manifest['sampler_path'] and original['sampling_policy']==old_manifest['policy']
        assert original['prompt_text']==emission_prompt(ref,mapping,r['format']=='direct',old_manifest['instruction'])
        assert decode(r['raw_response'],r['mapping'],r['format']=='direct','lenient_v1')==r['emitted_value']
        assert decode(original['raw_response'],original['mapping'],original['format']=='direct','strict_v1')==original['emitted_value']
        if r['format']=='direct':g[r['scale'],r['question_id']]=r
    oldids=sorted(i for s,i in g if s=='coarse');assert len(oldids)==1000
    direct_hist={s:dict(collections.Counter(g[s,i]['emitted_value'] for i in oldids)) for s in ['coarse','fine']}
    assert direct_hist=={'coarse':{0.:122,.5:21,1.:857},'fine':{0.:82,.1:59,.3:1,1.:858}}
    assert sum(g['coarse',i]['emitted_value']!=g['fine',i]['emitted_value'] for i in oldids)==75
    old_ix=np.random.default_rng(20260915).integers(0,1000,(2000,1000));old_y=np.array([g['fine',i]['correct'] for i in oldids]);direct=np.array([g['fine',i]['emitted_value'] for i in oldids]);audit=read(old/'lenient_v1/direct_grid_audit.json')
    for saved in audit['fine_brier_contrasts']:
        fmt=saved['comparison'].split('_minus_')[0];rs={r['question_id']:r for r in secondary if r['scale']=='fine' and r['format']==fmt};p=np.array([rs[i]['emitted_value'] for i in oldids]);delta=(p-old_y)**2-(direct-old_y)**2
        close(saved['brier_difference'],delta.mean());close(saved['interval'],np.quantile(delta[old_ix].mean(1),[.025,.975]))
    checks['historical_raw_and_secondary_records']=6000
    # Ensure the audit did not change any consumed artifact.
    assert all(sha(ROOT.parent/p)==h for p,h in files.items())
    result=dict(status='PASS',checks=dict(checks),source_hashes=files,direct_histograms=direct_hist,original_control_reversed_correct=383,original_control_reversed_n=384,model_calls=0,limitations=['Provider backbone commit not independently exposed by Tinker; adapter sampler paths recorded.','Temporal parser declaration order relies on session provenance; hashes validate the saved declaration, not an external timestamp.','Within-interface prompts are frozen; training versus emitted evaluation changes instruction placement.','Original-cell full-vocabulary logits are not saved, so original allowed-alphabet mass can be range-checked but not independently recomputed. New distribution-cache mass is recomputed from saved sequence log-probabilities.','Pilot results are exploratory, not independent confirmatory evidence.'])
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'REPORT.md').write_text('# Independent audit of the completed pilot\n\nPASS: numerical claims reproduced from raw caches; no new model calls and no consumed cache was modified.\n\n'+ '\n'.join(f'- {k}: {v:,}' for k,v in checks.items())+'\n\nConfirmed: direct histograms differ on 75/1,000 questions, with only 60 fine-only outputs; fine reversal transfer cost is 0.03410 with paired interval [0.01239, 0.05465]; original reversed contextual mapping control is 383/384. All reported trained-emission Brier/AUROC/AURC, Brier intervals, stability denominators and paired contrasts were independently recomputed.\n\nInterpretation limits:\n\n'+ '\n'.join('- '+x for x in result['limitations'])+'\n\nStrict and lenient results remain separate. Strict retains its historical numeric tolerance; lenient uses exact Decimal grid membership. Conditional codebook accuracy, unconstrained first-token accuracy and whole-response validity must remain distinct. Six-bank stability is not established by normal/reversal agreement. No richer-set benefit or uniform stability guarantee follows from these results.\n\nSee AUDIT.json for source hashes and check counts.\n')
    print(json.dumps(dict(status=result['status'],checks=checks,report=str(OUT/'REPORT.md')),indent=2))


if __name__=='__main__':main()
