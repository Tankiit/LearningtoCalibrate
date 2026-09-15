"""Rebuild candidate-label decision results only from fresh provenance-v1 caches."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from tinker_uq.reporting import SCHEMA,mapping_items
from tinker_uq.matched_analysis import metric_vector,csv_save,retention,pair_stability,METRICS
from tinker_uq.data import write_json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from data.schema import stable_id
from data.adapters.pavlick_nli import load as load_pavlick


def records(dataset):
    if dataset=='pavlick_nli':
        rs=load_pavlick();return {str(r.example_id):dict(question=r.question,pos=r.correct_answer,neg=r.wrong_answer) for r in rs}
    path=Path('/Users/tanmoy/.cache/huggingface/datasets/truthful_qa/multiple_choice/0.0.0/741b8276f2d1982aa3d5b832d3ee81ed3b896490/truthful_qa-validation.arrow')
    data=Dataset.from_file(str(path));out={}
    for r in data:
        choices=r['mc1_targets']['choices'];labels=r['mc1_targets']['labels']
        pos=[c for c,y in zip(choices,labels) if y==1];neg=[c for c,y in zip(choices,labels) if y==0]
        if pos and neg:out[stable_id('truthfulqa',r['question'])]=dict(question=r['question'],pos=pos[0],neg=neg[0])
    return out


def main():
    out=ROOT/'cached_results/fresh_candidate_correctness';out.mkdir(exist_ok=True)
    results=[];stable=[];decision=[];audit=[];comparisons=[]
    with (out/'verified_predictions.jsonl').open('w') as f:
        for dataset in ('truthfulqa','pavlick_nli'):
            ref=records(dataset)
            for model in ('llama3_8b','mistral_7b','qwen2_5_7b'):
                arrays=[];ordered_ids=None
                for legend in ('forward','reversed'):
                    path=ROOT/'letter11_provenance_v1'/model/dataset/f'{legend}.pt'
                    d=torch.load(path,map_location='cpu',weights_only=False);meta=d['meta']
                    assert meta['artifact']=='letter11_provenance_v1' and meta['legacy_artifact_used'] is False
                    ids=list(map(str,d['example_ids']));assert len(ids)==len(set(ids)) and set(ids)==set(ref)
                    if ordered_ids is None:ordered_ids=sorted(ids)
                    assert set(ordered_ids)==set(ids)
                    index={x:i for i,x in enumerate(ids)}
                    values=np.asarray(d['conf_values'],float)/100;mapping=dict(zip(meta['letters'],values.tolist()));mapping_items(mapping)
                    mat=[];maxdiff=0.
                    for qid in ordered_ids:
                        i=index[qid];scores=[]
                        for side,y in [('pos',1),('neg',0)]:
                            p=np.asarray(d[f'Vdist_{side}'][i],float);assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-5
                            p/=p.sum();mu=float(p@values);maxdiff=max(maxdiff,abs(mu-float(d[f'V_{side}'][i])))
                            mass=float(d[f'first_token_letter_mass_{side}'][i]);assert 0<=mass<=1.00001
                            text=meta['prompt_template'].format(q=ref[qid]['question'],a=ref[qid][side],legend='  '.join(f'{k}={v}%' for k,v in zip(meta['letters'],d['conf_values'])))
                            ph=hashlib.sha256(text.encode()).hexdigest();assert ph==str(d[f'prompt_hashes_{side}'][i]),(dataset,model,qid,side)
                            row=dict(schema_version=SCHEMA,row_id=f'{qid}:{side}',question_id=qid,dataset=dataset,model=model,arm='qa',stage='frozen',legend=legend,
                                correct=y,outcome_semantics='dataset_mc1_candidate_label' if dataset=='truthfulqa' else 'sign_of_mean_human_NLI_judgment_convention',
                                population='one curated positive and one curated negative candidate per question; not generated-answer accuracy',
                                mapping=mapping,letters=meta['letters'],probability_vector=p.tolist(),decoded_mean=mu,p_correct=mu,
                                allowed_alphabet_mass=mass,distribution_semantics=meta['distribution_semantics'],mass_semantics=meta['mass_semantics'],
                                prompt_provenance=dict(text=text,text_sha256=ph,model_revision=meta['model_revision'],candidate_token_ids=meta['candidate_ids'],common_prefix_ids=meta['common_prefix_ids']),
                                source=str(path.relative_to(ROOT)))
                            f.write(json.dumps(row)+'\n');scores.append(mu)
                        mat.append(scores)
                    arrays.append(np.array(mat));audit.append(dict(model=model,dataset=dataset,legend=legend,questions=len(ids),all_prompt_hashes_matched=True,max_decoded_error=maxdiff,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    assert maxdiff<1e-6
                normal,reverse=arrays;means=(normal+reverse)/2;y=np.tile([1,0],len(ordered_ids));rowids=[f'{qid}:{side}' for qid in ordered_ids for side in ('pos','neg')]
                rng=np.random.default_rng(20260915);draws=rng.integers(0,len(ordered_ids),size=(2000,len(ordered_ids)))
                boot_indices=(draws[:,:,None]*2+np.arange(2)).reshape(2000,-1)
                bs={};observed={}
                for name,mat in [('normal',normal),('reversed',reverse),('averaged',means)]:
                    score=mat.ravel();v=metric_vector(y,score);b=np.array([metric_vector(y[ix],score[ix]) for ix in boot_indices]);bs[name]=b;observed[name]=v
                    r=dict(model=model,dataset=dataset,score=name,questions=len(ordered_ids),candidates=len(y))
                    for k,m in enumerate(METRICS):
                        r[m]=float(v[k]);lo,hi=np.quantile(b[:,k],[.025,.975]);r[m+'_lo']=float(lo);r[m+'_hi']=float(hi)
                    results.append(r)
                stable.append(dict(model=model,dataset=dataset,**pair_stability(y,normal.ravel(),reverse.ravel(),rowids)))
                for frac in (.2,.5,.8,.9):
                    r=dict(model=model,dataset=dataset,**retention(y,normal.ravel(),reverse.ravel(),rowids,frac))
                    boots=[retention(y[ix],normal.ravel()[ix],reverse.ravel()[ix],[f'{rowids[j]}:{k}' for k,j in enumerate(ix)],frac) for ix in boot_indices]
                    for key in ('normal_risk','reversed_risk','risk_delta','intersection_fraction'):
                        lo,hi=np.quantile([b[key] for b in boots],[.025,.975]);r[key+'_lo']=float(lo);r[key+'_hi']=float(hi)
                    decision.append(r)
                for refscore in ('normal','reversed'):
                    r=dict(model=model,dataset=dataset,comparison='averaged_minus_'+refscore)
                    for k,m in enumerate(METRICS):
                        r[m]=float(observed['averaged'][k]-observed[refscore][k]);lo,hi=np.quantile(bs['averaged'][:,k]-bs[refscore][:,k],[.025,.975]);r[m+'_lo']=float(lo);r[m+'_hi']=float(hi)
                    comparisons.append(r)
                print(model,dataset,'verified and analyzed',flush=True)
    for name,rs in [('metrics',results),('stability',stable),('retained_risk',decision),('paired_differences',comparisons)]:csv_save(out/f'{name}.csv',rs)
    write_json(out/'audit.json',audit)
    text='# Fresh candidate-label correctness analysis\n\nOnly `letter11_provenance_v1` forward/reversed distributions are used. All question IDs and every candidate prompt hash are matched against locally reconstructed dataset candidates. No mean-logprob preference target is used.\n\nTruthfulQA labels are dataset MC1 correctness. Pavlick labels mean agreement with the adapter’s sign-of-mean human judgment convention, collapsing NLI ambiguity; they are not unqualified factual correctness. Both populations contain exactly one positive and one negative supplied candidate per question. They do not estimate accuracy or retained risk of naturally generated model answers.\n\n| Model | Dataset | Score | Brier | AUROC | AURC |\n|---|---|---|---:|---:|---:|\n'
    for r in results:text+=f"| {r['model']} | {r['dataset']} | {r['score']} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['aurc']:.4f} |\n"
    text+='\nBootstrap intervals use 2,000 paired question resamples, retaining both candidate rows together. Retention ranks candidates by decoded expected correctness; ties use row IDs. Overlap is intersection size divided by retained count; Jaccard is separately reported. Retained risk is the fraction of negative-labeled candidates retained. Letter-token mass semantics from the original fresh caches are preserved and differ from the new training contract’s full letter-plus-newline sequence mass. Historical prompt/token provenance is preserved rather than relabeled as a new prompt run.\n'
    (out/'REPORT.md').write_text(text)
if __name__=='__main__':main()
