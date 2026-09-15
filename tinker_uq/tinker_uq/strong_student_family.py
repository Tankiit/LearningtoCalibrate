"""Fit family selectors on validation, evaluate the preserved coarse test cache."""
import json
from pathlib import Path
import numpy as np
from .data import load_jsonl,write_json,write_jsonl
from .matched_experiment import score_file
from .matched_analysis import verified_group,metric_vector,csv_save
from .reporting import align_to_values
from .family_analysis import family_summary,predictions,top

def run(out):
 import tinker
 out=Path(out);root=out/'students/coarse/both_mean_brier';maps=json.loads((out/'mappings.json').read_text())['coarse'];data=load_jsonl(out/'data.jsonl');val=[r for r in data if r['split']=='val']
 cp=json.loads((root/'checkpoint.json').read_text());service=tinker.ServiceClient(timeout=60,max_retries=2);sampler=service.create_sampling_client(model_path=cp['sampler_path'])
 fd=root/'validation';fd.mkdir(exist_ok=True);score_file(sampler,val,sampler.get_tokenizer(),'both_mean_brier',fd,tinker,maps)
 analyze(out)

def analyze(out,draws=2000):
 out=Path(out);root=out/'students/coarse/both_mean_brier';maps=json.loads((out/'mappings.json').read_text())['coarse'];data=load_jsonl(out/'data.jsonl');refs={r['row_id']:r for r in data if r['split'] in ('val','test')};ids=sorted(refs);values=np.array([0,.5,1]);families={}
 for split,fd in [('val',root/'validation'),('test',root)]:
  rs={i:r for i,r in refs.items() if r['split']==split};gs=[verified_group(fd/f'{name}.jsonl',rs,mapping) for name,mapping in maps.items()]
  for i in rs:families[i]=[align_to_values(g[i]['probability_vector'],g[i]['mapping']) for g in gs]
 y=np.array([refs[i]['correct'] for i in ids]);fit=np.array([j for j,i in enumerate(ids) if refs[i]['split']=='val']);test=np.array([j for j,i in enumerate(ids) if refs[i]['split']=='test']);testids=[ids[j] for j in test]
 s=family_summary(np.array([families[i] for i in ids]),values);scores=predictions(s,y,fit,test,values);yy=y[test];rng=np.random.default_rng(20260915);ixs=rng.integers(0,len(test),(draws,len(test)));rows=[];paired=[];risks=[];bs={};obs={}
 for name,p in scores.items():
  p=p[test];obs[name]=metric_vector(yy,p);bs[name]=np.array([metric_vector(yy[ix],p[ix]) for ix in ixs]);r=dict(approach=name,n_fit=len(fit),n_test=len(test),calls=1 if name=='normal_report' else 2 if name=='two_orientation_average' else 6)
  for j,m in enumerate(['brier','auroc','aurc']):r[m]=float(obs[name][j]);r[m+'_lo'],r[m+'_hi']=map(float,np.nanquantile(bs[name][:,j],[.025,.975]))
  rows.append(r)
  for coverage in (.2,.5,.8,.9):
   risk=float(np.mean(1-yy[top(p,testids,coverage)]));bb=[]
   for ix in ixs:bb.append(float(np.mean(1-yy[ix][top(p[ix],[str(j) for j in range(len(ix))],coverage)])))
   lo,hi=np.quantile(bb,[.025,.975]);risks.append(dict(approach=name,coverage=coverage,risk=risk,lo=float(lo),hi=float(hi)))
 for a,b in [('fitted_family_shape','fitted_mixture_plus_elementary'),('fitted_family_shape','fitted_elementary_extrema'),('family_lower_mean','average_confidence')]:
  r=dict(comparison=a+'_minus_'+b)
  for j,m in enumerate(['brier','auroc','aurc']):r[m]=float(obs[a][j]-obs[b][j]);r[m+'_lo'],r[m+'_hi']=map(float,np.nanquantile((bs[a]-bs[b])[:,j],[.025,.975]))
  paired.append(r)
 fd=root/'family_analysis';fd.mkdir(exist_ok=True)
 for name,rr in [('metrics',rows),('paired_differences',paired),('retained_risk',risks)]:csv_save(fd/f'{name}.csv',rr)
 write_jsonl(fd/'predictions.jsonl',[dict(row_id=i,split=refs[i]['split'],correct=int(y[j]),aligned_family=families[i],scores={k:float(v[j]) for k,v in scores.items()}) for j,i in enumerate(ids)])
 write_json(fd/'protocol.json',dict(fit_split='128 validation questions',evaluation='128 preserved test questions',tuning='fixed C=1 standard-scaled logistic models; no test tuning',bootstrap='paired test questions, 2000 draws; fit held fixed',scope='auxiliary decoded-distribution family of strongest coarse supervised control; exploratory after inspected coarse test summary'))
 text='# Strong supervised student: family selection\n\nSelectors fitted only on 128 validation questions; evaluated on the preserved 128 test questions. All family/mixture/spread methods use the same six reports. This auxiliary decoded-distribution analysis is exploratory, following inspection of the earlier coarse test summary.\n\n| Approach | Brier | AUROC | AURC |\n|---|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['approach']} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['aurc']:.4f} |\n"
 (fd/'REPORT.md').write_text(text);print(text,flush=True)
if __name__=='__main__':
 import sys
 run(Path(sys.argv[1]))
