"""Complete supporting q/qa normal/reversed/average table with paired intervals."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from tinker_uq.data import load_jsonl,write_json
from tinker_uq.training import normalize_scores,prompt_text
from tinker_uq.matched_analysis import metric_vector,csv_save
ROOT=Path(__file__).resolve().parent;run=ROOT/'runs/full_llama31_8b_qwen3_8b';out=run/'qa_average_verified';data=ROOT/'data/full_llama31_8b/cached_answers.jsonl';manifest=json.loads((run/'manifest.json').read_text())
assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['data_sha256']
refs={r['row_id']:r for r in load_jsonl(data)};groups=defaultdict(dict);count=0
for r in load_jsonl(run/'predictions.jsonl'):
 ref=refs[r['row_id']];assert all(r[k]==ref[k] for k in ['question_id','correct','split']);p,lm=normalize_scores(r['label_logprobs'],r['legend']);assert abs(p-r['p_correct'])<1e-12 and abs(lm-r['allowed_label_logmass'])<1e-12
 key=(r['arm'],r['stage'],r['legend'],r['split']);assert r['row_id'] not in groups[key];groups[key][r['row_id']]=r;count+=1
for key,g in groups.items():assert set(g)=={i for i,r in refs.items() if r['split']==key[-1]}
audit=json.loads((run/'prompt_audit.json').read_text());first=next(iter(refs.values()))
for arm in ['q','qa']:assert audit[arm]['text']==prompt_text(first,arm,'normal')
ids=sorted(i for i,r in refs.items() if r['split']=='test');y=np.array([refs[i]['correct'] for i in ids]);summary=[];contrasts=[];observed={};boots={}
for arm in ['q','qa']:
 for stage in ['base','trained']:
  a,b=(np.array([groups[arm,stage,legend,'test'][i]['p_correct'] for i in ids]) for legend in ['normal','reversed']);scores={'normal':a,'reversed':b,'average':(a+b)/2}
  rng=np.random.default_rng(20260914);ixs=rng.integers(0,len(y),(2000,len(y)))
  for name,p in scores.items():
   obs=metric_vector(y,p);bb=np.array([metric_vector(y[ix],p[ix]) for ix in ixs]);observed[arm,stage,name]=obs;boots[arm,stage,name]=bb;r=dict(arm=arm,stage=stage,score=name,n=len(y))
   for j,m in enumerate(['brier','auroc','aurc']):r[m]=float(obs[j]);r[m+'_lo'],r[m+'_hi']=map(float,np.quantile(bb[:,j],[.025,.975]))
   summary.append(r)
  print('Binary table',arm,stage,flush=True)
for arm in ['q','qa']:
 for stage in ['base','trained']:
  for b in ['normal','reversed']:
   aa=(arm,stage,'average');bb=(arm,stage,b);r=dict(comparison=arm+'_'+stage+'_average_minus_'+b)
   for j,m in enumerate(['brier','auroc','aurc']):r[m]=float(observed[aa][j]-observed[bb][j]);r[m+'_lo'],r[m+'_hi']=map(float,np.quantile((boots[aa]-boots[bb])[:,j],[.025,.975]))
   contrasts.append(r)
for stage in ['base','trained']:
 for name in ['normal','reversed','average']:
  aa=('qa',stage,name);bb=('q',stage,name);r=dict(comparison=stage+'_'+name+'_qa_minus_q')
  for j,m in enumerate(['brier','auroc','aurc']):r[m]=float(observed[aa][j]-observed[bb][j]);r[m+'_lo'],r[m+'_hi']=map(float,np.quantile((boots[aa]-boots[bb])[:,j],[.025,.975]))
  contrasts.append(r)
csv_save(out/'q_qa_metrics.csv',summary);csv_save(out/'q_qa_paired_differences.csv',contrasts);write_json(out/'q_qa_audit.json',dict(rows_verified=count,matched_questions=len(ids),bootstrap_draws=2000,paired_unit='question',source_grade_rebuild='see existing source-hash-verified QA audit',reversed_prompt_limit='historical reversed prompt not independently saved; inferred from runner'))
text='# Supporting binary q/qa comparison\n\n10,647 matched test questions. Conditional correctness probabilities decoded from full A/B-plus-newline sequence scores. This is not an emitted numerical-confidence experiment.\n\n| Input | Stage | Score | Brier | AUROC | AURC |\n|---|---|---|---:|---:|---:|\n'
for r in summary:text+=f"| {r['arm']} | {r['stage']} | {r['score']} | {r['brier']:.6f} | {r['auroc']:.6f} | {r['aurc']:.6f} |\n"
text+='\nScore intervals and paired average-minus-individual / qa-minus-q differences use 2,000 shared question bootstrap draws, saved in q_qa_metrics.csv and q_qa_paired_differences.csv. Stability, explicitly defined overlap and retained risk are in binary_stability.csv; the scatter figure is q_qa_scatter.png.\n'
(out/'Q_QA_REPORT.md').write_text(text)
