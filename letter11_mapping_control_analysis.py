"""Audit original-cell known-value controls against provenance-v1 candidates."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from letter11_mapping_control import MODELS,SCHEMES,LETTERS,prompt_for,text_for

ROOT=Path(__file__).resolve().parent;out=ROOT/'cached_results/letter11_mapping_control_v1';metrics=[];paired=[];total=0;context_verified=0
for model in MODELS:
 obj=json.loads((out/f'{model}.json').read_text());assert (obj['model_id'],obj['revision'])==MODELS[model];rows=obj['rows'];total+=len(rows)
 refs={}
 for dataset in ['truthfulqa','pavlick_nli']:
  for legend in SCHEMES:
   source=torch.load(ROOT/'letter11_provenance_v1'/model/dataset/f'{legend}.pt',map_location='cpu',weights_only=False)
   assert source['meta']['model_revision']==obj['revision'];refs[dataset,legend]={str(i):{side:str(hashes[j]) for side,hashes in [('pos',source['prompt_hashes_pos']),('neg',source['prompt_hashes_neg'])]} for j,i in enumerate(source['example_ids'])}
 for r in rows:
  values=SCHEMES[r['legend']];text=text_for(r['value'],values,r['question'],r['answer']);assert text==r['prompt'] and hashlib.sha256(text.encode()).hexdigest()==r['prompt_sha256']
  p=np.array(r['probability_vector']);assert len(p)==11 and np.isclose(p.sum(),1) and np.all(p>=0);assert abs(float(p@(np.array(values)/100))-r['decoded_mean'])<1e-6
  assert r['expected_symbol']==LETTERS[list(values).index(r['value'])]
  if r['kind']=='contextual':
   original=prompt_for(r['question'],r['answer'],values);assert hashlib.sha256(original.encode()).hexdigest()==refs[r['dataset'],r['legend']][r['question_id']][r['candidate_side']];context_verified+=1
  r['unconstrained_first_token_correct']=bool(r['generated_token_ids'] and r['generated_token_ids'][0]==r['candidate_token_ids'][LETTERS.index(r['expected_symbol'])])
 for dataset in [None,'truthfulqa','pavlick_nli']:
  for legend in SCHEMES:
   rs=[r for r in rows if r['dataset']==dataset and r['legend']==legend];assert len(rs)==(11 if dataset is None else 64)
   metrics.append(dict(model=model,dataset=dataset or 'standalone',legend=legend,n=len(rs),conditional_correct=sum(r['conditional_correct'] for r in rs),conditional_accuracy=float(np.mean([r['conditional_correct'] for r in rs])),unconstrained_first_token_accuracy=float(np.mean([r['unconstrained_first_token_correct'] for r in rs])),strict_full_generation_accuracy=float(np.mean([r['emitted_correct'] for r in rs])),strict_full_generation_validity=float(np.mean([r['emitted_valid'] for r in rs])),allowed_mass_mean=float(np.mean([r['allowed_alphabet_mass'] for r in rs])),known_value_mean_absolute_error=float(np.mean([abs(r['decoded_mean']-r['value']/100) for r in rs]))))
  if dataset is not None:
   groups={legend:{(r['question_id'],r['candidate_side']):r for r in rows if r['dataset']==dataset and r['legend']==legend} for legend in SCHEMES};ids=sorted({k[0] for k in groups['forward']});assert set(groups['forward'])==set(groups['reversed']);a=np.array([[float(groups['forward'][i,s]['conditional_correct']) for s in ['pos','neg']] for i in ids]);b=np.array([[float(groups['reversed'][i,s]['conditional_correct']) for s in ['pos','neg']] for i in ids]);ixs=np.random.default_rng(20260915).integers(0,len(ids),(2000,len(ids)));lo,hi=np.quantile((b-a)[ixs].mean((1,2)),[.025,.975]);paired.append(dict(model=model,dataset=dataset,n_questions=len(ids),reversed_minus_forward_accuracy=float((b-a).mean()),lo=float(lo),hi=float(hi)))
for name,rs in [('metrics',metrics),('paired_differences',paired)]:
 with (out/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
(out/'AUDIT.json').write_text(json.dumps(dict(records=total,original_candidate_prompt_hashes_verified=context_verified,models=list(MODELS),task='known-value encoding, not answer correctness',paired_unit='question, both candidate contexts together',original_readout='conditional candidate-token distribution at original prefix',strict_generation_note='Extra continuation text can fail whole-response grammar despite a correct first token; report these readouts separately.'),indent=2)+'\n')
text='# ICLR original-cell letter11 mapping control\n\nOriginal pinned model/tokenizer revisions and provenance-v1 next-letter readout. The task supplies a known confidence value and requests its symbol, rather than asking the model to judge correctness. Standalone controls cover all 11 values; contextual controls use 32 deterministic question IDs per dataset, both candidate answers, and the same assigned value under both legends.\n\n| Model | Context | Forward conditional accuracy | Reversed conditional accuracy | Reversed allowed mass | Reversed first-token accuracy |\n|---|---|---:|---:|---:|---:|\n'
for model in MODELS:
 for dataset in ['standalone','truthfulqa','pavlick_nli']:
  a,b=[next(r for r in metrics if (r['model'],r['dataset'],r['legend'])==(model,dataset,l)) for l in ['forward','reversed']]
  text+=f"| {model} | {dataset} | {a['conditional_correct']}/{a['n']} | {b['conditional_correct']}/{b['n']} | {b['allowed_mass_mean']:.3f} | {b['unconstrained_first_token_accuracy']:.1%} |\n"
text+='\nAll 768 contextual prompt/candidate instances were matched to the original provenance-v1 source by question ID and original candidate-prompt hash. The 66 standalone instances are model controls, not additional independent dataset observations. Conditional accuracy tests the original probability readout; unconstrained first-token and strict whole-generation compliance are reported separately. These controls do not prove that natural correctness-confidence judgments interpret every legend correctly, and do not turn report distributions into calibrated correctness bounds.\n\nICLR scope: attach the reversal mapping control to the existing letter11 analysis, with §7 pointers to the separate training/direct-report follow-up. The ICLR two-failure headline is unchanged.\n'
(out/'REPORT.md').write_text(text);print(text)
