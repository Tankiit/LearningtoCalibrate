"""Actual-emission tables, paired by question; never impute invalid confidence."""
import json,hashlib
from pathlib import Path
import numpy as np
from .data import load_jsonl,write_json
from .emitted_reports import render,parse
from .reporting import digest
from .matched_analysis import metric_vector,csv_save

def analyze(out,draws=2000):
 out=Path(out);manifest=json.loads((out/'manifest.json').read_text());refs={r['question_id']:r for r in load_jsonl(out/'data.jsonl')};ids=sorted(refs)
 maps=json.loads((out/'mappings.json').read_text());reports=load_jsonl(out/'reports.jsonl');groups={};n=len(ids);y=np.array([refs[i]['correct'] for i in ids]);rng=np.random.default_rng(20260915);ixs=rng.integers(0,n,(draws,n))
 for r in reports:
  ref=refs[r['question_id']]
  if any(r[k]!=ref[k] for k in ('answer','question','correct','split')):raise ValueError('Question/candidate/label mismatch')
  mapping=maps[r['scale']]['normal' if r['format']=='direct' else r['format']]
  expected=render(ref['question'],ref['answer'],mapping,r['format']=='direct',manifest['instruction'])
  if r['mapping']!=mapping or r['prompt_text']!=expected or r['prompt_text_sha256']!=hashlib.sha256(expected.encode()).hexdigest():raise ValueError('Prompt or mapping mismatch')
  if r['full_prompt_hash']!=digest(r['prompt_token_ids']) or r['model_revision']!=manifest['sampler_path'] or r['sampling_policy']!=manifest['policy']:raise ValueError('Token/model/policy provenance mismatch')
  if parse(r['raw_response'],mapping,r['format']=='direct',contract_name=manifest.get('parse_contract','strict_percent_v1'))!=r['emitted_value']:raise ValueError('Emitted-value decoding mismatch')
  key=(r['scale'],r['format']);g=groups.setdefault(key,{})
  if r['question_id'] in g:raise ValueError('Duplicate emission')
  g[r['question_id']]=r
 metrics=[];stability=[];paired=[];raw={};boot={}
 for scale in manifest['scales']:
  matrices=[]
  for fmt in manifest['formats']:
   g=groups[scale,fmt]
   if set(g)!=set(ids):raise ValueError('Incomplete emission group')
   p=np.array([np.nan if g[i]['emitted_value'] is None else g[i]['emitted_value'] for i in ids]);matrices.append(p)
  ps=np.array(matrices).T
  scores={fmt:ps[:,j] for j,fmt in enumerate(manifest['formats'])}
  indices={fmt:j for j,fmt in enumerate(manifest['formats'])}
  scores.update(normal_reversed_average=ps[:,[indices['normal'],indices['reversed']]].mean(1),three_format_average=ps[:,[indices[x] for x in ['direct','normal','reversed']]].mean(1))
  legend_indices=[j for fmt,j in indices.items() if fmt!='direct']
  if len(legend_indices)>=6:scores['six_report_average']=ps[:,legend_indices[:6]].mean(1)
  for name,p in scores.items():
   valid=np.isfinite(p);loss=np.where(valid,(p-y)**2,1.);b=np.mean(loss[ixs],axis=1);raw[scale,name]=loss;boot[scale,name]=b
   vals=metric_vector(y[valid],p[valid]) if valid.sum() else [np.nan]*3
   row=dict(scale=scale,method=name,n_questions=n,n_valid=int(valid.sum()),invalid_rate=float(1-valid.mean()),penalized_brier=float(loss.mean()),brier_lo=float(np.quantile(b,.025)),brier_hi=float(np.quantile(b,.975)),valid_only_brier=float(vals[0]),valid_only_auroc=float(vals[1]),valid_only_aurc=float(vals[2]),deployment_calls=2 if name=='normal_reversed_average' else 3 if name=='three_format_average' else 6 if name=='six_report_average' else 1)
   metrics.append(row)
  for a,b in [('normal_reversed_average','normal'),('reversed','normal'),('direct','normal'),('three_format_average','normal_reversed_average')]:
   delta=boot[scale,a]-boot[scale,b];paired.append(dict(scale=scale,comparison=a+'_minus_'+b,penalized_brier=float(np.mean(raw[scale,a]-raw[scale,b])),lo=float(np.quantile(delta,.025)),hi=float(np.quantile(delta,.975))))
  for indices,name in [([1,2],'normal_reversed'),([0,1,2],'direct_normal_reversed')]:
   p=ps[:,indices];valid=np.isfinite(p).all(1);p=p[valid];ranges=np.ptp(p,axis=1)
   for tau in (.5,.75,.9):
    stability.append(dict(scale=scale,comparison=name,n_complete=int(valid.sum()),n_excluded_invalid=int(n-valid.sum()),threshold=tau,mean_range=float(ranges.mean()) if len(ranges) else float('nan'),p95_range=float(np.quantile(ranges,.95)) if len(ranges) else float('nan'),threshold_flip=float(np.mean((p.min(1)<tau)&(p.max(1)>=tau))) if len(p) else float('nan'),all_accept_fraction=float(np.mean(p.min(1)>=tau)) if len(p) else float('nan')))
 folder=out/'analysis';folder.mkdir(exist_ok=True)
 for name,rows in [('metrics',metrics),('paired_differences',paired),('stability',stability)]:csv_save(folder/f'{name}.csv',rows)
 write_json(folder/'protocol.json',dict(n=n,bootstrap_draws=draws,unit='paired question',primary='actual emitted confidence',invalid='loss penalty 1; valid-only discrimination explicitly conditional; average invalid if any component invalid',cost=dict(target_requests=len(reports),input_tokens=sum(r['input_tokens'] for r in reports),generated_tokens=sum(r['generated_tokens'] for r in reports)),scope=manifest.get('split_scope','within-grid equivalent contracts; coarse-trained checkpoint transferred to both grids; no fine-grid training claim')))
 text=f'# Actual emitted confidence: fixed prompt evaluation\n\nFrozen checkpoint ({manifest.get("checkpoint_description","coarse correctness-only Qwen")}) on {n} matched questions. Original partitions are unchanged; no fitting occurs in this analysis. Greedy decoding, 32-token cap, newline stop; strict parser, no semantic retry. Invalid output receives loss 1; valid-only AUROC/AURC are conditional.\n\n| Grid | Report | Penalized Brier | 95% interval | AUROC (valid only) | AURC (valid only) | Invalid |\n|---|---|---:|---|---:|---:|---:|\n'
 for r in metrics:text+=f"| {r['scale']} | {r['method']} | {r['penalized_brier']:.4f} | [{r['brier_lo']:.4f}, {r['brier_hi']:.4f}] | {r['valid_only_auroc']:.4f} | {r['valid_only_aurc']:.4f} | {r['invalid_rate']:.1%} |\n"
 text+='\nThese are free generations, not conditional permitted-token argmaxes or decoded expectations. Averaged estimators belong in a separate final-score panel. Same-grid comparisons share questions, candidate answers and checkpoint. The checkpoint choice was informed by the earlier exploratory coarse test summary.\n'
 if manifest.get('parse_contract')=='lenient_v1':
  text=text.replace('strict parser, no semantic retry','declared lenient_v1 parser, no semantic retry')
  timing='The parser contracts were declared before these new generations.' if manifest.get('schema')=='trained-emissions-v1' else 'Collection predates the lenient declaration; this is a secondary retrospective analysis.'
  text+='\nSecondary lenient_v1 contract: bare [0,1] numbers denote probabilities; bare (1,100] numbers and percent-suffixed values use percentage units. Exact grid membership is required. Strict results remain primary and unchanged. '+timing+' Prior GEPA searches were optimized under the strict contract, not this parser.\n'
 if manifest.get('parse_contract')=='optional_percent_v1':
  text=text.replace('strict parser, no semantic retry','declared optional-percent parser, no semantic retry')
  text+='\nSecondary contract: a bare numeral is read in percentage units. This is retrospective rescoring of preserved raw generations after observing suffix omissions, not a fresh elicitation run or a GEPA search optimized for this parser. The original strict scores remain preserved.\n'
 (folder/'REPORT.md').write_text(text)
 print(text)
if __name__=='__main__':
 import sys
 analyze(sys.argv[1])
