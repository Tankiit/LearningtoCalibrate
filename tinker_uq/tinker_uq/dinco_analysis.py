"""Separate raw emission, NVC, SC, and DINCO on matched pilot questions."""
import json
from pathlib import Path
import numpy as np
from .data import load_jsonl,write_json
from .matched_analysis import metric_vector,csv_save

def run(out,draws=2000):
 out=Path(out);records=load_jsonl(out/'verified_reports.jsonl');refs={r['question_id']:r for r in load_jsonl(out/'data.jsonl')};ids=sorted(refs);y=np.array([refs[i]['correct'] for i in ids]);n=len(ids);ixs=np.random.default_rng(20260915).integers(0,n,(draws,n));metric=[];stability=[];allp={}
 for scale in ['coarse','fine']:
  for fmt in ['direct','normal','reversed']:
   g={r['question_id']:r for r in records if r['scale']==scale and r['format']==fmt}
   if set(g)!=set(ids):raise ValueError('Incomplete DINCO question join')
   for method in ['raw','normalized','self_consistency','dinco']:
    p=np.array([g[i]['raw_emission'] if method=='raw' else g[i]['self_consistency'] if method=='self_consistency' else g[i]['scores'][method] if g[i]['valid'] else np.nan for i in ids],float);valid=np.isfinite(p);allp[scale,fmt,method]=p
    loss=np.where(valid,(p-y)**2,1.);b=loss[ixs].mean(1);v=metric_vector(y[valid],p[valid]) if valid.sum() else [np.nan]*3
    metric.append(dict(scale=scale,format=fmt,method=method,n=n,n_valid=int(valid.sum()),invalid_rate=float(1-valid.mean()),penalized_brier=float(loss.mean()),lo=float(np.quantile(b,.025)),hi=float(np.quantile(b,.975)),valid_only_brier=float(v[0]),valid_only_auroc=float(v[1]),valid_only_aurc=float(v[2])))
  for method in ['raw','normalized','self_consistency','dinco']:
   a,b=(allp[scale,f,method] for f in ['normal','reversed']);valid=np.isfinite(a)&np.isfinite(b);gap=abs(a[valid]-b[valid]);stability.append(dict(scale=scale,method=method,n_matched_valid=int(valid.sum()),mean_abs_shift=float(gap.mean()),p95_abs_shift=float(np.quantile(gap,.95)),flip_075=float(np.mean((a[valid]>=.75)!=(b[valid]>=.75)))))
  sc=[allp[scale,f,'self_consistency'] for f in ['direct','normal','reversed']]
  if not all(np.array_equal(sc[0],p) for p in sc[1:]):raise ValueError('SC differs across formats')
  na,nb=(allp[scale,f,'normalized'] for f in ['normal','reversed']);da,db=(allp[scale,f,'dinco'] for f in ['normal','reversed']);valid=np.isfinite(na)&np.isfinite(nb)
  if not np.allclose(abs(da[valid]-db[valid]),.5*abs(na[valid]-nb[valid]),atol=1e-12):raise ValueError('DINCO mixture identity failed')
 folder=out/'analysis';folder.mkdir(exist_ok=True);csv_save(folder/'metrics.csv',metric);csv_save(folder/'stability.csv',stability)
 aux=load_jsonl(out/'auxiliary.jsonl');calls=[c for r in records for c in r['candidate_reports'][1:]]+[c for r in aux for c in [r['distractor_generation']]+r['self_consistency_generations']]
 write_json(folder/'audit.json',dict(n=n,sc_frozen=True,mixture_identity_verified=True,distractor_lists_valid=sum(r['valid_distractor_list'] for r in aux),recorded_new_generation_calls=len(calls),reused_main_reports=len(records),recorded_new_input_tokens=sum(r['input_tokens'] for r in calls),recorded_new_generated_tokens=sum(r['generated_tokens'] for r in calls),cost_limit='Recorded calls exclude requests interrupted during the first NLI precision failure and later concurrency restart. Therefore totals are lower bounds, not exact realized cost. A clean run is required for an exact-cost comparison.',bootstrap='2000 paired question resamples within each table; validity denominators retained'))
 text='# DINCO supplied-answer implementation pilot\n\n32 deterministic development questions, with the same supplied answer and frozen auxiliary candidates/SC across formats. This is a black-box candidate-list adaptation with an untuned Qwen auxiliary generator and a coarse-trained confidence reporter, not an exact reproduction.\n\n| Grid | Format | Estimator | Valid / 32 | Penalized Brier | Valid-only AUROC | Valid-only AURC |\n|---|---|---|---:|---:|---:|---:|\n'
 for r in metric:text+=f"| {r['scale']} | {r['format']} | {r['method']} | {r['n_valid']} | {r['penalized_brier']:.4f} | {r['valid_only_auroc']:.4f} | {r['valid_only_aurc']:.4f} |\n"
 text+='\nNormal/reversed stability on matched valid questions:\n\n| Grid | Estimator | Mean shift | 95th percentile shift | Flip at .75 |\n|---|---|---:|---:|---:|\n'
 for r in stability:text+=f"| {r['scale']} | {r['method']} | {r['mean_abs_shift']:.4f} | {r['p95_abs_shift']:.4f} | {r['flip_075']:.1%} |\n"
 text+='\nThe final DINCO shift is exactly half the normalized-confidence shift because SC is held fixed. It does not demonstrate repair of raw emissions. Invalid output has loss penalty 1; conditional discrimination is not comparable across changing valid subsets without qualification. Recorded cost totals omit interrupted attempts and are lower bounds. No equal-cost superiority or main 1,000-question DINCO result is claimed.\n'
 (folder/'REPORT.md').write_text(text);print(text)
if __name__=='__main__':
 import sys
 run(sys.argv[1])
