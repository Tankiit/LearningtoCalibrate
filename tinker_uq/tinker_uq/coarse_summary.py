"""Compact individual-report versus ensemble readouts for the finished coarse study."""
import json
from pathlib import Path
import numpy as np
from .data import load_jsonl,write_json
from .matched_analysis import csv_save

def run(out):
 out=Path(out);fd=out/'students/analysis_coarse';rows=[]
 for method in ['teacher','both_mean_brier','aligned_distill','aligned_distill_supervised']:
  root=out/'students/coarse'/method;names=list(json.loads((out/'mappings.json').read_text())['coarse']);gs=[{r['question_id']:r for r in load_jsonl(root/f'{name}.jsonl')} for name in names];ids=sorted(gs[0]);y=np.array([gs[0][i]['correct'] for i in ids]);p=np.array([[g[i]['decoded_mean'] for g in gs] for i in ids]);loss=(p-y[:,None])**2;ixs=np.random.default_rng(20260915).integers(0,len(ids),(2000,len(ids)))
  r=dict(method=method,n=len(ids),normal=float(loss[:,0].mean()),reversed=float(loss[:,1].mean()),average_per_report=float(loss.mean()),worst_legend=float(loss.mean(0).max()),worst_legend_name=names[int(loss.mean(0).argmax())],six_report_average=float(np.mean((p.mean(1)-y)**2)),mean_pairwise_disagreement=float(np.mean([abs(p[:,i]-p[:,j]) for i in range(6) for j in range(i+1,6)])),p95_range=float(np.quantile(np.ptp(p,axis=1),.95)),threshold_flip_075=float(np.mean((p.min(1)<.75)&(p.max(1)>=.75))))
  for key,bb in [('average_per_report',loss[ixs].mean((1,2))),('worst_legend',loss[ixs].mean(1).max(1)),('six_report_average',((p.mean(1)-y)**2)[ixs].mean(1))]:r[key+'_lo'],r[key+'_hi']=map(float,np.quantile(bb,[.025,.975]))
  rows.append(r)
 csv_save(fd/'individual_and_ensemble.csv',rows)
 text='\nIndividual and ensemble Brier (decoded expectation):\n\n| Method | Normal | Reversed | Mean per-report | Worst legend | Six-report average | Mean pairwise gap | Any-legend flip at .75 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['method']} | {r['normal']:.4f} | {r['reversed']:.4f} | {r['average_per_report']:.4f} | {r['worst_legend']:.4f} | {r['six_report_average']:.4f} | {r['mean_pairwise_disagreement']:.4f} | {r['threshold_flip_075']:.1%} |\n"
 path=fd/'REPORT.md';s=path.read_text().split('\nIndividual and ensemble Brier')[0];path.write_text(s+text+'\nWorst legend means maximum dataset-average Brier across the six mappings, not the mean questionwise maximum. Zero threshold flips can result from every report remaining below the threshold; it does not establish useful acceptance.\n')
 write_json(out/'COARSE_COMPLETE.json',dict(complete=True,methods=[r['method'] for r in rows],questions=128,readout='decoded expectation',fine_training_complete=False))
 print(text)
if __name__=='__main__':
 import sys
 run(sys.argv[1])
