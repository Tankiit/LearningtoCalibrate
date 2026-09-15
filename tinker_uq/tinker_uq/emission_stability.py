"""Paired numerical instability and selective-risk readouts for actual emissions."""
import json,os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .data import load_jsonl,write_json
from .matched_analysis import csv_save,retention

def run(out,draws=2000):
 out=Path(out);rows=load_jsonl(out/'reports.jsonl');data=load_jsonl(out/'data.jsonl');ids=sorted(r['question_id'] for r in data);refs={r['question_id']:r for r in data};y=np.array([refs[i]['correct'] for i in ids]);n=len(ids);ixs=np.random.default_rng(20260915).integers(0,n,(draws,n));summary=[];risks=[];gaps={};plot={}
 for scale in ['coarse','fine']:
  gs={fmt:{r['question_id']:r['emitted_value'] for r in rows if r['scale']==scale and r['format']==fmt} for fmt in ['normal','reversed']}
  a=np.array([gs['normal'][i] for i in ids],float);b=np.array([gs['reversed'][i] for i in ids],float)
  if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Specify invalid-aware retention before applying to incomplete legends')
  delta=abs(a-b);gaps[scale]=delta;plot[scale]=(a,b);r=dict(scale=scale,n=n,mean_abs_shift=float(delta.mean()),p95_abs_shift=float(np.quantile(delta,.95)),p99_abs_shift=float(np.quantile(delta,.99)),spearman=float(spearmanr(a,b).statistic))
  for key,calc in [('mean_abs_shift',lambda ix:delta[ix].mean()),('p95_abs_shift',lambda ix:np.quantile(delta[ix],.95))]:r[key+'_lo'],r[key+'_hi']=map(float,np.quantile([calc(ix) for ix in ixs],[.025,.975]))
  for tau in [.5,.75,.9]:
   flips=(a>=tau)!=(b>=tau);key=f'flip_{tau:g}';r[key]=float(flips.mean());r[key+'_lo'],r[key+'_hi']=map(float,np.quantile(flips[ixs].mean(1),[.025,.975]))
  summary.append(r)
  for frac in [.2,.5,.8,.9]:
   r=dict(scale=scale,**retention(y,a,b,ids,frac));deltas=[]
   for ix in ixs:
    rr=retention(y[ix],a[ix],b[ix],[f'{ids[j]}:{k:06d}' for k,j in enumerate(ix)],frac);deltas.append(rr['risk_delta'])
   r['risk_delta_lo'],r['risk_delta_hi']=map(float,np.quantile(deltas,[.025,.975]));risks.append(r)
 folder=out/'analysis';csv_save(folder/'numerical_stability.csv',summary);csv_save(folder/'retained_risk.csv',risks)
 d=gaps['fine']-gaps['coarse'];write_json(folder/'cross_grid_gap.json',dict(mean_abs_shift_fine_minus_coarse=float(d.mean()),paired_question_interval=np.quantile(d[ixs].mean(1),[.025,.975]).tolist(),scope='same coarse-trained checkpoint; descriptive transfer across these two interfaces, no fine-training conclusion'))
 os.environ.setdefault('MPLCONFIGDIR','/private/tmp/confidence-mpl');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(8,3.7),sharex=True,sharey=True)
 for ax,(scale,(a,b)) in zip(axes,plot.items()):
  pairs,counts=np.unique(np.column_stack([a,b]),axis=0,return_counts=True);ax.plot([0,1],[0,1],c='.75',lw=1,zorder=0)
  ax.scatter(pairs[:,0],pairs[:,1],s=counts*1.4,alpha=.65,color='#26709c',edgecolors='white',linewidth=.5)
  ax.set(title=f'{scale.capitalize()} grid',xlabel='Normal legend: emitted confidence',xlim=(-.12,1.12),ylim=(-.12,1.12),xticks=[0,.5,1],yticks=[0,.5,1]);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Reversed legend: emitted confidence');fig.text(.5,.01,'1,000 matched validation questions; circle area is proportional to question count.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(folder/'emitted_normal_reversed.png',dpi=180);fig.savefig(folder/'emitted_normal_reversed.pdf');plt.close(fig)
if __name__=='__main__':
 import sys
 run(sys.argv[1])
