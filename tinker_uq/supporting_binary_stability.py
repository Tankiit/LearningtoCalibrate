"""Correctness-only supporting binary decision table and q/qa scatter panels."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tinker_uq.matched_analysis import retention,pair_stability,csv_save
ROOT=Path(__file__).resolve().parent
run=ROOT/'runs/full_llama31_8b_qwen3_8b';out=run/'qa_average_verified';out.mkdir(exist_ok=True)
groups=defaultdict(dict)
with (run/'predictions.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r['split']=='test':groups[r['arm'],r['stage'],r['legend']][r['row_id']]=r
rows=[];fig,axes=plt.subplots(2,2,figsize=(8,7),sharex=True,sharey=True)
for j,arm in enumerate(('q','qa')):
 for k,stage in enumerate(('base','trained')):
  normal=groups[arm,stage,'normal'];reverse=groups[arm,stage,'reversed'];assert set(normal)==set(reverse)
  ids=sorted(normal);assert all(normal[i]['correct']==reverse[i]['correct'] and normal[i]['question_id']==reverse[i]['question_id'] for i in ids)
  y=np.array([normal[i]['correct'] for i in ids]);a=np.array([normal[i]['p_correct'] for i in ids]);b=np.array([reverse[i]['p_correct'] for i in ids])
  for fraction in (.2,.5,.8,.9):rows.append(dict(arm=arm,stage=stage,questions=len(ids),**pair_stability(y,a,b,ids),**retention(y,a,b,ids,fraction)))
  ax=axes[k,j];ax.scatter(a,b,c=y,cmap='coolwarm_r',s=3,alpha=.15,vmin=0,vmax=1,rasterized=True)
  ax.plot([0,1],[0,1],color='0.6',lw=.8);ax.set(title=f'{arm} / {stage}',xlim=(0,1),ylim=(0,1));ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(handles=[Line2D([],[],marker='o',ls='',color=plt.cm.coolwarm_r(1.),label='Correct'),Line2D([],[],marker='o',ls='',color=plt.cm.coolwarm_r(0.),label='Incorrect')],frameon=False)
fig.supxlabel('Normal decoded correctness probability');fig.supylabel('Reversed decoded correctness probability');fig.tight_layout();fig.savefig(out/'q_qa_scatter.png',dpi=180);plt.close(fig)
csv_save(out/'binary_stability.csv',rows)
print(out/'q_qa_scatter.png')
