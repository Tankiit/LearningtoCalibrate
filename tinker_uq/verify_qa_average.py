"""Verify cached QA scores and evaluate legend averaging, without API calls."""
from pathlib import Path
import hashlib, json, csv
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from prepare_triviaqa import build
from tinker_uq.training import prompt_text, normalize_scores

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/full_llama31_8b_qwen3_8b'
OUT=RUN/'qa_average_verified'
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((RUN/'manifest.json').read_text())
prep=json.loads((ROOT/'data/full_llama31_8b/prepare_manifest.json').read_text())
source=ROOT.parent/'evaluations_revisited/llama3.1_8b_chat/trivia_qa_60k/base/main_generations_evaluated_revisited.csv'
assert sha(source)==prep['source_sha256']
data=ROOT/'data/full_llama31_8b/cached_answers.jsonl'
assert sha(data)==manifest['data_sha256']
rows=[json.loads(s) for s in data.read_text().splitlines()]
rebuilt,_=build(source,prep['generator_model'],prep['dataset'],prep['grading_column'],list(prep['split_fractions'].values()),prep['split_salt'],0,0)
assert rows==rebuilt, 'Source grade/answer/question reconstruction mismatch'
byid={r['row_id']:r for r in rows}
assert len(byid)==len(rows)==len({r['question_id'] for r in rows})
audit=json.loads((RUN/'prompt_audit.json').read_text())
assert audit['qa']['text']==prompt_text(rows[0],'qa','normal')
assert manifest['label_token_ids']==[[32,198],[33,198]]
groups=defaultdict(dict); maxerr=0.; count=0
with (RUN/'predictions.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        if r['arm']!='qa': continue
        ref=byid[r['row_id']]
        assert all(r[k]==ref[k] for k in ('question_id','split','correct'))
        assert np.isfinite(r['label_logprobs']).all()
        p,lm=normalize_scores(r['label_logprobs'],r['legend'])
        maxerr=max(maxerr,abs(p-r['p_correct']))
        assert abs(p-r['p_correct'])<1e-12 and abs(lm-r['allowed_label_logmass'])<1e-12
        key=(r['stage'],r['legend'],r['split'])
        assert r['row_id'] not in groups[key]
        groups[key][r['row_id']]=r
        count+=1
for stage in ('base','trained'):
    for legend in ('normal','reversed'):
        for split in ('train','val','cal','test'):
            assert set(groups[stage,legend,split])=={r['row_id'] for r in rows if r['split']==split}

def metrics(y,p):
    n=len(y); npos=y.sum(); ranks=rankdata(p)
    auc=(ranks[y==1].sum()-npos*(npos+1)/2)/(npos*(n-npos))
    order=np.argsort(-p,kind='stable'); scores=p[order]; err=1-y[order]
    starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
    sizes=np.diff(np.r_[starts,n]); rates=np.add.reduceat(err,starts)/sizes
    expected=np.repeat(rates,sizes)
    aurc=np.mean(np.cumsum(expected)/np.arange(1,n+1))
    return np.array([np.mean((p-y)**2),auc,aurc])

records=[]; summary=[]; comparisons=[]
for stage in ('base','trained'):
    ids=sorted(groups[stage,'normal','test']); a=groups[stage,'normal','test']; b=groups[stage,'reversed','test']
    y=np.array([a[i]['correct'] for i in ids]); normal=np.array([a[i]['p_correct'] for i in ids]); rev=np.array([b[i]['p_correct'] for i in ids])
    scores=[normal,rev,(normal+rev)/2]; names=['normal','reversed','averaged']
    observed=np.array([metrics(y,p) for p in scores])
    rng=np.random.default_rng(20260914); boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y)); boot.append([metrics(y[ix],p[ix]) for p in scores])
    boot=np.array(boot)
    for j,name in enumerate(names):
        entry={'stage':stage,'score':name,'questions':len(y)}
        for k,metric in enumerate(('Brier','AUROC','AURC')):
            entry[metric]=float(observed[j,k]); entry[metric+'_ci95']=np.quantile(boot[:,j,k],[.025,.975]).tolist()
        summary.append(entry)
    for j in (0,1):
        entry={'stage':stage,'comparison':'averaged_minus_'+names[j]}
        for k,metric in enumerate(('Brier','AUROC','AURC')):
            entry[metric]=float(observed[2,k]-observed[j,k]); entry[metric+'_ci95']=np.quantile(boot[:,2,k]-boot[:,j,k],[.025,.975]).tolist()
        comparisons.append(entry)
    for idx,rid in enumerate(ids):
        records.append({'row_id':rid,'question_id':a[rid]['question_id'],'split':'test','arm':'qa','stage':stage,'correct':int(y[idx]),'normal':float(normal[idx]),'reversed':float(rev[idx]),'averaged':float(scores[2][idx]),'normal_label_logprobs':a[rid]['label_logprobs'],'reversed_label_logprobs':b[rid]['label_logprobs']})
    print(stage,observed.tolist(),flush=True)
with (OUT/'verified_predictions.jsonl').open('w') as f:
    for r in records: f.write(json.dumps(r)+'\n')
result={'verification':{'qa_rows_checked':count,'max_probability_reconstruction_error':maxerr,'source_sha256':sha(source),'predictions_sha256':sha(RUN/'predictions.jsonl'),'data_sha256':sha(data),'source_rows_rebuilt_exactly':True,'normal_prompt_example_matches':True},'bootstrap':{'draws':2000,'seed':20260914,'unit':'test question','paired':True},'metrics':summary,'paired_differences':comparisons}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
for filename,items in [('metrics.csv',summary),('paired_differences.csv',comparisons)]:
    with (OUT/filename).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0])); w.writeheader(); w.writerows(items)
text='# Verified answer-aware legend-average baseline\n\n'
text+='Existing Qwen3-8B QA reporter on cached Llama 3.1 8B TriviaQA answers; base and trained checkpoints, 10,647 test questions per stage. No new training or inference.\n\n'
text+='| Stage | Score | Brier | AUROC | AURC |\n|---|---|---:|---:|---:|\n'
for r in summary: text+=f"| {r['stage']} | {r['score']} | {r['Brier']:.6f} | {r['AUROC']:.6f} | {r['AURC']:.6f} |\n"
text+='\nPaired differences (average minus individual; negative favors average for Brier/AURC, positive for AUROC). 95% percentile intervals from 2,000 shared question bootstrap draws; conditional on the saved checkpoint, not training-seed uncertainty.\n\n| Stage | Comparison | Brier Δ [95% CI] | AUROC Δ [95% CI] | AURC Δ [95% CI] |\n|---|---|---|---|---|\n'
for r in comparisons:
    text+=f"| {r['stage']} | {r['comparison']} | "+' | '.join(f"{r[m]:.6f} [{r[m+'_ci95'][0]:.6f}, {r[m+'_ci95'][1]:.6f}]" for m in ('Brier','AUROC','AURC'))+' |\n'
text+='\n## Protocol and verification\n\nBoth completions are full sequences `A\\n`, `B\\n`, token IDs `[32,198]`, `[33,198]`. Saved label log-likelihoods are in that order. Normal correctness probability is exp(L_A)/[exp(L_A)+exp(L_B)]; reversed correctness probability uses L_B in the numerator. The averaged score is (p_normal+p_reversed)/2 after decoding. These probabilities are conditional on the two allowed strings.\n\n'
text+='QA user-message template (current runner):\n\n```text\n'+prompt_text({'question':'{question}','answer':'{answer}'},'qa','normal')+'\n```\n\nReversed replaces the legend line with `Report legend: B = correct; A = incorrect.` The runner applies the tokenizer chat template with generation prompt enabled and thinking disabled. The saved audit includes only the normal example; reversed prompt provenance is reconstructed from the current code, not independently captured token-by-token in the historical artifact. No API replay was performed.\n\n'
text+=f"Verified {count:,} QA prediction rows across all splits, stages and legends against cached IDs, split membership, binary grades and sequence-score decoding (maximum probability error {maxerr:g}). Rebuilt all {len(rows):,} answer records exactly from the SHA-256-matched source CSV using `correct_revisited`; this verifies grade provenance, not an independent regrading of answer truth. Matched cached data hash to training manifest and normal example text to saved prompt audit.\n\n"
text+='Brier is mean squared probability error. AUROC uses average ranks for ties. AURC is mean cumulative error over descending confidence, averaging uniformly within exact-score ties. Test questions are unique, so bootstrap questions equal bootstrap rows. Verified test predictions, score intervals and paired intervals are saved alongside this report.\n'
(OUT/'REPORT.md').write_text(text)
