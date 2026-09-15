"""Small supplied-answer DINCO adaptation with frozen auxiliary generations."""
import argparse,json,re,concurrent.futures
from pathlib import Path
import numpy as np
from .data import load_jsonl,write_json,write_jsonl
from .emitted_reports import Generator,render,parse
from .reporting import digest
from .dinco_baseline import scores,sc_from_entailment

NLI_MODEL='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

def clean(text):
 return re.sub(r'\s+',' ',text.split('\n')[0].replace('Answer:','').strip())

def run(source,out,n=32):
 source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((source/'manifest.json').read_text());maps=json.loads((source/'mappings.json').read_text());rows=load_jsonl(source/'data.jsonl')[:n]
 plan=dict(n=n,source=str(source),question_ids=[r['question_id'] for r in rows],reporter=manifest['sampler_path'],auxiliary_model='provider-default:Qwen/Qwen3-8B',
   distractor_generation='one strict JSON list of five plausible candidate answers; black-box candidate-list adaptation',self_consistency_samples=5,
   sc_policy=dict(temperature=1.,top_p=.95,max_tokens=100),nli_model=NLI_MODEL,formats=manifest['formats'],scales=manifest['scales'],
   adaptation='fixed supplied Llama answer at index 0; base Qwen auxiliary generator, frozen across all reporter formats; no new main answer selection',
   scope='32-question development implementation pilot, not a completed 1000-question comparison',
   sc_semantics='public runner: mean of five bidirectional entailment scores, exact string matches=1; no automatically included self-sample for externally supplied answer')
 if (out/'protocol.json').exists():
  if json.loads((out/'protocol.json').read_text())!=plan:raise ValueError('Frozen DINCO protocol mismatch')
 else:write_json(out/'protocol.json',plan);write_jsonl(out/'data.jsonl',rows)
 aux=Generator(base_model='Qwen/Qwen3-8B');reporter=Generator(manifest['sampler_path']);auxpath=out/'auxiliary.jsonl';old=load_jsonl(auxpath) if auxpath.exists() else [];done={r['question_id']:r for r in old}
 def generate(row):
  q=row['question'];prompt='Question: '+q+'\nList five distinct plausible short answers to this question, ordered by plausibility. Reply only with a JSON array of five strings. Do not explain.'
  distractor=aux.generate(prompt,dict(temperature=0.,max_tokens=256,seed=0))
  try:
   candidates=json.loads(distractor['raw_response']);valid=isinstance(candidates,list) and len(candidates)==5 and all(isinstance(x,str) and x.strip() for x in candidates)
  except (ValueError,TypeError):candidates=[];valid=False
  normalized=set();filtered=[]
  for x in candidates if valid else []:
   text=clean(x);key=text.lower().replace('.','')
   if key and key not in normalized and key!=clean(row['answer']).lower().replace('.',''):filtered.append(text);normalized.add(key)
  sc=[aux.generate('Answer with a concise phrase only.\nQuestion: '+q+'\nAnswer:',dict(temperature=1.,top_p=.95,max_tokens=100,seed=j)) for j in range(5)]
  return dict(question_id=row['question_id'],fixed_answer=row['answer'],candidates=[row['answer']]+filtered,valid_distractor_list=valid,distractor_generation=distractor,self_consistency_generations=sc)
 with auxpath.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for r in pool.map(generate,[r for r in rows if r['question_id'] not in done]):f.write(json.dumps(r)+'\n');f.flush();done[r['question_id']]=r;print('DINCO auxiliary',len(done),'/',n,flush=True)
 # NLI weights are computed once; semantic labels are verified from the model config.
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 cache=str(out/'model_cache');tok=AutoTokenizer.from_pretrained(NLI_MODEL,cache_dir=cache);model=AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,cache_dir=cache).eval()
 labels={int(k):v.lower() for k,v in model.config.id2label.items()}
 if labels!={0:'entailment',1:'neutral',2:'contradiction'}:raise ValueError(f'Unexpected NLI labels: {labels}')
 write_json(out/'nli_provenance.json',dict(model=NLI_MODEL,commit=getattr(model.config,'_commit_hash',None),labels=labels,softmax_dtype='float32',prior_float16_nli_cache='preserved but not used; failed probability-sum audit'))
 def infer(question,pairs):
  answers=[]
  for start in range(0,len(pairs),16):
   batch=pairs[start:start+16];inputs=tok([f'Question: {question}\nAnswer: {p}' for p,h in batch],[f'Answer: {h}' for p,h in batch],padding=True,return_tensors='pt',truncation=True,max_length=512)
   with torch.inference_mode():answers.extend(torch.softmax(model(**inputs).logits.float(),-1).tolist())
  return answers
 nlipath=out/'nli_float32.jsonl';nlis={r['question_id']:r for r in load_jsonl(nlipath)} if nlipath.exists() else {}
 with nlipath.open('a') as f:
  for row in rows:
   qid=row['question_id']
   if qid in nlis:continue
   a=done[qid];cs=a['candidates'];matrix=np.zeros((len(cs),len(cs),3));matrix[np.arange(len(cs)),np.arange(len(cs)),0]=1
   indices=[(i,j) for i in range(len(cs)) for j in range(len(cs)) if i!=j]
   for ij,p in zip(indices,infer(row['question'],[(cs[i],cs[j]) for i,j in indices])):matrix[ij]=p
   sampled=[clean(s['raw_response']) for s in a['self_consistency_generations']];main=clean(row['answer']);pairs=[pair for s in sampled for pair in [(main,s),(s,main)]]
   entail=np.array(infer(row['question'],pairs))[:,0].reshape(5,2);sc=sc_from_entailment(entail,[main==s for s in sampled]);r=dict(question_id=qid,nli=matrix.tolist(),self_consistency=sc,bidirectional_entailment=entail.tolist(),auxiliary_hash=digest(a))
   f.write(json.dumps(r)+'\n');f.flush();nlis[qid]=r;print('DINCO NLI',len(nlis),'/',n,flush=True)
 main={(r['question_id'],r['scale'],r['format']):r for r in load_jsonl(source/'reports.jsonl')};path=out/'reports.jsonl';existing=load_jsonl(path) if path.exists() else [];finished={(r['question_id'],r['scale'],r['format']) for r in existing}
 def one(task):
  row,scale,fmt=task;key=(row['question_id'],scale,fmt)
  a=done[row['question_id']];mapping=maps[scale]['normal' if fmt=='direct' else fmt];calls=[main[key]]
  for candidate in a['candidates'][1:]:
   r=reporter.generate(render(row['question'],candidate,mapping,fmt=='direct'));r['emitted_value']=parse(r['raw_response'],mapping,fmt=='direct');calls.append(r)
  p=[r['emitted_value'] for r in calls];valid=a['valid_distractor_list'] and all(v is not None for v in p);ni=nlis[row['question_id']]
  result=scores(p,ni['nli'],ni['self_consistency']) if valid else None
  return dict(question_id=row['question_id'],correct=row['correct'],scale=scale,format=fmt,valid=valid,scores=result,raw_emission=p[0],self_consistency=ni['self_consistency'],candidate_reports=calls,auxiliary_hash=ni['auxiliary_hash'])
 tasks=[(row,scale,fmt) for row in rows for scale in manifest['scales'] for fmt in manifest['formats'] if (row['question_id'],scale,fmt) not in finished]
 with path.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for k,record in enumerate(pool.map(one,tasks),1):
   f.write(json.dumps(record)+'\n');f.flush()
   if k%12==0:print('DINCO report bundles',len(finished)+k,'/',n*6,flush=True)
 # Final audited scores always use the same float32 NLI/SC cache, including resumed records.
 final=load_jsonl(path)
 for r in final:
  ni=nlis[r['question_id']];r['self_consistency']=ni['self_consistency']
  if r['valid']:r['scores']=scores([c['emitted_value'] for c in r['candidate_reports']],ni['nli'],ni['self_consistency'])
 write_jsonl(out/'verified_reports.jsonl',final)
 write_json(out/'STATUS.json',dict(complete=True,n=n,scope='development implementation pilot; no claim of exact reproduction'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',default='runs/emitted_validation_v1');p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=32);a=p.parse_args();run(a.source,a.out,a.n)
