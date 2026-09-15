"""Frozen six-arm letter11 objective pilot; restart skips completed checkpoints/groups."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
from .data import read_rows, write_json, write_jsonl, load_jsonl, digest as file_digest, audit
from .reporting import freeze_mappings, digest, align_to_values, codebook_prompt, decode
from .training import completions, encode_prompt, make_batch, loss_function, score_rows

METHODS = {
 'normal_token_brier': ('token_brier',['normal','normal']),
 'both_token_brier': ('token_brier',['normal','reversed']),
 'both_mean_brier': ('mean_brier',['normal','reversed']),
 'mean_consistency': ('mean_consistency',['normal','reversed']),
 'aligned_distill': ('distill',['normal','reversed']),
 'aligned_distill_supervised': ('distill_supervised',['normal','reversed'])}

def prepare(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    rows=read_rows(args.data); selected=[]
    for split,n in {'train':args.train_n,'val':32,'cal':32,'test':args.test_n}.items():
        group=sorted([r for r in rows if r['split']==split],key=lambda r:digest(['matched-objectives-v1',r['question_id']]))
        if len(group)<n: raise ValueError('Not enough source questions')
        selected.extend(group[:n])
    if len({r['question_id'] for r in selected})!=len(selected): raise ValueError('One answer per question required')
    write_jsonl(out/'data.jsonl',selected); maps=freeze_mappings(); write_json(out/'mappings.json',maps)
    manifest=dict(schema='matched-letter11-objectives-v1',model='Qwen/Qwen3-8B',arm='qa',seed=0,rank=16,lr=1e-4,
        epochs=1,batch_size=8,max_tokens=4096,consistency_weight=1.,supervision_weight=1.,
        teacher='frozen_base_normal_reversed_value_aligned_average',data_source=str(Path(args.data).resolve()),
        source_sha256=file_digest(args.data),data_sha256=file_digest(out/'data.jsonl'),mappings_sha256=file_digest(out/'mappings.json'),
        methods={k:dict(objective=v[0],training_mappings=v[1]) for k,v in METHODS.items()},
        training_mapping_names=['normal','reversed'],heldout_mapping_names=list(maps)[2:],audit=audit(selected),
        optimizer_steps=(args.train_n+7)//8,
        budget='same questions/order/updates and two views x 11 complete sequences per question; normal-only duplicates normal',
        inference='full letter+newline sequence likelihood; conditional alphabet distribution',
        selection='smallest SHA256(question ID with fixed namespace) within original split; label independent',
        claims='exploratory single-seed pilot; eight fixed held-out mappings, not all 11!; no test tuning',
        versions={p:importlib.metadata.version(p) for p in ('tinker','torch','transformers')})
    write_json(out/'manifest.json',manifest); print(f'Frozen experiment: {out}',flush=True)

def score_file(client,rows,tokenizer,stage,path,tinker,mappings):
    for name,mapping in mappings.items():
        target=path/f'{name}.jsonl'
        if target.exists():
            saved=load_jsonl(target)
            if len(saved)!=len(rows) or {r['row_id'] for r in saved}!={r['row_id'] for r in rows}: raise ValueError('Incomplete saved group')
            continue
        temp=target.with_suffix('.partial')
        with temp.open('w') as f: score_rows(client,rows,tokenizer,'qa',stage,f,tinker,window=16,mappings={name:mapping})
        temp.replace(target)

def codebook_scores(client,tokenizer,mappings,path,tinker):
    if path.exists(): return
    output=[]
    for name,mapping in mappings.items():
        cs=completions(tokenizer,mapping)
        for value in sorted(mapping.values()):
            text=codebook_prompt(mapping,value)
            prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            futures=[client.compute_logprobs(tinker.ModelInput.from_ints(prompt+c)) for c in cs]; scores=[]
            for future,c in zip(futures,cs):
                lp=future.result()
                if len(lp)!=len(prompt)+len(c) or any(x is None for x in lp[len(prompt):]): raise ValueError('Missing codebook logprobs')
                scores.append(sum(lp[len(prompt):]))
            result=decode(scores,mapping); expected=next(k for k,v in mapping.items() if v==value)
            argmax=result['letters'][max(range(len(cs)),key=lambda i:result['probability_vector'][i])]
            result.update(mapping_name=name,target_value=value,expected_letter=expected,argmax_letter=argmax,
                codebook_correct=argmax==expected,expected_letter_probability=result['probability_vector'][result['letters'].index(expected)],
                numerical_absolute_error=abs(result['decoded_mean']-value),prompt_provenance={'text':text,'token_ids_sha256':digest(prompt)})
            output.append(result)
        print(f'codebook {path.parent.name}/{name}',flush=True)
    write_json(path,output)

def run(args):
    import numpy as np
    import tinker
    out=Path(args.out); manifest=json.loads((out/'manifest.json').read_text())
    if file_digest(out/'data.jsonl')!=manifest['data_sha256'] or file_digest(out/'mappings.json')!=manifest['mappings_sha256']: raise ValueError('Frozen inputs changed')
    rows=read_rows(out/'data.jsonl'); maps=json.loads((out/'mappings.json').read_text())
    train=[r for r in rows if r['split']=='train']; test=[r for r in rows if r['split']=='test']
    service=tinker.ServiceClient(timeout=60,max_retries=2)
    base=service.create_sampling_client(base_model=manifest['model']); tokenizer=base.get_tokenizer()
    if not tokenizer.chat_template: raise ValueError('Missing chat template')
    for r in rows:
        for m in maps.values():
            if len(encode_prompt(tokenizer,r,'qa',mapping=m))+max(map(len,completions(tokenizer,m)))>manifest['max_tokens']: raise ValueError('Overlong input')
    write_json(out/'runtime.json',dict(chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        completion_token_ids=completions(tokenizer,maps['normal']),source_code_sha256={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')}))
    teacher_dir=out/'teacher_train'; teacher_dir.mkdir(exist_ok=True)
    score_file(base,train,tokenizer,'base',teacher_dir,tinker,{k:maps[k] for k in ('normal','reversed')})
    distributions=defaultdict(list)
    for name in ('normal','reversed'):
        for r in load_jsonl(teacher_dir/f'{name}.jsonl'): distributions[r['row_id']].append(align_to_values(r['probability_vector'],r['mapping']))
    if any(len(x)!=2 for x in distributions.values()): raise ValueError('Teacher pair missing')
    teacher={rid:np.mean(x,axis=0).tolist() for rid,x in distributions.items()}; write_json(out/'aligned_teacher.json',teacher)
    for method,(objective,names) in METHODS.items():
        folder=out/method; folder.mkdir(exist_ok=True)
        if (folder/'checkpoint.json').exists(): continue
        client=service.create_lora_training_client(base_model=manifest['model'],rank=manifest['rank'],seed=manifest['seed'])
        shuffled=list(train); random.Random(manifest['seed']).shuffle(shuffled); codebooks=[maps[n] for n in names]; log=[]
        for offset in range(0,len(shuffled),manifest['batch_size']):
            batch=shuffled[offset:offset+manifest['batch_size']]
            data,starts,labels=make_batch(batch,tokenizer,'qa',tinker,mappings=codebooks)
            objective_fn=loss_function(starts,labels,mappings=codebooks,objective=objective,
                consistency_weight=manifest['consistency_weight'],supervision_weight=manifest['supervision_weight'],
                teacher=[teacher[r['row_id']] for r in batch] if objective.startswith('distill') else None)
            before=time.time(); result=client.forward_backward_custom(data,objective_fn).result()
            client.optim_step(tinker.AdamParams(learning_rate=manifest['lr'])).result()
            log.append(dict(step=offset//manifest['batch_size'],question_ids=[r['question_id'] for r in batch],training_mappings=names,
                sequences=len(data),input_tokens=sum(len(d.model_input.to_ints()) for d in data),metrics=result.metrics,seconds=time.time()-before))
            write_jsonl(folder/'training.jsonl',log); print(f'{method}: step {len(log)}/{manifest["optimizer_steps"]} {result.metrics}',flush=True)
        state=client.save_state(f'{method}-final').result(); saved=client.save_weights_for_sampler(f'{method}-sampler').result()
        write_json(folder/'checkpoint.json',dict(state_path=state.path,sampler_path=saved.path,steps=len(log)))
    for method in ['base']+list(METHODS):
        folder=out/method; folder.mkdir(exist_ok=True)
        client=base if method=='base' else service.create_sampling_client(model_path=json.loads((folder/'checkpoint.json').read_text())['sampler_path'])
        score_file(client,test,tokenizer,method,folder,tinker,maps); codebook_scores(client,tokenizer,maps,folder/'codebook.json',tinker)
    from .matched_analysis import analyze
    analyze(out); write_json(out/'COMPLETE.json',{'complete':True,'methods':list(METHODS),'test_questions':len(test)})

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','run','analyze']); p.add_argument('--out',required=True)
    p.add_argument('--data'); p.add_argument('--train-n',type=int,default=128); p.add_argument('--test-n',type=int,default=128); a=p.parse_args()
    if a.action=='prepare': prepare(a)
    elif a.action=='run': run(a)
    else:
        from .matched_analysis import analyze
        analyze(Path(a.out))
if __name__=='__main__': main()
