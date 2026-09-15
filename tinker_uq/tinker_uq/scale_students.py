"""Matched students after fixed-checkpoint validation: distillation and supervision controls."""
import argparse,json,random,time,concurrent.futures
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import load_jsonl,read_rows,write_json,write_jsonl,digest
from .reporting import align_to_values
from .training import make_batch,loss_function
from .matched_experiment import METHODS,score_file,codebook_scores

COARSE=('both_mean_brier','aligned_distill','aligned_distill_supervised')

def prepare(out):
    out=Path(out)
    if not (out/'VALIDATION_COMPLETE.json').exists():raise ValueError('Complete validation before another student')
    m=json.loads((out/'manifest.json').read_text())
    plan=dict(schema='scale-students-v1',initial_state=m['state_path'],initial_sampler=m['sampler_path'],model=m['model'],
        teacher='same existing QA checkpoint; six value-aligned distributions averaged per training question',
        methods={'coarse':list(COARSE),'fine':list(METHODS)},seed=0,rank=16,lr=1e-4,batch_size=8,epochs=1,
        optimizer_steps=16,consistency_weight=1.,supervision_weight=1.,training_legends=['normal','reversed'],
        data_sha256=m['data_sha256'],mappings_sha256=m['mappings_sha256'],
        supervision='All start from identical previously correctness-trained QA weights with reset optimizer. Pure distillation adds no new labels; supervised distillation adds the identical decoded-mean Brier term used by correctness-only control.',
        matching='128 identical supplied training answers per method/scale; same question order, 16 updates, rank16, LR1e-4; two views per question, normal-only repeats normal',
        teacher_mapping_scope='coarse exhaustive six; fine approximate six. Student sees normal/reversed only. Fine evaluation includes four further mappings absent even from teacher averaging.',
        validation_scope='validation comparison completed before preparation; no hyperparameter/selector tuning from observed validation results')
    write_json(out/'student_plan.json',plan)
    print('Frozen student plan',flush=True)


def run(out,plan_filename="student_plan.json",workers=1):
    import tinker
    out=Path(out);p=json.loads((out/plan_filename).read_text());maps=json.loads((out/'mappings.json').read_text());meta=json.loads((out/'manifest.json').read_text())
    assert digest(out/'data.jsonl')==p['data_sha256'] and digest(out/'mappings.json')==p['mappings_sha256']
    rows=read_rows(out/'data.jsonl');train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
    service=tinker.ServiceClient(timeout=60,max_retries=2);teacher_client=service.create_sampling_client(model_path=p['initial_sampler']);tok=teacher_client.get_tokenizer()
    for scale,methods in p['methods'].items():
        folder=out/'students'/scale;folder.mkdir(parents=True,exist_ok=True);family=maps[scale];teacher_maps={k:family[k] for k in meta['initial_mapping_names'][scale]}
        td=folder/'teacher_train';td.mkdir(exist_ok=True);score_file(teacher_client,train,tok,'teacher',td,tinker,teacher_maps)
        collected=defaultdict(list)
        for name in teacher_maps:
            for r in load_jsonl(td/f'{name}.jsonl'):collected[r['row_id']].append(align_to_values(r['probability_vector'],r['mapping']))
        if len(collected)!=len(train) or any(len(v)!=6 for v in collected.values()):raise ValueError('Missing teacher distributions')
        teacher={rid:np.mean(v,axis=0).tolist() for rid,v in collected.items()};write_json(folder/'aligned_teacher.json',teacher)
        def train_one(method):
            fd=folder/method;fd.mkdir(exist_ok=True)
            if (fd/'checkpoint.json').exists():return
            client=service.create_training_client_from_state(p['initial_state'])
            shuffled=list(train);random.Random(p['seed']).shuffle(shuffled);objective,names=METHODS[method];codebooks=[family[n] for n in names];logs=[]
            for offset in range(0,len(train),p['batch_size']):
                batch=shuffled[offset:offset+p['batch_size']];data,starts,labels=make_batch(batch,tok,'qa',tinker,mappings=codebooks)
                objective_fn=loss_function(starts,labels,mappings=codebooks,objective=objective,consistency_weight=p['consistency_weight'],supervision_weight=p['supervision_weight'],
                    teacher=[teacher[r['row_id']] for r in batch] if objective.startswith('distill') else None)
                before=time.time();result=client.forward_backward_custom(data,objective_fn).result();client.optim_step(tinker.AdamParams(learning_rate=p['lr'])).result()
                logs.append(dict(step=len(logs),question_ids=[r['question_id'] for r in batch],training_mappings=names,sequences=len(data),input_tokens=sum(len(d.model_input.to_ints()) for d in data),metrics=result.metrics,seconds=time.time()-before))
                write_jsonl(fd/'training.jsonl',logs);print(scale,method,f'{len(logs)}/16',result.metrics,flush=True)
            state=client.save_state(f'{scale}-{method}-final').result();saved=client.save_weights_for_sampler(f'{scale}-{method}-sampler').result()
            write_json(fd/'checkpoint.json',dict(state_path=state.path,sampler_path=saved.path,steps=len(logs)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(train_one,methods))
        # Teacher test scores are also fresh and never used for optimization.
        fd=folder/'teacher';fd.mkdir(exist_ok=True);score_file(teacher_client,test,tok,'teacher',fd,tinker,family)
        def evaluate_one(method):
            fd=folder/method;sampler=service.create_sampling_client(model_path=json.loads((fd/'checkpoint.json').read_text())['sampler_path'])
            score_file(sampler,test,tok,method,fd,tinker,family);codebook_scores(sampler,tok,family,fd/'codebook.json',tinker)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(evaluate_one,methods))
    from .student_analysis import analyze
    analyze(out,plan_filename=plan_filename);write_json(out/'STUDENTS_COMPLETE.json',dict(complete=True,plan=plan_filename))


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','analyze']);a.add_argument('--out',required=True);a.add_argument('--plan',default='student_plan.json');a.add_argument('--workers',type=int,default=1);args=a.parse_args()
    if args.action=='prepare':prepare(args.out)
    elif args.action=='run':run(args.out,args.plan,args.workers)
    else:
        from .student_analysis import analyze
        analyze(Path(args.out))
if __name__=='__main__':main()
