"""DSPy GEPA with one mutable instruction and an immutable emission wrapper."""
import argparse,json,os,threading
from pathlib import Path
os.environ.setdefault('DSPY_CACHEDIR','/private/tmp/confidence-dspy-cache')
import dspy
import numpy as np
from .emitted_reports import Generator,render,parse,contract,INSTRUCTION
from .reporting import digest
from .data import load_jsonl,write_json,write_jsonl

class Report(dspy.Signature):
    question:str=dspy.InputField()
    answer:str=dspy.InputField()
    reporting_contract:str=dspy.InputField()
    report:str=dspy.OutputField()
Report=Report.with_instructions(INSTRUCTION)

class FixedAdapter(dspy.Adapter):
    def __init__(self,generator,path):
        super().__init__();self.generator=generator;self.path=Path(path);self.lock=threading.Lock()
    def __call__(self,lm,lm_kwargs,signature,demos,inputs):
        if demos:raise ValueError('Few-shot scaffolding is outside the frozen program')
        text=signature.instructions+'\nQuestion:\n'+inputs['question']+'\nCandidate answer:\n'+inputs['answer']+'\n'+inputs['reporting_contract']
        result=self.generator.generate(text)
        with self.lock,self.path.open('a') as f:f.write(json.dumps(dict(result,kind='target',instruction=signature.instructions))+'\n')
        return [{'report':result['raw_response']}]

class Program(dspy.Module):
    def __init__(self,family):super().__init__();self.reporter=dspy.Predict(Report);self.family=family
    def forward(self,question,answer):
        reports=[]
        for fmt in ('direct','normal','reversed'):
            mapping=self.family['normal' if fmt=='direct' else fmt]
            raw=self.reporter(question=question,answer=answer,reporting_contract=contract(mapping,fmt=='direct')).report
            reports.append(dict(format=fmt,raw=raw,value=parse(raw,mapping,fmt=='direct')))
        return dspy.Prediction(reports=reports)

def metric_function(weight):
    def metric(gold,pred,trace=None,pred_name=None,pred_trace=None):
        values=[r['value'] for r in pred.reports];valid=all(v is not None for v in values)
        brier=float(np.mean([(v-gold.correct)**2 if v is not None else 1. for v in values]))
        # Invalidity receives the maximal feasible variance penalty, with no fabricated value.
        variance=float(np.var(values)) if valid else .25
        score=1-(brier+weight*variance)/(1+weight*.25)
        return dspy.Prediction(score=score,feedback=json.dumps(dict(correctness_label=gold.correct,emissions=pred.reports,mean_per_format_brier=brier,variance=variance,invalid_penalty=not valid,
            instruction_constraint='Only improve confidence-elicitation instructions. The supplied answer, grid, legend, parser, and independent contexts are fixed.')))
    return metric

def run(source,out,scale='coarse',budget=64):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text());maps=json.loads((source/'mappings.json').read_text())
    root=Path(__file__).resolve().parents[1];rows=load_jsonl(root/'data/full_llama31_8b/cached_answers.jsonl')
    train=sorted((r for r in rows if r['split']=='train'),key=lambda r:digest(['gepa-reflection-v1',r['question_id']]))[:16]
    validation=sorted((r for r in load_jsonl(source/'data.jsonl') if int(digest(['selection-fit-v1',r['question_id']]),16)%2==0),key=lambda r:digest(['gepa-selection-v1',r['question_id']]))[:8]
    write_jsonl(out/'train.jsonl',train);write_jsonl(out/'validation.jsonl',validation)
    write_json(out/'protocol.json',dict(scale=scale,max_metric_calls=budget,train_questions=len(train),selection_questions=len(validation),
        target_checkpoint=manifest['sampler_path'],reflection_checkpoint='provider-default:Qwen/Qwen3-8B',reflection_max_tokens=2048,
        objectives={'quality':0.,'quality_consistency':1.},format_exposure=['direct','normal','reversed'],
        scope='small optimizer plumbing pilot, not a definitive prompt-search budget; selected on development only',
        invalid_penalty='Brier 1 and variance .25, no output repair',training_prevalence=float(np.mean([r['correct'] for r in rows if r['split']=='train']))))
    def examples(rs):return [dspy.Example(question=r['question'],answer=r['answer'],correct=r['correct'],question_id=r['question_id']).with_inputs('question','answer') for r in rs]
    generator=Generator(manifest['sampler_path']);reflector=Generator(base_model='Qwen/Qwen3-8B')
    for name,weight in [('quality',0.),('quality_consistency',1.)]:
        folder=out/name;folder.mkdir();log=folder/'calls.jsonl';adapter=FixedAdapter(generator,log)
        def reflection(prompt):
            result=reflector.generate(prompt,dict(temperature=.7,max_tokens=2048,seed=0))
            with log.open('a') as f:f.write(json.dumps(dict(result,kind='reflection'))+'\n')
            return [result['raw_response']]
        dspy.configure(lm=dspy.BaseLM(model='tinker/'+manifest['sampler_path'],cache=False),adapter=adapter)
        optimizer=dspy.GEPA(metric=metric_function(weight),max_metric_calls=budget,reflection_lm=reflection,reflection_minibatch_size=3,
            use_merge=False,num_threads=2,seed=0,track_stats=True,skip_perfect_score=False,log_dir=str(folder/'gepa'))
        best=optimizer.compile(Program(maps[scale]),trainset=examples(train),valset=examples(validation))
        best.save(str(folder/'program.json'))
        stats=best.detailed_results
        calls=load_jsonl(log)
        write_json(folder/'result.json',dict(instruction=best.reporter.signature.instructions,
            validation_scores=stats.val_aggregate_scores,best_idx=stats.best_idx,total_metric_calls=stats.total_metric_calls,
            calls={kind:dict(requests=sum(r['kind']==kind for r in calls),input_tokens=sum(r['input_tokens'] for r in calls if r['kind']==kind),generated_tokens=sum(r['generated_tokens'] for r in calls if r['kind']==kind)) for kind in ('target','reflection')}))
    write_json(out/'STATUS.json',dict(complete=True,status='prompt search complete; independent evaluation still required'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='runs/emitted_validation_v1');p.add_argument('--out',required=True);p.add_argument('--scale',choices=['coarse','fine'],default='coarse');p.add_argument('--budget',type=int,default=64);a=p.parse_args();run(a.source,a.out,a.scale,a.budget)
