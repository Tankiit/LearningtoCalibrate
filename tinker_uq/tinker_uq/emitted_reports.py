"""Fresh, unconstrained numerical emissions under frozen reporting contracts."""
import argparse,json,re,hashlib,concurrent.futures
from decimal import Decimal
from pathlib import Path
from .data import load_jsonl,write_json,write_jsonl,digest as file_digest
from .reporting import mapping_items,digest

INSTRUCTION='Judge whether the supplied candidate answer is correct. Report the probability that this answer is correct.'
POLICY=dict(temperature=0.0,max_tokens=32,seed=0,stop=['\n'])

def contract(mapping,direct=False):
    items=mapping_items(mapping)
    if direct:
        return 'Permitted confidence percentages: '+', '.join(f'{v*100:g}%' for v in sorted(mapping.values()))+'.\nReply with one permitted percentage followed by a newline.'
    return 'Report legend: '+'  '.join(f'{k}={v*100:g}%' for k,v in items)+'\nReply with one letter followed by a newline.'

def render(question,answer,mapping,direct=False,instruction=INSTRUCTION):
    return instruction+'\nQuestion:\n'+question+'\nCandidate answer:\n'+answer+'\n'+contract(mapping,direct)

def parse(raw,mapping,direct=False,*,contract_name="strict_percent_v1"):
    if contract_name not in ('strict_v1','strict_percent_v1','optional_percent_v1','lenient_v1'):raise ValueError('Unknown parser contract')
    text=raw.strip()  # whitespace only: never extract a substring or repair a symbol
    if not direct:return mapping.get(text) if text in mapping else None
    if contract_name=='lenient_v1':
        if not re.fullmatch(r'(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?',text):return None
        value=Decimal(text.removesuffix('%'))
        if text.endswith('%') or value>1:value=value/100
        return next((v for v in mapping.values() if Decimal(str(v))==value),None)
    pattern=r'(?:0|[1-9]\d*)(?:\.\d+)?'+('%' if contract_name in ('strict_v1','strict_percent_v1') else '%?')
    if not re.fullmatch(pattern,text):return None
    value=float(text.removesuffix('%'))/100
    return next((v for v in mapping.values() if abs(v-value)<1e-10),None)

class Generator:
    def __init__(self,sampler_path=None,base_model=None):
        import tinker
        self.tinker=tinker;self.service=tinker.ServiceClient(timeout=60,max_retries=2)
        self.sampler=self.service.create_sampling_client(model_path=sampler_path) if sampler_path else self.service.create_sampling_client(base_model=base_model)
        self.tokenizer=self.sampler.get_tokenizer();self.path=sampler_path or 'provider-default:'+base_model
    def generate(self,text,policy=None):
        policy=policy or POLICY
        tokens=self.tokenizer.apply_chat_template([{'role':'user','content':text}],enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
        result=self.sampler.sample(self.tinker.types.ModelInput.from_ints(tokens),num_samples=1,sampling_params=self.tinker.types.SamplingParams(**policy)).result()
        sample=result.sequences[0] if hasattr(result,'sequences') else result.samples[0]
        raw=self.tokenizer.decode(sample.tokens,skip_special_tokens=True)
        return dict(raw_response=raw,raw_response_with_special_tokens=self.tokenizer.decode(sample.tokens,skip_special_tokens=False),output_token_ids=sample.tokens,
            stop_reason=str(sample.stop_reason),prompt_text=text,prompt_text_sha256=hashlib.sha256(text.encode()).hexdigest(),prompt_token_ids=tokens,full_prompt_hash=digest(tokens),
            input_tokens=len(tokens),generated_tokens=len(sample.tokens),sampling_policy=policy,model_revision=self.path,
            chat_template_sha256=hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest())

def prepare(out,n=1000):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    rows=load_jsonl(root/'data/full_llama31_8b/cached_answers.jsonl')
    selected=sorted((r for r in rows if r['split']=='val'),key=lambda r:digest(['scale-family-v1',r['question_id']]))[:n]
    write_jsonl(out/'data.jsonl',selected)
    maps=json.loads((root/'runs/scale_family_v1/mappings.json').read_text());write_json(out/'mappings.json',maps)
    cp=json.loads((root/'runs/scale_family_v1/students/coarse/both_mean_brier/checkpoint.json').read_text())
    write_json(out/'manifest.json',dict(schema='actual-emission-v1',n=n,model='Qwen/Qwen3-8B',sampler_path=cp['sampler_path'],
        checkpoint_description='coarse correctness-only both-legend student; same checkpoint on both grids, not an eleven-level-trained model',
        policy=POLICY,instruction=INSTRUCTION,formats=['direct','normal','reversed'],scales=['coarse','fine'],
        data_sha256=file_digest(out/'data.jsonl'),mappings_sha256=file_digest(out/'mappings.json'),
        primary='actual parsed emitted confidence; decoded expectations are a separate auxiliary experiment',
        invalid_policy='retain raw invalid outputs; Brier/search loss penalty 1; no numerical replacement; report valid-only discrimination with denominator',
        selection='first 1000 validation question IDs under fixed label-blind scale-family-v1 hash; no test or calibration change',
        retries='SDK transport retry max_retries=2; no semantic retry or output repair',
        adaptive_disclosure='choice of coarse supervised control informed by inspected exploratory coarse test summary'))

def run(out,workers=8):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());maps=json.loads((out/'mappings.json').read_text())
    assert file_digest(out/'data.jsonl')==m['data_sha256'] and file_digest(out/'mappings.json')==m['mappings_sha256']
    rows=load_jsonl(out/'data.jsonl');gen=Generator(m['sampler_path']);path=out/'reports.jsonl'
    existing=load_jsonl(path) if path.exists() else [];done={(r['question_id'],r['scale'],r['format']) for r in existing}
    if len(done)!=len(existing):raise ValueError('Duplicate emission records')
    tasks=[(r,scale,fmt) for r in rows for scale in m['scales'] for fmt in m['formats'] if (r['question_id'],scale,fmt) not in done]
    def one(task):
        row,scale,fmt=task;mapping=maps[scale]['normal' if fmt=='direct' else fmt]
        text=render(row['question'],row['answer'],mapping,fmt=='direct',m['instruction']);result=gen.generate(text,m['policy']);v=parse(result['raw_response'],mapping,fmt=='direct')
        return dict(row,**result,scale=scale,format=fmt,mapping=mapping,grid=sorted(mapping.values()),fixed_answer_id=digest([row['question_id'],row['answer']]),
                    correctness_label=row['correct'],emitted_value=v,valid=v is not None,emitted_brier=(v-row['correct'])**2 if v is not None else None,
                    penalized_brier=(v-row['correct'])**2 if v is not None else 1.,permitted_output_mass=None,mass_status='not measured by free generation; do not infer from validity')
    with path.open('a') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(one,tasks),1):
            stream.write(json.dumps(result)+'\n');stream.flush()
            if i%30==0:print(f"Emissions {len(done)+i}/{len(rows)*len(m['scales'])*len(m['formats'])}",flush=True)
    write_json(out/'STATUS.json',dict(complete=True,n_questions=len(rows),reports=len(rows)*len(m['scales'])*len(m['formats']),primary='actual emitted confidence'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=1000);a=p.parse_args()
    prepare(a.out,a.n) if a.action=='prepare' else run(a.out)
