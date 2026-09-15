"""Fixed-checkpoint coarse/fine reporting-family comparison, before student fitting."""
import argparse,itertools,json,hashlib
from pathlib import Path
from .data import read_rows,load_jsonl,write_json,write_jsonl,digest as file_digest,audit
from .reporting import freeze_mappings,digest
from .matched_experiment import score_file,codebook_scores


def prepare(out,n=128):
    root=Path(__file__).resolve().parents[1]; out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=root/'data/full_llama31_8b/cached_answers.jsonl'; data=read_rows(source)
    selected=[]
    for split,size in [('train',128),('val',n),('cal',32),('test',128)]:
        selected+=sorted([r for r in data if r['split']==split],key=lambda r:digest(['scale-family-v1',r['question_id']]))[:size]
    write_jsonl(out/'data.jsonl',selected)
    full=root/'runs/full_llama31_8b_qwen3_8b'; cps=load_jsonl(full/'checkpoints.jsonl')
    sampler=next(r['sampler_path'] for r in cps if r['arm']=='qa' and 'sampler_path' in r)
    state=next(r['state_path'] for r in cps if r['arm']=='qa' and 'state_path' in r)
    coarse={'normal':dict(zip('ABC',[0.,.5,1.])), 'reversed':dict(zip('ABC',[1.,.5,0.]))}
    for values in itertools.permutations([0.,.5,1.]):
        if tuple(values) in [tuple(m.values()) for m in coarse.values()]:continue
        coarse[f'heldout_{len(coarse)-2:02d}']=dict(zip('ABC',values))
    fine=freeze_mappings(count=8)
    maps={'coarse':coarse,'fine':fine};write_json(out/'mappings.json',maps)
    write_json(out/'manifest.json',dict(schema='scale-family-v1',model='Qwen/Qwen3-8B',sampler_path=sampler,state_path=state,
        checkpoint='existing full-run QA reporter; identical checkpoint for both scales',data_sha256=file_digest(out/'data.jsonl'),
        source_sha256=file_digest(source),mappings_sha256=file_digest(out/'mappings.json'),audit=audit(selected),
        initial_mapping_names={k:list(v)[:6] for k,v in maps.items()},
        query_budget='six reports per validation question per scale; coarse exhaustive, fine approximate',
        additional_legends='prefix two reports versus all six; fine extra four reserved for post-student evaluation',
        controls='known probability encoding for every level and every initial mapping',
        validation_analysis='fixed 50:50 ID-hash fitting/evaluation split for matched supervised selection baselines',
        thresholds=[.5,.75,.9],retentions=[.2,.5,.8,.9],seed=0,
        family='convex hull of value-aligned report distributions; extrema evaluated at listed vertices',
        interval='min/max decoded expectations over evaluated legends; sensitivity interval, not probability confidence interval'))
    print(f'Frozen scale comparison: {out}',flush=True)


def run(out):
    import tinker
    out=Path(out);m=json.loads((out/'manifest.json').read_text());maps=json.loads((out/'mappings.json').read_text())
    assert file_digest(out/'data.jsonl')==m['data_sha256'] and file_digest(out/'mappings.json')==m['mappings_sha256']
    rows=[r for r in read_rows(out/'data.jsonl') if r['split']=='val']
    service=tinker.ServiceClient(timeout=60,max_retries=2)
    sampler=service.create_sampling_client(model_path=m['sampler_path']);tok=sampler.get_tokenizer()
    write_json(out/'runtime.json',dict(chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        source_code_sha256={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')}))
    for scale,family in maps.items():
        folder=out/scale;folder.mkdir(exist_ok=True);initial={k:family[k] for k in m['initial_mapping_names'][scale]}
        score_file(sampler,rows,tok,'existing_qa',folder,tinker,initial)
        codebook_scores(sampler,tok,initial,folder/'codebook.json',tinker)
    from .family_analysis import analyze
    analyze(out)
    write_json(out/'VALIDATION_COMPLETE.json',dict(complete=True,questions=len(rows),reports_per_scale=6))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--out',required=True);a=p.parse_args()
    if a.action=='prepare':prepare(a.out)
    elif a.action=='run':run(a.out)
    else:
        from .family_analysis import analyze
        analyze(Path(a.out))
if __name__=='__main__':main()
