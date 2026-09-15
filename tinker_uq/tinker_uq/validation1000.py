"""Canonical 1,000-validation-question cache with explicit budget and checkpoint lineage."""
import argparse,hashlib,json,shutil
from pathlib import Path
from .data import read_rows,load_jsonl,write_json,write_jsonl,digest
from .reporting import digest as object_digest
from .scale_family import prepare as prepare_scale
from .matched_experiment import score_file,codebook_scores
from .training import prompt_text,encode_prompt


def prepare(out):
    out=Path(out);prepare_scale(out,n=1000)
    m=json.loads((out/'manifest.json').read_text());m.update(schema='validation1000-v1',
        checkpoint_status='binary-correctness-trained QA reporter; not an eleven-level-trained reporter',
        model_revision=m['sampler_path'],tokenizer_revision='saved checkpoint tokenizer; chat-template/token hashes saved at extraction',
        pilot_size_note='proposed 1000 validation questions, not a power calculation',
        question_selection='smallest SHA256 JSON([scale-family-v1,question_id]) within original validation split; independent of labels',
        test_partition='original full test partition unchanged and not evaluated in this validation study',
        validation_analysis='500 fixed ID-hash fit questions and 500 disjoint evaluation questions; all hyperparameters fixed',
        deployment_cost='6 report prompts per scale/question for full average or family; 1 for normal; 2 for normal/reversed average',
        backend_cost='score each letter+newline sequence: 3 requests per coarse report, 11 per fine report; report-prompt parity is not backend/FLOP parity',
        training_target_calls=0,student_training='not part of this validation cache; gated on family decision value')
    write_json(out/'manifest.json',m);write_json(out/'settings.json',dict(prompt_template=prompt_text({'question':'{question}','answer':'{fixed_answer}'},'qa',mapping=json.loads((out/'mappings.json').read_text())['fine']['normal']),
        normalization='softmax of complete letter+newline sequence log-likelihoods over permitted outputs',
        primary_probability='expectation of actual legend-aligned numerical confidence distribution',
        rendering=dict(add_generation_prompt=True,enable_thinking=False),selection_by_label=False,
        mappings='mappings.json',checkpoint=m['sampler_path'],confidence_interval_claim=False))


def extract(out):
    import tinker
    out=Path(out);root=Path(__file__).resolve().parents[1];m=json.loads((out/'manifest.json').read_text());maps=json.loads((out/'mappings.json').read_text())
    assert digest(out/'data.jsonl')==m['data_sha256'] and digest(out/'mappings.json')==m['mappings_sha256']
    rows=[r for r in read_rows(out/'data.jsonl') if r['split']=='val'];assert len(rows)==1000
    service=tinker.ServiceClient(timeout=60,max_retries=2);sampler=service.create_sampling_client(model_path=m['sampler_path']);tok=sampler.get_tokenizer()
    template_hash=hashlib.sha256(tok.chat_template.encode()).hexdigest()
    old=root/'runs/scale_family_v1';old_m=json.loads((old/'manifest.json').read_text());assert old_m['sampler_path']==m['sampler_path']
    new_calls=0;reused_calls=0
    write_json(out/'runtime.json',dict(model_revision=m['sampler_path'],chat_template_sha256=template_hash,source_code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    for scale,family in maps.items():
        folder=out/scale;folder.mkdir(exist_ok=True)
        for name in m['initial_mapping_names'][scale]:
            mapping=family[name];target=folder/f'{name}.jsonl'
            if target.exists():continue
            reused={}
            old_file=old/scale/f'{name}.jsonl'
            if old_file.exists():
                for r in load_jsonl(old_file):reused[r['row_id']]=r
            for row in rows:
                if row['row_id'] not in reused:continue
                r=reused[row['row_id']]
                assert r['question_id']==row['question_id'] and r['correct']==row['correct'] and r['split']=='val' and r['mapping']==mapping
                text=prompt_text(row,'qa',mapping=mapping);assert r['prompt_provenance']['text']==text
                assert r['prompt_provenance']['chat_template_sha256']==template_hash
                assert r['prompt_provenance']['token_ids_sha256']==object_digest(encode_prompt(tok,row,'qa',mapping=mapping))
            missing=[r for r in rows if r['row_id'] not in reused]
            piece=folder/f'new_{name}';piece.mkdir(exist_ok=True)
            score_file(sampler,missing,tok,'existing_binary_qa',piece,tinker,{name:mapping})
            combined=dict(reused);combined.update({r['row_id']:r for r in load_jsonl(piece/f'{name}.jsonl')})
            export=[]
            for row in rows:
                r=combined[row['row_id']];r.update(scale=scale,fixed_answer_id=row['row_id'],correctness_label=row['correct'],model_revision=m['sampler_path'],
                    grid=sorted(mapping.values()),full_prompt_hash=r['prompt_provenance']['token_ids_sha256'],
                    aligned_probability_vector=[r['probability_vector'][j] for j in sorted(range(len(r['letters'])),key=lambda j:mapping[r['letters'][j]])],
                    permitted_output_mass=r['allowed_alphabet_mass'],cache_origin='128_question_pilot' if row['row_id'] in reused else '1000_question_extension')
                export.append(r)
            write_jsonl(target,export);new_calls+=len(missing);reused_calls+=len(rows)-len(missing)
            print('completed',scale,name,'new',len(missing),'reused',len(rows)-len(missing),flush=True)
        if not (folder/'codebook.json').exists():shutil.copyfile(old/scale/'codebook.json',folder/'codebook.json')
    all_rows=[]
    for scale in maps:
        for name in m['initial_mapping_names'][scale]:all_rows+=load_jsonl(out/scale/f'{name}.jsonl')
    write_jsonl(out/'validation_cache.jsonl',all_rows)
    write_json(out/'query_accounting.json',dict(validation_questions=1000,report_prompts_per_question_per_scale=6,
        total_report_prompts_in_cache=12000,reused_report_prompts=sum(r['cache_origin']=='128_question_pilot' for r in all_rows),
        new_report_prompts=sum(r['cache_origin']=='1000_question_extension' for r in all_rows),
        logical_backend_sequence_requests_in_cache=1000*6*(3+11),new_backend_sequence_requests=sum(len(r['grid']) for r in all_rows if r['cache_origin']=='1000_question_extension'),
        codebook_report_prompts_reused=18+66,training_target_report_prompts=0,
        note='All ensemble/family selection approaches share these caches. A one-report normal baseline and two-report average have different deployment costs and are labeled separately.'))
    from .family_analysis import analyze
    analyze(out)
    write_json(out/'VALIDATION_COMPLETE.json',dict(complete=True,questions=1000,records=len(all_rows)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','extract','analyze']);p.add_argument('--out',required=True);a=p.parse_args()
    if a.action=='prepare':prepare(a.out)
    elif a.action=='extract':extract(a.out)
    else:
        from .family_analysis import analyze
        analyze(Path(a.out))
if __name__=='__main__':main()
