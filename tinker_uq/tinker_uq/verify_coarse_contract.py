"""Reconstruct literal prompts/tokenization and export self-contained coarse records."""
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from .data import load_jsonl,write_json
from .training import prompt_text,encode_prompt,completions
from .reporting import align_to_values,digest
from .matched_analysis import verified_group

def run(out,scale='coarse',plan_filename='student_plan.json'):
 out=Path(out);maps=json.loads((out/'mappings.json').read_text())[scale];plan=json.loads((out/plan_filename).read_text());refs={r['row_id']:r for r in load_jsonl(out/'data.jsonl') if r['split']=='test'};tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',local_files_only=True);folder=out/('students/analysis_coarse' if plan_filename=='student_plan.json' and scale=='coarse' else f'students/contract_audit_{scale}');folder.mkdir(exist_ok=True);count=0
 with (folder/'verified_reporting_predictions.jsonl').open('w') as f:
  for method in ['teacher']+plan['methods'][scale]:
   root=out/'students'/scale/method;revision=plan['initial_sampler'] if method=='teacher' else json.loads((root/'checkpoint.json').read_text())['sampler_path']
   for name,mapping in maps.items():
    for rid,r in verified_group(root/f'{name}.jsonl',refs,mapping).items():
     ref=refs[rid];text=prompt_text(ref,'qa',mapping=mapping);tokens=encode_prompt(tok,ref,'qa',mapping=mapping);prov=r['prompt_provenance']
     if prov['text']!=text or prov['text_sha256']!=hashlib.sha256(text.encode()).hexdigest() or prov['token_ids_sha256']!=digest(tokens):raise ValueError('Literal prompt/token mismatch')
     if prov['chat_template_sha256']!=hashlib.sha256(tok.chat_template.encode()).hexdigest() or r['completion_token_ids']!=completions(tok,mapping):raise ValueError('Template/completion mismatch')
     record=dict(r,method=method,scale=scale,model='Qwen/Qwen3-8B',model_revision=revision,base_model_revision_status='provider base revision not independently exposed; exact adapter sampler path and tokenizer/template hashes recorded',
       fixed_answer_id=digest([ref['question_id'],ref['answer']]),question=ref['question'],answer=ref['answer'],correctness_label=ref['correct'],grid=sorted(mapping.values()),aligned_probability_vector=align_to_values(r['probability_vector'],mapping),
       full_prompt_hash=digest(tokens),student_training_exposure=name in (['normal'] if method=='normal_token_brier' else ['normal','reversed']) if method!='teacher' else None,teacher_target_exposure=name in list(maps)[:6],teacher_target_used_by_method=method in ('aligned_distill','aligned_distill_supervised'))
     f.write(json.dumps(record)+'\n');count+=1
 write_json(folder/'contract_audit.json',dict(verified_records=count,questions=len(refs),checks=['question/split/label joins','literal prompt','prompt token hash','chat template hash','completion tokenization','mapping','full probability vector','decoded expectation','allowed sequence mass'],probability_readout='conditional full letter-plus-newline completion distribution',base_revision_limit='exact adapter paths available; no independent provider backbone commit exposed'))
 print('Verified',scale,'records',count)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--scale',choices=['coarse','fine'],default='coarse');p.add_argument('--plan',default='student_plan.json');a=p.parse_args();run(a.source,a.scale,a.plan)
