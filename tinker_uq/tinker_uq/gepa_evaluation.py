"""Evaluate frozen coarse GEPA prompts on development IDs excluded from search."""
import json,concurrent.futures
from pathlib import Path
from .data import load_jsonl,write_json,write_jsonl,digest as file_digest
from .reporting import digest
from .emitted_reports import Generator,render,parse
from .emission_analysis import analyze

def run(source,search,n=128):
 source=Path(source);search=Path(search);m=json.loads((source/'manifest.json').read_text());maps=json.loads((source/'mappings.json').read_text());used={r['question_id'] for r in load_jsonl(search/'validation.jsonl')}
 rows=sorted((r for r in load_jsonl(source/'data.jsonl') if int(digest(['selection-fit-v1',r['question_id']]),16)%2==1),key=lambda r:digest(['gepa-eval-v1',r['question_id']]))[:n]
 assert not used & {r['question_id'] for r in rows};gen=Generator(m['sampler_path']);cached={r['full_prompt_hash']:r for r in load_jsonl(source/'reports.jsonl')}
 for arm in ('quality','quality_consistency'):
  instruction=json.loads((search/arm/'result.json').read_text())['instruction'];out=search/arm/'evaluation';out.mkdir(exist_ok=True)
  write_jsonl(out/'data.jsonl',rows);write_json(out/'mappings.json',maps);mm=dict(m,instruction=instruction,scales=['coarse'],n=n,data_sha256=file_digest(out/'data.jsonl'),mappings_sha256=file_digest(out/'mappings.json'),search_excluded_ids=True)
  write_json(out/'manifest.json',mm);path=out/'reports.jsonl';old=load_jsonl(path) if path.exists() else [];done={(r['question_id'],r['format']) for r in old}
  def one(task):
   row,fmt=task;mapping=maps['coarse']['normal' if fmt=='direct' else fmt];text=render(row['question'],row['answer'],mapping,fmt=='direct',instruction)
   tokens=gen.tokenizer.apply_chat_template([{'role':'user','content':text}],enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
   previous=cached.get(digest(tokens))
   if previous is not None:
    assert previous['prompt_text']==text and previous['model_revision']==gen.path
    result=dict(previous,cache_reused=True)
   else:result=dict(row,**gen.generate(text),cache_reused=False)
   v=parse(result['raw_response'],mapping,fmt=='direct');result.update(scale='coarse',format=fmt,mapping=mapping,grid=sorted(mapping.values()),emitted_value=v,valid=v is not None,correctness_label=row['correct'],fixed_answer_id=digest([row['question_id'],row['answer']]))
   return result
  with path.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
   for r in pool.map(one,[(r,fmt) for r in rows for fmt in m['formats'] if (r['question_id'],fmt) not in done]):f.write(json.dumps(r)+'\n');f.flush()
  analyze(out)
 write_json(search/'EVALUATION_COMPLETE.json',dict(n=n,split='validation, excluded from prompt search',scales=['coarse'],scope='exploratory optimizer pilot; unequal realized search calls despite common configured cap, see result.json'))
if __name__=='__main__':
 import sys
 run(sys.argv[1],sys.argv[2])
