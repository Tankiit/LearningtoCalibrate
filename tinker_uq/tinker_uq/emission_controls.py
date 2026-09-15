"""Generation-based known-value codebook controls for both reporting grids."""
import json,concurrent.futures
from pathlib import Path
from .emitted_reports import Generator,contract,parse,POLICY
from .data import load_jsonl,write_json

def run(out,workers=1):
 out=Path(out);m=json.loads((out/'manifest.json').read_text());maps=json.loads((out/'mappings.json').read_text());g=Generator(m['sampler_path']);path=out/'mapping_controls.jsonl'
 done=load_jsonl(path) if path.exists() else [];keys={(r['scale'],r['format'],r['known_value']) for r in done}
 jobs=[]
 for scale in m['scales']:
  for fmt in m['formats']:
   mapping=maps[scale]['normal' if fmt=='direct' else fmt]
   jobs.extend((scale,fmt,mapping,value) for value in sorted(mapping.values()) if (scale,fmt,value) not in keys)
 def generate(job):
  scale,fmt,mapping,value=job
  prompt=f'The supplied probability is {value*100:g}%.\nEncode only this known probability using the reporting contract.\n'+contract(mapping,fmt=='direct')
  r=g.generate(prompt,POLICY);v=parse(r['raw_response'],mapping,fmt=='direct');r.update(scale=scale,format=fmt,mapping=mapping,known_value=value,emitted_value=v,valid=v is not None,correct=v==value)
  return r
 with path.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  for r in pool.map(generate,jobs):
   f.write(json.dumps(r)+'\n');f.flush()
 rows=load_jsonl(path);table=[]
 for scale in m['scales']:
  for fmt in m['formats']:
   rs=[r for r in rows if (r['scale'],r['format'])==(scale,fmt)]
   table.append(dict(scale=scale,format=fmt,n=len(rs),accuracy=sum(r['correct'] for r in rs)/len(rs),invalid=sum(not r['valid'] for r in rs)/len(rs)))
 write_json(out/'mapping_control_summary.json',table)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--workers',type=int,default=1);a=p.parse_args();run(a.source,a.workers)
