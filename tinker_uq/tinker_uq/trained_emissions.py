"""Prospective strict/lenient emission readouts from matched trained reporters."""
import json
from pathlib import Path
from .data import load_jsonl,write_json,write_jsonl,digest
from .emitted_reports import run as extract,INSTRUCTION,POLICY
from .second_contract import rescore
from .emission_analysis import analyze

def run(source,scales=None):
 source=Path(source);plan=json.loads((source/'student_plan_conftuner.json').read_text());maps=json.loads((source/'mappings.json').read_text());rows=[r for r in load_jsonl(source/'data.jsonl') if r['split']=='test'];root=Path(__file__).resolve().parents[1]
 for scale,methods in plan['methods'].items():
  if scales is not None and scale not in scales:continue
  for method in methods:
   cp=json.loads((source/'students'/scale/method/'checkpoint.json').read_text());out=source/'trained_emissions'/scale/method;out.mkdir(parents=True,exist_ok=True)
   if not (out/'manifest.json').exists():
    write_jsonl(out/'data.jsonl',rows);write_json(out/'mappings.json',{scale:maps[scale]})
    write_json(out/'manifest.json',dict(schema='trained-emissions-v1',n=len(rows),model=plan['model'],sampler_path=cp['sampler_path'],checkpoint_description=f'{scale} {method}',
      policy=POLICY,instruction=INSTRUCTION,formats=['direct']+list(maps[scale]),scales=[scale],data_sha256=digest(out/'data.jsonl'),mappings_sha256=digest(out/'mappings.json'),
      parse_contract='strict_v1',secondary_contract='lenient_v1',lenient_declaration_sha256=digest(root/'contracts/lenient_v1.json'),
      split_scope='preserved 128-question exploratory test subset; no objective/hyperparameter selection on test',
      exposure='normal-only token Brier trains normal; all other students train normal/reversed. All six coarse legends used by teacher; fine last four beyond teacher.',
      budget='normal/reversed panel uses one or two reports; six-legend families compared to same six-report averages; all-legend totals differ by grid'))
   extract(out)
   # Individual formats and the two-legends panel; full-family analysis is separate.
   analyze(out)
   rescore(out,'lenient_v1')
 name='TRAINED_EMISSIONS_COMPLETE.json' if scales is None else 'TRAINED_EMISSIONS_'+'_'.join(scales).upper()+'_COMPLETE.json'
 write_json(source/name,dict(complete=True,scales=list(plan['methods']) if scales is None else scales,questions=len(rows),contracts=['strict_v1','lenient_v1']))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--scales',nargs='+',choices=['coarse','fine']);a=p.parse_args();run(a.source,a.scales)
