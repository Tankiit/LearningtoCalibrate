"""Declared secondary percentage grammar; preserve original prompts and outputs."""
import json
from pathlib import Path
from .data import load_jsonl,write_json,write_jsonl,digest
from .emitted_reports import parse
from .emission_analysis import analyze

def rescore(source,contract_name="optional_percent_v1"):
 source=Path(source);out=source/contract_name;out.mkdir(exist_ok=True)
 declaration=Path(__file__).resolve().parents[1]/'contracts'/f'{contract_name}.json'
 if contract_name=='lenient_v1' and not declaration.exists():raise ValueError('Declare lenient_v1 before reading the cache')
 m=json.loads((source/'manifest.json').read_text());m.update(parse_contract=contract_name,source_raw_cache=str(source/'reports.jsonl'),source_raw_sha256=digest(source/'reports.jsonl'),declaration='2026-09-15, after inspecting strict-parser suffix omissions; secondary retrospective analysis',grammar='Whitespace-trimmed numeral with optional percent suffix, always interpreted in percentage units; must equal a permitted grid value. No substring extraction, clipping, rounding, scientific notation, fraction-unit inference or semantic retries.')
 if contract_name=='lenient_v1':
  m.update(declaration=json.loads(declaration.read_text()),declaration_sha256=digest(declaration),grammar=json.loads(declaration.read_text())['grammar'],units=json.loads(declaration.read_text())['units'])
  (out/'contract_declaration.json').write_bytes(declaration.read_bytes())
 write_json(out/'manifest.json',m)
 for name in ['data.jsonl','mappings.json']:(out/name).write_bytes((source/name).read_bytes())
 rr=[]
 for original in load_jsonl(source/'reports.jsonl'):
  r=dict(original);v=parse(r['raw_response'],r['mapping'],r['format']=='direct',contract_name=contract_name)
  r.update(strict_emitted_value=original['emitted_value'],strict_valid=original['valid'],emitted_value=v,valid=v is not None,parse_contract=contract_name,emitted_brier=(v-r['correct'])**2 if v is not None else None,penalized_brier=(v-r['correct'])**2 if v is not None else 1.)
  rr.append(r)
 write_jsonl(out/'reports.jsonl',rr);analyze(out)
 if (source/'mapping_controls.jsonl').exists():
  controls=[]
  for r in load_jsonl(source/'mapping_controls.jsonl'):
   v=parse(r['raw_response'],r['mapping'],r['format']=='direct',contract_name=contract_name);controls.append(dict(r,strict_valid=r['valid'],strict_correct=r['correct'],emitted_value=v,valid=v is not None,correct=v==r['known_value'],parse_contract=contract_name))
  write_jsonl(out/'mapping_controls.jsonl',controls)
 write_json(out/'STATUS.json',dict(complete=True,new_model_calls=0,records=len(rr),scope='secondary parser applied to same raw generations; '+('contracts declared before collection' if m.get('schema')=='trained-emissions-v1' else 'retrospective analysis')))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--contract',choices=['optional_percent_v1','lenient_v1'],default='optional_percent_v1');p.add_argument('sources',nargs='+');a=p.parse_args()
 for source in a.sources:rescore(source,a.contract)
