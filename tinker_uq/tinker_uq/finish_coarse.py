"""Finish existing coarse checkpoint evaluations only; never start training."""
import json
from pathlib import Path
from .data import read_rows
from .matched_experiment import score_file,codebook_scores

def main():
 import tinker
 out=Path('runs/scale_family_v1');m=json.loads((out/'manifest.json').read_text());family=json.loads((out/'mappings.json').read_text())['coarse']
 rows=[r for r in read_rows(out/'data.jsonl') if r['split']=='test'];service=tinker.ServiceClient(timeout=60,max_retries=2)
 for method in ('both_mean_brier','aligned_distill','aligned_distill_supervised'):
  fd=out/'students/coarse'/method;cp=json.loads((fd/'checkpoint.json').read_text());sampler=service.create_sampling_client(model_path=cp['sampler_path']);tok=sampler.get_tokenizer()
  score_file(sampler,rows,tok,method,fd,tinker,family);codebook_scores(sampler,tok,family,fd/'codebook.json',tinker)
 print('Existing coarse collection complete',flush=True)
if __name__=='__main__':main()
