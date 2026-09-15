"""ICLR mapping control on the original pinned letter11 model/readout cells."""
import hashlib,json,os
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version="3.11")
 .pip_install("torch==2.14.0","transformers==5.16.1","accelerate==1.14.0","datasets==5.0.1","numpy==2.4.6","tqdm==4.70.0")
 .add_local_dir(str(ROOT/"data"),remote_path="/root/data")
 .add_local_dir("/Users/tanmoy/research/data/NLI-variation-data",remote_path="/root/local_data/NLI-variation-data"))
hf_cache=modal.Volume.from_name("hf-model-cache",create_if_missing=True)
out_vol=modal.Volume.from_name("ova-arr-extract",create_if_missing=True)
secret=modal.Secret.from_name("huggingface")
MODELS = {
    "llama3_8b": ("meta-llama/Llama-3.1-8B-Instruct",
                  "0e9e39f249a16976918f6564b8830bc894c89659"),
    "mistral_7b": ("mistralai/Mistral-7B-Instruct-v0.3",
                   "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    "qwen2_5_7b": ("Qwen/Qwen2.5-7B-Instruct",
                   "a09a35458c702b33eeacc393d103063234e8bc28"),
}

DATASETS = ("truthfulqa", "pavlick_nli")

LETTERS = tuple("ABCDEFGHJKL")

SCHEMES = {
    "forward": tuple(range(0, 101, 10)),
    "reversed": tuple(range(100, -1, -10)),
}

BASE_PROMPT = (
    "Question: {q}\nAnswer: {a}\n"
    "How confident are you that this answer is correct?\n"
    "{legend}\nAnswer with a single letter.\nConfidence:"
)

def prompt_for(q, a, values):
    legend = "  ".join(f"{x}={v}%" for x, v in zip(LETTERS, values))
    return BASE_PROMPT.format(q=q, a=a, legend=legend)

def token_setup(tokenizer, values):
    probe = prompt_for("q", "a", values)
    base = tokenizer(probe, add_special_tokens=True).input_ids
    tails = []
    for label in LETTERS:
        full = tokenizer(probe + " " + label, add_special_tokens=True).input_ids
        if full[:len(base)] != base:
            raise RuntimeError("prompt prefix changed while finding letter IDs")
        tails.append(full[len(base):])
    lead = 0
    while all(len(t) > lead for t in tails):
        if len({t[lead] for t in tails}) != 1:
            break
        lead += 1
    stripped = [t[lead:] for t in tails]
    if any(len(t) != 1 for t in stripped):
        raise RuntimeError(f"letters are not single-token: {stripped}")
    return [t[0] for t in stripped], list(tails[0][:lead])

app=modal.App('ova-arr-letter11-mapping-control')
CONTROL='The confidence value is supplied as {value}%. Do not estimate correctness or change this value. Encode the supplied value using the legend.\n{legend}\nAnswer with a single letter.\nConfidence:'
def text_for(value,values,question=None,answer=None):
 prefix='' if question is None else f'Question: {question}\nAnswer: {answer}\n'
 return prefix+CONTROL.format(value=value,legend='  '.join(f'{k}={v}%' for k,v in zip(LETTERS,values)))

def sha(text):return hashlib.sha256(text.encode()).hexdigest()

@app.function(image=image,gpu='A100-40GB',volumes={'/root/.cache/huggingface':hf_cache,'/extract':out_vol},secrets=[secret],timeout=7200)
def collect(model_key:str,n_questions:int=32):
 import numpy as np
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from data.registry import load_dataset
 path=f'/extract/letter11_mapping_control_v1/{model_key}.json'
 if os.path.exists(path):return json.load(open(path))
 model_id,revision=MODELS[model_key];tok=AutoTokenizer.from_pretrained(model_id,revision=revision);tok.pad_token=tok.eos_token;tok.padding_side='left'
 model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16,device_map='auto').eval();device=next(model.parameters()).device
 tasks=[]
 for value in range(0,101,10):tasks.append(dict(kind='standalone',dataset=None,question_id=f'known-{value}',candidate_side=None,value=value,question=None,answer=None))
 for dataset in DATASETS:
  records=sorted(load_dataset(dataset),key=lambda r:sha('letter11-codebook-v1:'+dataset+':'+str(r.example_id)))[:n_questions]
  for j,r in enumerate(records):
   for side,field in [('pos','correct_answer'),('neg','wrong_answer')]:tasks.append(dict(kind='contextual',dataset=dataset,question_id=str(r.example_id),candidate_side=side,value=(j%11)*10,question=r.question,answer=getattr(r,field)))
 results=[]
 for name,values in SCHEMES.items():
  ids,prefix=token_setup(tok,values);it=torch.tensor(ids,device=device);vv=np.asarray(values)/100.
  for start in range(0,len(tasks),8):
   batch=tasks[start:start+8];texts=[text_for(t['value'],values,t['question'],t['answer']) for t in batch];enc=tok(texts,padding=True,return_tensors='pt').to(device)
   if prefix:
    pp=torch.tensor(prefix,device=device).expand(len(batch),-1);enc['input_ids']=torch.cat([enc['input_ids'],pp],1);enc['attention_mask']=torch.cat([enc['attention_mask'],torch.ones_like(pp)],1)
   with torch.inference_mode():
    logits=model(**enc).logits[:,-1,:].float();pr=torch.softmax(logits.index_select(1,it),-1).cpu().numpy();mass=torch.softmax(logits,-1).index_select(1,it).sum(-1).cpu().tolist()
    generated=model.generate(**enc,do_sample=False,max_new_tokens=8,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   for j,t in enumerate(batch):
    output=generated[j,enc['input_ids'].shape[1]:].tolist();raw=tok.decode(output,skip_special_tokens=True);expected=LETTERS[list(values).index(t['value'])];argmax=LETTERS[int(pr[j].argmax())]
    results.append(dict(t,legend=name,mapping=dict(zip(LETTERS,[v/100 for v in values])),prompt=texts[j],prompt_sha256=sha(texts[j]),prompt_token_ids=enc['input_ids'][j][enc['attention_mask'][j].bool()].tolist(),candidate_token_ids=ids,common_prefix_ids=prefix,
      probability_vector=pr[j].tolist(),decoded_mean=float(pr[j]@vv),allowed_alphabet_mass=mass[j],expected_symbol=expected,conditional_argmax=argmax,conditional_correct=argmax==expected,
      raw_response=raw,generated_token_ids=output,emitted_valid=raw.strip() in LETTERS,emitted_correct=raw.strip()==expected))
   print(model_key,name,min(start+8,len(tasks)),'/',len(tasks),flush=True)
 result=dict(model_key=model_key,model_id=model_id,revision=revision,control_template=CONTROL,chat_template_applied=False,readout='original provenance-v1 final token after tokenizer-derived common continuation prefix',n_context_questions_per_dataset=n_questions,known_values_assignment='standalone exhaustive; contextual ID-hash order cyclic 0..100, identical for both candidates and legends',rows=results)
 os.makedirs(os.path.dirname(path),exist_ok=True)
 with open(path,'w') as f:json.dump(result,f)
 out_vol.commit();return result

@app.local_entrypoint()
def main():
 out=Path('cached_results/letter11_mapping_control_v1');out.mkdir(parents=True,exist_ok=True)
 (out/'DECLARATION.json').write_text(json.dumps(dict(models=MODELS,datasets=DATASETS,context_questions_per_dataset=32,standalone_values=list(range(0,101,10)),legends=SCHEMES,task='encode supplied confidence, not judge candidate correctness',purpose='ICLR original letter11 reversal mapping control; independent of Qwen3 fine-grid training study',scope='paired forward/reversed controls, standalone and original-dataset context; no correctness guarantee inferred from successful encoding'),indent=2)+'\n')
 for result in collect.map(list(MODELS)):
  (out/f"{result['model_key']}.json").write_text(json.dumps(result)+'\n')
 print('Original-cell mapping controls saved',flush=True)
