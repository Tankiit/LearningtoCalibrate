"""Neutral-prompt token-prior diagnostic for the format experiments.

Run with Modal before digit10-direct. This is deliberately separate from the
item-conditioned elicitation and writes JSON diagnostics to the extraction
volume, not to the digit-arm result directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("ova-arr-neutral-token-prior")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch>=2.1.0", "transformers>=4.44.0", "accelerate>=0.28.0",
                      "numpy>=1.26.0"))
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ova-arr-extract", create_if_missing=True)
secret = modal.Secret.from_name("huggingface")

MODELS = {
    "llama3_8b": ("meta-llama/Llama-3.1-8B-Instruct", "A100-40GB", "float16"),
    "mistral_7b": ("mistralai/Mistral-7B-Instruct-v0.3", "A100-40GB", "float16"),
    "qwen2_5_7b": ("Qwen/Qwen2.5-7B-Instruct", "A100-40GB", "float16"),
}

SCHEMES = {
    "letter11": {
        "levels": list(zip("ABCDEFGHJKL", range(0, 101, 10))),
        "prompt": ("Question: {q}\nAnswer: {a}\n"
                   "How confident are you that this answer is correct?\n"
                   "A=0%  B=10%  C=20%  D=30%  E=40%  F=50%  G=60%  "
                   "H=70%  J=80%  K=90%  L=100%\n"
                   "Answer with a single letter.\nConfidence:"),
    },
    "digit10-direct": {
        "levels": list(zip("0123456789", range(0, 100, 10))),
        "prompt": ("Question: {q}\nAnswer: {a}\n"
                   "How confident are you that this answer is correct?\n"
                   "0=0%  1=10%  2=20%  3=30%  4=40%  5=50%  6=60%  "
                   "7=70%  8=80%  9=90%\n"
                   "Answer with a single digit.\nConfidence:"),
    },
}


def _ids(tokenizer, levels, prompt):
    probe = prompt.format(q="[neutral question]", a="[neutral answer]")
    base = tokenizer(probe, add_special_tokens=True).input_ids
    continuations = {}
    dropped = {}
    for label, value in levels:
        full = tokenizer(probe + " " + label, add_special_tokens=True).input_ids
        if full[:len(base)] != base:
            dropped[label] = "prompt prefix changed"
        else:
            continuations[label] = (value, full[len(base):])
    if not continuations:
        return {}, dropped
    tails = [tail for _, tail in continuations.values()]
    lead_n = 0
    for i in range(min(map(len, tails))):
        if len({tail[i] for tail in tails}) != 1:
            break
        lead_n += 1
    result = {}
    for label, (value, tail) in continuations.items():
        remaining = tail[lead_n:]
        if len(remaining) != 1:
            dropped[label] = f"not single-token ({len(remaining)})"
        else:
            result[label] = (remaining[0], value)
    return result, dropped


def _run(model_key: str):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_name, gpu, dtype_name = MODELS[model_key]
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=dtype,
                                                 device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for scheme, spec in SCHEMES.items():
        prompt = spec["prompt"].format(q="[neutral question]", a="[neutral answer]")
        candidates, dropped = _ids(tok, spec["levels"], spec["prompt"])
        ids = [x[0] for x in candidates.values()]
        with torch.no_grad():
            input_ids = torch.tensor([tok(prompt, add_special_tokens=True).input_ids],
                                     dtype=torch.long, device=device)
            probs = torch.softmax(model(input_ids).logits[0, -1], dim=-1).float().cpu().numpy()
        mass = float(probs[ids].sum()) if ids else 0.0
        p = np.asarray([probs[i] for i in ids], dtype=float)
        if p.sum() > 0:
            q = p / p.sum()
            entropy = float(-(q * np.log(q)).sum())
            effective = float(np.exp(entropy))
        else:
            entropy = effective = float("nan")
        rows.append({"model": model_key, "model_id": hf_name,
                     "conf_scheme": scheme, "neutral_question": "[neutral question]",
                     "neutral_answer": "[neutral answer]", "candidate_values": [v for _, v in candidates.values()],
                     "candidate_labels": list(candidates),
                     "candidate_token_ids": ids, "dropped": dropped,
                     "candidate_mass": mass, "omitted_mass": 1.0 - mass,
                     "conditional_entropy": entropy, "effective_support": effective,
                     "tokenizer": tok.__class__.__name__})
    out_dir = "/extract/neutral_token_prior"
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir + "/" + model_key + ".json"
    with open(out, "w") as f:
        json.dump({"protocol": "neutral-token-prior-v1", "rows": rows}, f, indent=2)
    out_vol.commit()
    print(out)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": hf_cache,
              "/extract": out_vol}, secrets=[secret], timeout=3600)
def run(model_key: str):
    _run(model_key)


@app.local_entrypoint()
def main(model: str = "llama3_8b,mistral_7b,qwen2_5_7b"):
    for model_key in [x.strip() for x in model.split(",")]:
        if model_key not in MODELS:
            raise ValueError(f"unknown model: {model_key}")
        run.remote(model_key)
