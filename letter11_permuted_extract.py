"""Modal extraction for the frozen letter11-permuted preregistration.

This writes a separate artifact namespace and never modifies canonical caches.
The reversed legend is fixed in source, as required by the preregistration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("ova-arr-letter11-permuted")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1.0", "transformers>=4.44.0", "accelerate>=0.28.0",
                 "datasets>=2.18.0", "numpy>=1.26.0", "tqdm>=4.66.0")
    .add_local_dir(str(ROOT / "data"), remote_path="/root/data")
    .add_local_dir("/Users/tanmoy/research/data/NLI-variation-data",
                   remote_path="/root/local_data/NLI-variation-data")
)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ova-arr-extract", create_if_missing=True)
secret = modal.Secret.from_name("huggingface")

MODELS = {
    "llama3_8b": ("meta-llama/Llama-3.1-8B-Instruct", "float16"),
    "mistral_7b": ("mistralai/Mistral-7B-Instruct-v0.3", "float16"),
    "qwen2_5_7b": ("Qwen/Qwen2.5-7B-Instruct", "float16"),
}
DATASETS = ("truthfulqa", "pavlick_nli")
LETTERS = tuple("ABCDEFGHJKL")
VALUES = tuple(range(100, -1, -10))
LEGEND = "  ".join(f"{label}={value}%" for label, value in zip(LETTERS, VALUES))
PROMPT = (
    "Question: {q}\nAnswer: {a}\n"
    "How confident are you that this answer is correct?\n"
    f"{LEGEND}\nAnswer with a single letter.\nConfidence:"
)


def _candidate_ids(tokenizer):
    probe = PROMPT.format(q="q", a="a")
    base = tokenizer(probe, add_special_tokens=True).input_ids
    continuations = []
    for label, value in zip(LETTERS, VALUES):
        full = tokenizer(probe + " " + label, add_special_tokens=True).input_ids
        if full[:len(base)] != base:
            raise RuntimeError(f"prompt prefix changed for {label}")
        continuations.append((label, value, full[len(base):]))
    lead = 0
    while all(len(tail) > lead for _, _, tail in continuations):
        if len({tail[lead] for _, _, tail in continuations}) != 1:
            break
        lead += 1
    tails = [tail[lead:] for _, _, tail in continuations]
    if any(len(tail) != 1 for tail in tails):
        raise RuntimeError(f"permuted letter arm is not single-token: {tails}")
    return [tail[0] for tail in tails], list(VALUES), [int(x) for x in base]


def _run(model_key: str, datasets: list[str], batch_size: int = 8):
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data.registry import load_dataset

    hf_name, dtype_name = MODELS[model_key]
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=dtype, device_map="auto").eval()
    device = next(model.parameters()).device
    candidate_ids, values, _ = _candidate_ids(tokenizer)
    candidate_ids_t = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    values_np = np.asarray(values, dtype=np.float32) / 100.0
    print(f"{model_key}: candidate IDs={candidate_ids}, values={values}")

    for dataset in datasets:
        records = load_dataset(dataset)
        ids = []
        pos_rows, neg_rows = [], []
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            for side, field, rows in (("pos", "correct_answer", pos_rows),
                                      ("neg", "wrong_answer", neg_rows)):
                prompts = [PROMPT.format(q=r.question, a=getattr(r, field))
                           for r in batch]
                enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                last = enc["attention_mask"].sum(1).long() - 1
                row_logits = logits[torch.arange(len(batch), device=device), last]
                selected = row_logits.index_select(1, candidate_ids_t)
                dist = torch.softmax(selected, dim=1).float().cpu().numpy()
                rows.extend(dist)
                del enc, logits, selected
            ids.extend(r.example_id for r in batch)
            if (start // batch_size) % 25 == 0:
                print(f"{model_key}/{dataset}: {min(start + batch_size, len(records))}/{len(records)}")
        pos = np.asarray(pos_rows, dtype=np.float32)
        neg = np.asarray(neg_rows, dtype=np.float32)
        if pos.shape != (len(records), 11) or neg.shape != pos.shape:
            raise RuntimeError(f"unexpected shapes: {pos.shape}, {neg.shape}")
        out = {
            "Vdist_pos": pos, "Vdist_neg": neg,
            "V_pos": pos @ values_np, "V_neg": neg @ values_np,
            "Vmass_pos": np.ones(len(records), dtype=np.float32),
            "Vmass_neg": np.ones(len(records), dtype=np.float32),
            "conf_values": np.asarray(values, dtype=np.int64),
            "example_ids": np.asarray(ids, dtype=object),
            "meta": {
                "model_id": hf_name, "model_key": model_key, "dataset": dataset,
                "conf_scheme": "letter11-permuted", "permutation": "reversal",
                "conf_prompt": PROMPT, "has_s_conf": False,
                "candidate_ids": candidate_ids,
            },
        }
        import torch
        out_path = f"/extract/letter11_permuted/{model_key}/{dataset}/letter11_permuted.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        out_vol.commit()
        print(f"saved {out_path}")


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": hf_cache,
              "/extract": out_vol}, secrets=[secret], timeout=21600)
def run(model_key: str, datasets: list[str]):
    _run(model_key, datasets)


@app.local_entrypoint()
def main(model: str = "llama3_8b,mistral_7b,qwen2_5_7b"):
    for model_key in [x.strip() for x in model.split(",")]:
        if model_key not in MODELS:
            raise ValueError(model_key)
        run.remote(model_key, list(DATASETS))
