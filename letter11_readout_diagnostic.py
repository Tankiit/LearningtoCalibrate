"""Controlled legacy/current readout comparison for letter11-permuted.

The two variants use the same model, tokenizer, prompts, batches, and item IDs.
Only the readout path differs:
  legacy: read after the prompt as stored by the old runner;
  current: append the tokenizer-derived common prefix before reading.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("ova-arr-letter11-readout-diagnostic")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1.0", "transformers>=4.44.0", "accelerate>=0.28.0",
                 "datasets>=2.18.0", "numpy>=1.26.0", "scipy>=1.11.0",
                 "tqdm>=4.66.0")
    .add_local_dir(str(ROOT / "data"), remote_path="/root/data")
    .add_local_dir("/Users/tanmoy/research/data/NLI-variation-data",
                   remote_path="/root/local_data/NLI-variation-data")
)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ova-arr-extract", create_if_missing=True)
secret = modal.Secret.from_name("huggingface")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LETTERS = tuple("ABCDEFGHJKL")
VALUES = tuple(range(100, -1, -10))
LEGEND = "  ".join(f"{x}={v}%" for x, v in zip(LETTERS, VALUES))
PROMPT = (
    "Question: {q}\nAnswer: {a}\n"
    "How confident are you that this answer is correct?\n"
    f"{LEGEND}\nAnswer with a single letter.\nConfidence:"
)


def candidate_ids(tokenizer):
    probe = PROMPT.format(q="q", a="a")
    base = tokenizer(probe, add_special_tokens=True).input_ids
    tails = []
    for label in LETTERS:
        full = tokenizer(probe + " " + label, add_special_tokens=True).input_ids
        if full[:len(base)] != base:
            raise RuntimeError("prompt prefix changed")
        tails.append(full[len(base):])
    lead = 0
    while all(len(t) > lead for t in tails):
        if len({t[lead] for t in tails}) != 1:
            break
        lead += 1
    stripped = [t[lead:] for t in tails]
    if any(len(t) != 1 for t in stripped):
        raise RuntimeError(f"non-single-token letters: {stripped}")
    return [t[0] for t in stripped], tails[0][:lead]


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


@app.function(image=image, gpu="A100-40GB",
              volumes={"/root/.cache/huggingface": hf_cache, "/extract": out_vol},
              secrets=[secret], timeout=3600)
def run(n_items: int = 64, batch_size: int = 8):
    import numpy as np
    import torch
    from data.registry import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto").eval()
    device = next(model.parameters()).device
    ids11, common_prefix = candidate_ids(tokenizer)
    ids11_t = torch.tensor(ids11, dtype=torch.long, device=device)
    records = load_dataset("truthfulqa")[:n_items]

    legacy_rows, current_rows, row_ids = [], [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prompts = [PROMPT.format(q=r.question, a=r.correct_answer) for r in batch]
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            old_logits = model(**enc).logits[:, -1, :].float()
            if common_prefix:
                prefix = torch.tensor(common_prefix, device=device).expand(len(batch), -1)
                enc["input_ids"] = torch.cat((enc["input_ids"], prefix), dim=1)
                enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(prefix)), dim=1)
            new_logits = model(**enc).logits[:, -1, :].float()
        legacy_rows.extend(torch.softmax(old_logits.index_select(1, ids11_t), dim=1).cpu().numpy())
        current_rows.extend(torch.softmax(new_logits.index_select(1, ids11_t), dim=1).cpu().numpy())
        row_ids.extend(str(r.example_id) for r in batch)

    legacy_rows = np.asarray(legacy_rows, dtype=np.float32)
    current_rows = np.asarray(current_rows, dtype=np.float32)
    path = "/extract/letter11_permuted/llama3_8b/truthfulqa/letter11_permuted.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved_idx = {str(x): i for i, x in enumerate(saved["example_ids"])}
    keep = [i for i, x in enumerate(row_ids) if x in saved_idx]
    ref = np.asarray([saved["Vdist_pos"][saved_idx[row_ids[i]]] for i in keep], dtype=np.float32)
    old = legacy_rows[keep]
    new = current_rows[keep]
    old_dev = np.abs(old - ref).max(axis=1)
    new_dev = np.abs(new - ref).max(axis=1)
    old_v = old @ (np.asarray(VALUES, dtype=np.float32) / 100)
    new_v = new @ (np.asarray(VALUES, dtype=np.float32) / 100)
    ref_v = ref @ (np.asarray(saved["conf_values"], dtype=np.float32) / 100)
    print({
        "items": len(keep), "common_prefix_ids": list(map(int, common_prefix)),
        "candidate_ids": ids11,
        "legacy_max_dev": float(old_dev.max()),
        "legacy_median_dev": float(np.median(old_dev)),
        "legacy_frac_gt_05": float((old_dev > .5).mean()),
        "current_max_dev": float(new_dev.max()),
        "current_median_dev": float(np.median(new_dev)),
        "current_frac_gt_05": float((new_dev > .5).mean()),
        "rho_legacy_reference": spearman(old_v, ref_v),
        "rho_current_reference": spearman(new_v, ref_v),
    })


@app.local_entrypoint()
def main(n_items: int = 64, batch_size: int = 8):
    run.remote(n_items=n_items, batch_size=batch_size)
