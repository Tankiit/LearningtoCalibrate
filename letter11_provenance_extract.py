"""Fresh provenance-controlled forward/reversed letter11 extraction.

Writes a new namespace and never reads or overwrites legacy letter11 artifacts.
The forward arm is the reference for the paired reversed arm.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("ova-arr-letter11-provenance")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.0", "transformers==5.16.1", "accelerate==1.14.0",
                 "datasets==5.0.1", "numpy==2.4.6", "tqdm==4.70.0")
    .add_local_dir(str(ROOT / "data"), remote_path="/root/data")
    .add_local_dir("/Users/tanmoy/research/data/NLI-variation-data",
                   remote_path="/root/local_data/NLI-variation-data")
)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ova-arr-extract", create_if_missing=True)
secret = modal.Secret.from_name("huggingface")

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


def sha256_bytes(x):
    return hashlib.sha256(x).hexdigest()


@app.function(image=image, gpu="A100-40GB",
              volumes={"/root/.cache/huggingface": hf_cache, "/extract": out_vol},
              secrets=[secret], timeout=21600)
def run(model_key: str, datasets: list[str] | None = None,
        batch_size: int = 8, force: bool = False):
    import numpy as np
    import torch
    from data.registry import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    datasets = datasets or list(DATASETS)
    model_id, revision = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16,
        device_map="auto").eval()
    device = next(model.parameters()).device
    versions = {p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "accelerate", "datasets", "numpy")}
    tok_json = getattr(tokenizer, "init_kwargs", {})
    print(f"{model_key}: revision={revision}, versions={versions}")

    for dataset in datasets:
        records = load_dataset(dataset)
        for arm, values in SCHEMES.items():
            out_path = f"/extract/letter11_provenance_v1/{model_key}/{dataset}/{arm}.pt"
            if not force and os.path.exists(out_path):
                print(f"skip existing {out_path}")
                continue
            ids11, common_prefix = token_setup(tokenizer, values)
            ids11_t = torch.tensor(ids11, dtype=torch.long, device=device)
            vals = np.asarray(values, dtype=np.float32) / 100.0
            out_ids, pos_rows, neg_rows = [], [], []
            pos_mass, neg_mass = [], []
            prompt_hashes_pos, prompt_hashes_neg = [], []
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                for side, field, rows, masses in (
                    ("pos", "correct_answer", pos_rows, pos_mass),
                    ("neg", "wrong_answer", neg_rows, neg_mass),
                ):
                    prompts = [prompt_for(r.question, getattr(r, field), values)
                               for r in batch]
                    batch_hashes = [sha256_bytes(x.encode()) for x in prompts]
                    enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                    if common_prefix:
                        prefix = torch.tensor(common_prefix, device=device).expand(len(batch), -1)
                        enc["input_ids"] = torch.cat((enc["input_ids"], prefix), dim=1)
                        enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(prefix)), dim=1)
                    with torch.inference_mode():
                        logits = model(**enc).logits[:, -1, :].float()
                    selected = logits.index_select(1, ids11_t)
                    rows.extend(torch.softmax(selected, dim=1).cpu().numpy())
                    masses.extend(torch.softmax(logits, dim=1).index_select(1, ids11_t).sum(1).cpu().numpy())
                    (prompt_hashes_pos if side == "pos" else prompt_hashes_neg).extend(batch_hashes)
                out_ids.extend(str(r.example_id) for r in batch)

            pos = np.asarray(pos_rows, dtype=np.float32)
            neg = np.asarray(neg_rows, dtype=np.float32)
            mp = np.asarray(pos_mass, dtype=np.float32)
            mn = np.asarray(neg_mass, dtype=np.float32)
            if pos.shape != (len(records), 11) or neg.shape != pos.shape:
                raise RuntimeError(f"shape mismatch {pos.shape} {neg.shape}")
            meta = {
                "schema_version": 1, "artifact": "letter11_provenance_v1",
                "arm": arm, "model_key": model_key, "model_id": model_id,
                "model_revision": revision, "dataset": dataset,
                "prompt_template": BASE_PROMPT, "letters": list(LETTERS),
                "values": list(values), "candidate_ids": ids11,
                "common_prefix_ids": common_prefix, "padding_side": "left",
                "read_position": "final token after common continuation prefix",
                "distribution_semantics": "conditional_on_candidate_tokens",
                "mass_semantics": "full_vocabulary_probability_of_candidate_tokens",
                "versions": versions,
                "tokenizer_init_kwargs_sha256": sha256_bytes(json.dumps(tok_json, sort_keys=True, default=str).encode()),
                "prompt_hashes_pos_sha256": sha256_bytes(json.dumps(prompt_hashes_pos).encode()),
                "prompt_hashes_neg_sha256": sha256_bytes(json.dumps(prompt_hashes_neg).encode()),
                "n_questions": len(records),
                "legacy_artifact_used": False,
            }
            out = {
                "Vdist_pos": pos, "Vdist_neg": neg,
                "V_pos": pos @ vals, "V_neg": neg @ vals,
                "first_token_letter_mass_pos": mp,
                "first_token_letter_mass_neg": mn,
                "conf_values": np.asarray(values, dtype=np.int64),
                "example_ids": np.asarray(out_ids, dtype=object),
                "prompt_hashes_pos": np.asarray(prompt_hashes_pos, dtype=object),
                "prompt_hashes_neg": np.asarray(prompt_hashes_neg, dtype=object),
                "meta": meta,
            }
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(out, out_path)
            out_vol.commit()
            print(f"saved {out_path}; usable={( (mp>.10)&(mn>.10) ).mean():.3f}")


@app.local_entrypoint()
def main(model: str = "llama3_8b,mistral_7b,qwen2_5_7b", force: bool = False):
    for key in [x.strip() for x in model.split(",")]:
        run.remote(key, list(DATASETS), force=force)
