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
V3_SCHEMA = 3
OUT_PREFIX = "letter11_permuted_v3"
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
    candidate_ids = [tail[0] for tail in tails]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("confidence letters collide at the read position")
    common_prefix = continuations[0][2][:lead]
    return candidate_ids, list(VALUES), common_prefix


def _run(model_key: str, datasets: list[str], batch_size: int = 8,
         force: bool = False):
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
    candidate_ids, values, common_prefix = _candidate_ids(tokenizer)
    candidate_ids_t = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    values_np = np.asarray(values, dtype=np.float32) / 100.0
    print(f"{model_key}: candidate IDs={candidate_ids}, values={values}")

    for dataset in datasets:
        records = load_dataset(dataset)
        v2_candidates = (
            f"/extract/letter11_permuted_v2/{model_key}/{dataset}/letter11_permuted.pt",
            f"/extract/letter11_permuted/{model_key}/{dataset}/letter11_permuted.pt",
        )
        v2_path = next((path for path in v2_candidates if os.path.exists(path)), None)
        if v2_path is None:
            raise FileNotFoundError(
                "v2 regression artifact missing; checked: " + ", ".join(v2_candidates)
            )
        v2 = torch.load(v2_path, map_location="cpu", weights_only=False)
        v2_by_id = {str(item_id): i for i, item_id in enumerate(v2["example_ids"])}
        ids = []
        pos_rows, neg_rows = [], []
        mass_rows = {"pos": [], "neg": []}
        out_path = f"/extract/{OUT_PREFIX}/{model_key}/{dataset}/letter11_permuted.pt"
        if not force and os.path.exists(out_path):
            saved = torch.load(out_path, map_location="cpu", weights_only=False)
            if saved.get("meta", {}).get("schema_version") == V3_SCHEMA:
                print(f"{model_key}/{dataset}: schema {V3_SCHEMA} already present — skipping")
                continue
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            for side, field, rows in (("pos", "correct_answer", pos_rows),
                                      ("neg", "wrong_answer", neg_rows)):
                prompts = [PROMPT.format(q=r.question, a=getattr(r, field))
                           for r in batch]
                enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                # Condition on any shared continuation tokens before reading
                # the token that actually distinguishes the confidence values.
                if common_prefix:
                    prefix = torch.tensor(common_prefix, device=device).expand(len(batch), -1)
                    enc["input_ids"] = torch.cat((enc["input_ids"], prefix), dim=1)
                    enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(prefix)), dim=1)
                with torch.no_grad():
                    logits = model(**enc).logits
                # With left padding the final real token is at column -1.
                # sum(mask)-1 is valid only for right-padded batches.
                row_logits = logits[:, -1, :].float()
                selected = row_logits.index_select(1, candidate_ids_t)
                # Preserve v2's exact candidate-conditional distribution while
                # recording the alphabet mass before candidate renormalisation.
                dist = torch.softmax(selected, dim=1).float().cpu().numpy()
                full_probs = torch.softmax(row_logits, dim=1)
                p_letters = full_probs.index_select(1, candidate_ids_t)
                mass = p_letters.sum(dim=1)
                mass_rows[side].extend(mass.cpu().numpy())
                rows.extend(dist)
                del enc, logits, selected, full_probs, p_letters
            ids.extend(r.example_id for r in batch)
            if (start // batch_size) % 25 == 0:
                print(f"{model_key}/{dataset}: {min(start + batch_size, len(records))}/{len(records)}")
        pos = np.asarray(pos_rows, dtype=np.float32)
        neg = np.asarray(neg_rows, dtype=np.float32)
        if pos.shape != (len(records), 11) or neg.shape != pos.shape:
            raise RuntimeError(f"unexpected shapes: {pos.shape}, {neg.shape}")

        if len(v2_by_id) != len(v2["example_ids"]):
            raise RuntimeError("v2 regression artifact has duplicate example IDs")
        missing = [str(item_id) for item_id in ids if str(item_id) not in v2_by_id]
        if missing:
            raise RuntimeError(
                f"v3/v2 regression ID mismatch: {len(missing)} v3 items missing from v2; "
                "provenance, not result"
            )
        max_abs_dev = 0.0
        for j, item_id in enumerate(ids):
            k = v2_by_id[str(item_id)]
            max_abs_dev = max(
                max_abs_dev,
                float(np.abs(pos[j] - np.asarray(v2["Vdist_pos"][k])).max()),
                float(np.abs(neg[j] - np.asarray(v2["Vdist_neg"][k])).max()),
            )
        if max_abs_dev > 1e-3:
            raise RuntimeError(
                f"v3/v2 Vdist mismatch {max_abs_dev:.2e} on {model_key}/{dataset}: "
                "provenance, not result"
            )

        mass_pos = np.asarray(mass_rows["pos"], dtype=np.float32)
        mass_neg = np.asarray(mass_rows["neg"], dtype=np.float32)
        usable = (mass_pos > 0.10) & (mass_neg > 0.10)
        out = {
            "Vdist_pos": pos, "Vdist_neg": neg,
            "V_pos": pos @ values_np, "V_neg": neg @ values_np,
            "first_token_letter_mass_pos": mass_pos,
            "first_token_letter_mass_neg": mass_neg,
            "conf_values": np.asarray(values, dtype=np.int64),
            "example_ids": np.asarray(ids, dtype=object),
            "meta": {
                "model_id": hf_name, "model_key": model_key, "dataset": dataset,
                "schema_version": V3_SCHEMA,
                "conf_scheme": "letter11-permuted", "permutation": "reversal",
                "conf_prompt": PROMPT, "has_s_conf": False,
                "candidate_ids": candidate_ids,
                "extractor_version": "leftpad-readout-v3",
                "read_position": "final token after common continuation prefix",
                "common_prefix_ids": common_prefix,
                "padding_side": "left",
                "mass_semantics": "full_vocabulary_probability_of_candidate_tokens",
                "distribution_semantics": "conditional_on_candidate_tokens",
                "mass_gate": 0.10,
                "usable_fraction": float(usable.mean()),
                "usable_fraction_threshold": 0.80,
                "v2_regression_max_abs_dev": float(max_abs_dev),
                "v2_items_compared": int(len(ids)),
                "legend": "value->letter map reversed, letter order fixed",
            },
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        out_vol.commit()
        print(f"saved {out_path}")


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": hf_cache,
              "/extract": out_vol}, secrets=[secret], timeout=21600)
def run(model_key: str, datasets: list[str], force: bool = False):
    _run(model_key, datasets, force=force)


@app.local_entrypoint()
def main(model: str = "llama3_8b,mistral_7b,qwen2_5_7b", force: bool = False):
    for model_key in [x.strip() for x in model.split(",")]:
        if model_key not in MODELS:
            raise ValueError(model_key)
        run.remote(model_key, list(DATASETS), force=force)
