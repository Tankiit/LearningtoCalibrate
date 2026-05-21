"""
extraction/extract_full.py
==========================

Comprehensive extraction pipeline that caches hidden states across ALL layers
and ALL pooling strategies. This replaces the single-layer, single-pool
approach of modal_app.py with a "cache everything" strategy so that downstream
(pool, layer, probe) exploration is free after the initial GPU spend.

After this script has run for all models x datasets, all downstream work
(probe training, variant audits, layer sweeps, pool sweeps, new methods)
is FREE. This is the GPU-spend lock-in.

INTUITION on what we cache:
- ALL layers (not a subset)
- ALL pool variants (last, response_mean, response_max)
- ALL generation strategies (greedy, sample_t07, sample_t10)
- Contrastive (h+, h-) for probe training
- Generation h_gen + text + logprob + seq_entropy for evaluation
- N sampled generations per item (for semantic entropy baseline)

Output layout on the volume:
    /extract_full/{model_key}/{dataset}/
        contrastive_h.pt        -> dict(h_pos, h_neg) each (N, n_layers, n_pools, dim)
        contrastive_meta.pt     -> dict(records, y_model)
        gen_{strategy}_h.pt     -> dict(h) (N, n_layers, n_pools, dim)
        gen_{strategy}_meta.pt  -> dict(records, text, logprob, seq_entropy)
        judge_{strategy}.pt     -> dict(y_correct, scores)

Usage:
    modal run extraction/extract_full.py -- --model llama3_8b --dataset truthfulqa
    modal run extraction/extract_full.py -- --small-models
    modal run extraction/extract_full.py -- --all --dry-run
    modal run extraction/extract_full.py -- --skip-samples
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

import modal

# =============================================================================
# Modal app + volume + image
# =============================================================================

app = modal.App("ova-arr-extract-full")
volume = modal.Volume.from_name("ova-arr-extract-full", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
local_data = modal.Volume.from_name("ova-arr-local-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.28.0",
        "datasets>=2.18.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.23.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "pyyaml",
    )
    .add_local_python_source("data", "utils")
)

HF_SECRET = modal.Secret.from_name("huggingface")

# =============================================================================
# Model & pool registry (matches existing modal_app.py conventions)
# =============================================================================

MODEL_REGISTRY = {
    "llama3_8b": {
        "hf_name":    "meta-llama/Llama-3.1-8B-Instruct",
        "gpu":        "A100-40GB",
        "n_gpu":      1,
        "dtype":      "float16",
    },
    "mistral_7b": {
        "hf_name":    "mistralai/Mistral-7B-Instruct-v0.3",
        "gpu":        "A100-40GB",
        "n_gpu":      1,
        "dtype":      "float16",
    },
    "qwen2_5_7b": {
        "hf_name":    "Qwen/Qwen2.5-7B-Instruct",
        "gpu":        "A100-40GB",
        "n_gpu":      1,
        "dtype":      "float16",
    },
    "gemma2_9b": {
        "hf_name":    "google/gemma-2-9b-it",
        "gpu":        "A100-40GB",
        "n_gpu":      1,
        "dtype":      "bfloat16",
    },
    "llama3_70b": {
        "hf_name":    "meta-llama/Llama-3.1-70B-Instruct",
        "gpu":        "A100-80GB",
        "n_gpu":      2,
        "dtype":      "float16",
    },
}

POOLS = ["last", "response_mean", "response_max"]  # always in this order
STRATEGIES = ["greedy", "sample_t07", "sample_t10"]
DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "pubmedqa",
    "truthfulqa",
    "medqa",
    "triviaqa",
    "popqa",
    "bioasq",
    "multinli_disagreement",
]

SAVE_EVERY = 100
MAX_NEW_TOKENS = 128


# =============================================================================
# Helpers
# =============================================================================

def _prompt_for_record(rec, side: str) -> str:
    """
    Build the prompt for a Record.

    side="right"    -> prompt + correct_answer   (for contrastive h^+)
    side="wrong"    -> prompt + wrong_answer     (for contrastive h^-)
    side="question" -> prompt only               (for generation)
    """
    prefix = f"Question: {rec.question}\nAnswer: "
    if side == "right":
        return prefix + rec.correct_answer
    elif side == "wrong":
        return prefix + rec.wrong_answer
    else:
        return prefix


def _build_response_mask(enc, side: str, tokenizer) -> "torch.Tensor":
    """
    Build a (B, T) mask where 1 = response token, 0 = prompt token.

    For contrastive (side="right"/"wrong"), we need to mark answer tokens.
    We encode prompt-only and full (prompt+answer) separately to find the
    answer boundary. For generation, the caller handles this directly.
    """
    import torch
    B = enc["input_ids"].size(0)

    # We can't easily split inside a batch, so we return the full mask
    # and let the caller slice. For contrastive extraction, the mask is
    # built per-record in the extraction loop.
    raise NotImplementedError(
        "Response mask must be built per-record in the extraction loop. "
        "See _extract_contrastive_batch and _extract_generation_batch."
    )


def _pool_hidden_states(hidden_states: "torch.Tensor",
                        response_mask: "torch.Tensor",
                        pool: str) -> "torch.Tensor":
    """
    Pool hidden states over response tokens.

    hidden_states:  (B, T, D)
    response_mask:  (B, T) -- 1 on response tokens, 0 elsewhere

    INTUITION:
      'last'          = last response token (current default, but biased --
                        aliases with the model's own logprob)
      'response_mean' = mean over response tokens (recommended canonical pool)
      'response_max'  = element-wise max -- sometimes catches strong features

    Returns (B, D).
    """
    import torch

    if pool == "last":
        # Index of the last 1 in response_mask per batch item
        # cumsum trick: position of last 1 = (cumsum == total_sum).argmax
        token_counts = response_mask.sum(dim=1, keepdim=True)  # (B, 1)
        cumsum = response_mask.cumsum(dim=1)  # (B, T)
        # The last response token has cumsum == token_counts
        idx = (cumsum == token_counts).float().argmax(dim=1)  # (B,)
        return hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            idx,
        ]
    elif pool == "response_mean":
        mask = response_mask.unsqueeze(-1).float()  # (B, T, 1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif pool == "response_max":
        mask = response_mask.unsqueeze(-1).bool()  # (B, T, 1)
        h = hidden_states.masked_fill(~mask, float("-inf"))
        return h.max(dim=1).values
    else:
        raise ValueError(f"Unknown pool: {pool}")


def _extract_all_pools_all_layers(
    model, tokenizer, texts: list[str],
    answer_lengths: list[int], device,
) -> "torch.Tensor":
    """
    For one batch of texts, extract hidden states at ALL layers and ALL pools.

    texts:          list of full (prompt + answer) strings
    answer_lengths: number of answer tokens for each text in the batch

    Returns:
        h: (B, n_layers, n_pools, D)  in fp16
    """
    import torch

    enc = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
        # out.hidden_states: tuple of (n_layers+1) tensors, each (B, T, D)
        # Drop embedding layer (index 0)
        layer_hs = out.hidden_states[1:]

    # Stack layers: (B, n_layers, T, D)
    stacked = torch.stack(layer_hs, dim=1)

    B, L, T, D = stacked.shape

    # Build response masks from answer_lengths
    # We need to account for padding -- find where the answer starts
    response_mask = torch.zeros(B, T, dtype=torch.long, device=device)
    for i, ans_len in enumerate(answer_lengths):
        prompt_len = T - ans_len  # approximate; works when padding is right
        # More accurate: find the actual prompt length by encoding prompt-only
        # For now we use attention_mask to find real tokens
        real_tokens = enc["attention_mask"][i].sum().item()
        start = real_tokens - ans_len
        if start < 0:
            start = 0
        response_mask[i, start:real_tokens] = 1

    # Apply each pool to each layer
    h_all = torch.zeros(B, L, len(POOLS), D,
                        dtype=stacked.dtype, device=device)
    for li in range(L):
        for pi, pool in enumerate(POOLS):
            h_all[:, li, pi, :] = _pool_hidden_states(
                stacked[:, li, :, :], response_mask, pool,
            )

    return h_all.half().cpu()


# =============================================================================
# Contrastive extraction (for probe training)
# =============================================================================

@app.function(
    gpu="A100-40GB",
    volumes={
        "/extract_full": volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/local_data": local_data,
    },
    image=image,
    secrets=[HF_SECRET],
    timeout=60 * 60 * 3,
)
def extract_contrastive(model_key: str, dataset_name: str, batch_size: int = 4):
    """
    Cache (h+, h-) hidden states for probe training, across ALL layers and pools.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    cfg = MODEL_REGISTRY[model_key]
    hf_name = cfg["hf_name"]
    device = "cuda"

    out_dir = Path("/extract_full") / model_key / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "contrastive_h.pt"
    meta_file = out_dir / "contrastive_meta.pt"
    if cache_file.exists() and meta_file.exists():
        print(f"[skip] {cache_file} exists")
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=dtype, device_map="auto",
    ).eval()

    n_total_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    from data.registry import load_dataset
    records = load_dataset(dataset_name)
    print(f"[contrastive] {model_key} x {dataset_name}: {len(records)} records, "
          f"{n_total_layers} layers, d={d_model}")

    h_pos_all, h_neg_all = [], []

    for i in tqdm(range(0, len(records), batch_size),
                  desc=f"contrastive {model_key}/{dataset_name}"):
        batch = records[i : i + batch_size]

        # Build prompts for positive and negative sides
        pos_texts = [_prompt_for_record(r, "right") for r in batch]
        neg_texts = [_prompt_for_record(r, "wrong") for r in batch]

        # Compute answer lengths for response masking
        pos_ans_lens = [
            len(tokenizer.encode(r.correct_answer, add_special_tokens=False))
            for r in batch
        ]
        neg_ans_lens = [
            len(tokenizer.encode(r.wrong_answer, add_special_tokens=False))
            for r in batch
        ]

        h_pos = _extract_all_pools_all_layers(
            model, tokenizer, pos_texts, pos_ans_lens, device,
        )  # (B, L, P, D)
        h_neg = _extract_all_pools_all_layers(
            model, tokenizer, neg_texts, neg_ans_lens, device,
        )

        h_pos_all.append(h_pos)
        h_neg_all.append(h_neg)

        if (i // batch_size) % 20 == 0:
            print(f"  contrastive batch {i}/{len(records)}")

    h_pos_cat = torch.cat(h_pos_all, dim=0)  # (N, n_layers, n_pools, D)
    h_neg_cat = torch.cat(h_neg_all, dim=0)

    torch.save({"h_pos": h_pos_cat, "h_neg": h_neg_cat}, cache_file)
    torch.save({
        "records": [r.to_dict() for r in records],
        "y_model": None,  # filled by downstream: ones for h^+, zeros for h^-
        "example_ids": [r.example_id for r in records],
        "meta": {
            "model_key": model_key,
            "dataset": dataset_name,
            "n_questions": len(records),
            "n_layers": n_total_layers,
            "n_pools": len(POOLS),
            "pool_names": POOLS,
            "d": d_model,
        },
    }, meta_file)
    volume.commit()
    print(f"[done] contrastive: {model_key} x {dataset_name} "
          f"({len(records)} records, shape {tuple(h_pos_cat.shape)})")


# =============================================================================
# Generation extraction (for AURC evaluation)
# =============================================================================

@app.function(
    gpu="A100-40GB",
    volumes={
        "/extract_full": volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/local_data": local_data,
    },
    image=image,
    secrets=[HF_SECRET],
    timeout=60 * 60 * 3,
)
def extract_generations(model_key: str, dataset_name: str,
                        strategy: str, batch_size: int = 4):
    """
    For each item: generate one continuation under `strategy`, then extract
    hidden states for the generation (all layers, all pools). Also store
    the generated text, logprob, and sequence entropy.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    cfg = MODEL_REGISTRY[model_key]
    hf_name = cfg["hf_name"]

    out_dir = Path("/extract_full") / model_key / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_h    = out_dir / f"gen_{strategy}_h.pt"
    cache_meta = out_dir / f"gen_{strategy}_meta.pt"
    if cache_h.exists() and cache_meta.exists():
        print(f"[skip] {cache_h.name} exists")
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=dtype, device_map="auto",
    ).eval()

    n_total_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    from data.registry import load_dataset
    records = load_dataset(dataset_name)
    print(f"[gen-{strategy}] {model_key} x {dataset_name}: {len(records)} records")

    gen_kwargs = _gen_kwargs_for(strategy, tokenizer)

    h_all, texts, logprobs, seq_entropies = [], [], [], []

    for i in tqdm(range(0, len(records), batch_size),
                  desc=f"gen-{strategy} {model_key}/{dataset_name}"):
        batch = records[i : i + batch_size]
        prompts = [_prompt_for_record(r, "question") for r in batch]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to("cuda")
        prompt_len = enc.input_ids.size(1)

        with torch.no_grad():
            out = model.generate(
                **enc,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
            )

        gen_ids = out.sequences[:, prompt_len:]  # response only
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        lp, se = _compute_logprob_and_entropy(
            out.scores, gen_ids, tokenizer.pad_token_id,
        )

        # Re-run a full forward (prompt + generation) to get hidden states
        full_ids = out.sequences
        attn = (full_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            f = model(
                input_ids=full_ids, attention_mask=attn,
                output_hidden_states=True, return_dict=True,
            )
        layer_hs = f.hidden_states[1:]  # drop embedding layer

        # Build response mask: positions after prompt
        T_full = full_ids.size(1)
        response_mask = torch.zeros_like(full_ids)
        response_mask[:, prompt_len:] = 1
        response_mask = response_mask * attn  # zero out padding

        # Stack layers: (B, n_layers, T_full, D)
        stacked = torch.stack(layer_hs, dim=1)
        B, L, T_, D = stacked.shape

        h_batch = torch.zeros(B, L, len(POOLS), D, dtype=stacked.dtype)
        for li in range(L):
            for pi, pool in enumerate(POOLS):
                h_batch[:, li, pi, :] = _pool_hidden_states(
                    stacked[:, li, :, :], response_mask, pool,
                )

        h_all.append(h_batch.half().cpu())
        texts.extend(gen_text)
        logprobs.extend(lp.tolist())
        seq_entropies.extend(se.tolist())

    h = torch.cat(h_all, dim=0)
    torch.save({"h": h}, cache_h)
    torch.save({
        "records": [r.to_dict() for r in records],
        "example_ids": [r.example_id for r in records],
        "text":         texts,
        "logprob":      logprobs,
        "seq_entropy":  seq_entropies,
        "meta": {
            "model_key":   model_key,
            "dataset":     dataset_name,
            "strategy":    strategy,
            "n_questions": len(records),
            "n_layers":    n_total_layers,
            "n_pools":     len(POOLS),
            "pool_names":  POOLS,
            "d":           d_model,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    }, cache_meta)
    volume.commit()
    print(f"[done] gen-{strategy}: {model_key} x {dataset_name} "
          f"({len(records)} records, shape {tuple(h.shape)})")


def _gen_kwargs_for(strategy: str, tokenizer) -> dict:
    base = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
    )
    if strategy == "greedy":
        return {**base, "do_sample": False}
    if strategy == "sample_t07":
        return {**base, "do_sample": True, "temperature": 0.7, "top_p": 0.95}
    if strategy == "sample_t10":
        return {**base, "do_sample": True, "temperature": 1.0, "top_p": 1.0}
    raise ValueError(strategy)


def _compute_logprob_and_entropy(scores, gen_ids, pad_id):
    """
    scores: tuple of (B, V) logits, length T_new
    gen_ids: (B, T_new)
    Returns (logprob_mean_per_seq, mean_token_entropy_per_seq).
    """
    import torch
    import torch.nn.functional as F

    logprobs_tok, entropies_tok = [], []
    for t, step_logits in enumerate(scores):
        log_p = F.log_softmax(step_logits.float(), dim=-1)
        p     = log_p.exp()
        ent   = -(p * log_p).sum(dim=-1)             # (B,)
        sel   = log_p.gather(1, gen_ids[:, t:t+1]).squeeze(1)  # (B,)
        logprobs_tok.append(sel)
        entropies_tok.append(ent)

    lp = torch.stack(logprobs_tok, dim=1)    # (B, T_new)
    en = torch.stack(entropies_tok, dim=1)   # (B, T_new)
    mask = (gen_ids != pad_id).float()
    lp_mean = (lp * mask).sum(1) / mask.sum(1).clamp(min=1)
    en_mean = (en * mask).sum(1) / mask.sum(1).clamp(min=1)
    return lp_mean.cpu(), en_mean.cpu()


# =============================================================================
# Judge extraction (correctness labels for generations)
# =============================================================================

@app.function(
    gpu="A10G",
    volumes={
        "/extract_full": volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/local_data": local_data,
    },
    image=image,
    secrets=[HF_SECRET],
    timeout=60 * 60 * 2,
)
def extract_judge(model_key: str, dataset_name: str, strategy: str):
    """
    Run judge on generated text to get per-example correctness labels.
    Reuses the TruthfulQA judge from run_judge.py.
    """
    import torch
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path("/extract_full") / model_key / dataset_name
    meta_file = out_dir / f"gen_{strategy}_meta.pt"
    judge_file = out_dir / f"judge_{strategy}.pt"

    if not meta_file.exists():
        print(f"[skip] generation meta not found: {meta_file}")
        return
    if judge_file.exists():
        print(f"[skip] {judge_file.name} exists")
        return

    meta = torch.load(meta_file, map_location="cpu", weights_only=False)
    texts = meta["text"]
    records_raw = meta["records"]
    N = len(texts)

    # Reconstruct Record objects for questions
    from data.schema import Record
    records = [Record.from_dict(r) for r in records_raw]
    questions = [r.question for r in records]

    JUDGE_MODEL = "truthfulqa/truthfulqa-judge-deberta-v3-large"
    jtokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    jmodel = AutoModelForSequenceClassification.from_pretrained(
        JUDGE_MODEL, torch_dtype=torch.float16,
    ).eval()
    if torch.cuda.is_available():
        jmodel = jmodel.cuda()

    labels = np.zeros(N, dtype=bool)
    scores = np.zeros(N, dtype=np.float32)
    p_yes  = np.zeros(N, dtype=np.float32)

    for i in range(N):
        prompt = (
            f"Question: {questions[i]}\n"
            f"Answer: {texts[i]}\n"
            f"Is the above answer truthful?"
        )
        inputs = jtokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs = {k: v.to(jmodel.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = jmodel(**inputs).logits[0]
        score = (logits[1] - logits[0]).float().item()
        py = torch.softmax(logits.float(), dim=0)[1].item()
        labels[i] = py >= 0.5
        scores[i] = score
        p_yes[i]  = py

    n_true = int(labels.sum())
    print(f"[judge-{strategy}] {model_key}/{dataset_name}: "
          f"{n_true}/{N} truthful ({100*n_true/N:.1f}%)")

    torch.save({
        "example_ids": meta["example_ids"],
        "y_correct":  labels,
        "scores":     scores,
        "p_yes":      p_yes,
    }, judge_file)
    volume.commit()
    print(f"[done] judge-{strategy}: {model_key} x {dataset_name}")


# =============================================================================
# N-sample generation (for semantic entropy baseline)
# =============================================================================

@app.function(
    gpu="A100-40GB",
    volumes={
        "/extract_full": volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/local_data": local_data,
    },
    image=image,
    secrets=[HF_SECRET],
    timeout=60 * 60 * 4,
)
def extract_n_samples(model_key: str, dataset_name: str,
                      n_samples: int = 10, temperature: float = 1.0,
                      batch_size: int = 4):
    """
    Generates n_samples per item under temperature sampling. Stores raw text
    only (semantic entropy operates on text, not hidden states).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cfg = MODEL_REGISTRY[model_key]
    hf_name = cfg["hf_name"]

    out_dir = Path("/extract_full") / model_key / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"nsamples{n_samples}.pt"
    if cache_file.exists():
        print(f"[skip] {cache_file}")
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=dtype, device_map="auto",
    ).eval()

    from data.registry import load_dataset
    records = load_dataset(dataset_name)
    print(f"[nsamples] {model_key} x {dataset_name}: "
          f"{len(records)} records, {n_samples} samples each")

    all_samples = [[] for _ in records]  # all_samples[i] = list of n_samples strings

    for s in range(n_samples):
        torch.manual_seed(1000 + s)
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            prompts = [_prompt_for_record(r, "question") for r in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to("cuda")
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    do_sample=True, temperature=temperature, top_p=1.0,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out_ids[:, enc.input_ids.size(1):]
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for j, t in enumerate(gen_text):
                all_samples[i + j].append(t)
        print(f"  sample {s+1}/{n_samples} pass complete")

    torch.save({
        "samples":     all_samples,
        "records":     [r.to_dict() for r in records],
        "example_ids": [r.example_id for r in records],
        "meta": {
            "model_key":  model_key,
            "dataset":    dataset_name,
            "n_samples":  n_samples,
            "temperature": temperature,
        },
    }, cache_file)
    volume.commit()
    print(f"[done] n_samples: {model_key} x {dataset_name}")


# =============================================================================
# List files utility
# =============================================================================

@app.function(
    image=image,
    volumes={"/extract_full": volume},
    timeout=120,
)
def list_files():
    rows = []
    for root, dirs, files in os.walk("/extract_full"):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith(".pt"):
                continue
            path = os.path.join(root, fname)
            mb = os.path.getsize(path) / 1e6
            rows.append((path.replace("/extract_full/", ""), round(mb, 1)))
    return rows


# =============================================================================
# Orchestrator
# =============================================================================

@app.local_entrypoint()
def main(
    only_models: Optional[str] = None,
    only_datasets: Optional[str] = None,
    skip_samples: bool = False,
    skip_judge: bool = False,
    skip_contrastive: bool = False,
    skip_gen: bool = False,
    dry_run: bool = False,
    list_files_: bool = False,
):
    """
    Launch all extractions.

    --only-models    Comma-separated model keys (default: all)
    --only-datasets  Comma-separated dataset names (default: all)
    --skip-samples   Skip n-sample generation (semantic entropy)
    --skip-judge     Skip judge labeling
    --skip-contrastive  Skip contrastive extraction
    --skip-gen       Skip generation extraction
    --dry-run        Show plan without running
    --list-files     Show volume contents
    """
    # -- List files -----------------------------------------------------------
    if list_files_:
        rows = list_files.remote()
        if not rows:
            print("Volume is empty.")
            return
        print(f"\n{'File':<65}  {'MB':>7}")
        print("  " + "-" * 74)
        for rel, mb in rows:
            print(f"  {rel:<63}  {mb:>7.1f}")
        print(f"\n  Total: {sum(mb for _, mb in rows):.0f} MB  "
              f"({len(rows)} files)")
        return

    models = list(MODEL_REGISTRY.keys()) if not only_models \
        else [m.strip() for m in only_models.split(",")]
    datasets = DATASETS if not only_datasets \
        else [d.strip() for d in only_datasets.split(",")]

    bad_models = [m for m in models if m not in MODEL_REGISTRY]
    if bad_models:
        print(f"Unknown models: {bad_models}. Valid: {list(MODEL_REGISTRY.keys())}")
        return
    bad_ds = [d for d in datasets if d not in DATASETS]
    if bad_ds:
        print(f"Unknown datasets: {bad_ds}. Valid: {DATASETS}")
        return

    # -- Dry run --------------------------------------------------------------
    if dry_run:
        RATES = {"A100-40GB": 2.10, "A100-80GB": 2.50, "A10G": 0.70}
        print(f"\nDRY RUN | models ({len(models)}): {models}")
        print(f"         | datasets ({len(datasets)}): {datasets}")
        print(f"\nJobs to launch:")
        total = 0.0
        for m in models:
            cfg = MODEL_REGISTRY[m]
            for d in datasets:
                if not skip_contrastive:
                    hrs = 2.0 * (len(datasets) / 5)
                    cost = RATES[cfg["gpu"]] * cfg["n_gpu"] * hrs / len(datasets)
                    total += cost
                    print(f"  contrastive   {m}/{d}  "
                          f"{cfg['n_gpu']}x{cfg['gpu']}  ~${cost:.0f}")
                if not skip_gen:
                    for strat in STRATEGIES:
                        hrs = 1.5 * (len(datasets) / 5)
                        cost = RATES[cfg["gpu"]] * cfg["n_gpu"] * hrs / len(datasets)
                        total += cost
                        print(f"  gen-{strat:<12} {m}/{d}  "
                              f"{cfg['n_gpu']}x{cfg['gpu']}  ~${cost:.0f}")
                if not skip_judge:
                    cost = RATES["A10G"] * 0.5
                    total += cost
                    print(f"  judge          {m}/{d}  A10G  ~${cost:.0f}")
                if not skip_samples:
                    hrs = 4.0 * (len(datasets) / 5)
                    cost = RATES[cfg["gpu"]] * cfg["n_gpu"] * hrs / len(datasets)
                    total += cost
                    print(f"  nsamples10     {m}/{d}  "
                          f"{cfg['n_gpu']}x{cfg['gpu']}  ~${cost:.0f}")
        print(f"\n  TOTAL ESTIMATE: ~${total:.0f}")
        return

    # -- Launch all jobs ------------------------------------------------------
    futures = []
    for m in models:
        for d in datasets:
            if not skip_contrastive:
                futures.append(extract_contrastive.spawn(m, d))
            if not skip_gen:
                for strat in STRATEGIES:
                    futures.append(extract_generations.spawn(m, d, strat))
            if not skip_judge:
                for strat in STRATEGIES:
                    futures.append(extract_judge.spawn(m, d, strat))
            if not skip_samples:
                futures.append(extract_n_samples.spawn(m, d))

    print(f"\nLaunched {len(futures)} jobs across {len(models)} models, "
          f"{len(datasets)} datasets")

    done = 0
    failed = 0
    for f in futures:
        try:
            f.get()
            done += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\n{'='*58}")
    print(f"  Done: {done} succeeded, {failed} failed")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"{'='*58}")
