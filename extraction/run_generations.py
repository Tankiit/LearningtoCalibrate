"""
extraction/generations.py
─────────────────────────
Generate from an LLM unconditioned (no answer prepended) and extract hidden
states at chosen pool positions × layers.

This is the second extraction pass (the first one extracts contrastive
h^+/h^- for probe training). Generations are evaluated in step 4 instead of
the contrastive pairs.

Output schema per (model, dataset, gen_strategy):
    /extract/{model_key}/{dataset}/gen_{strategy}.pt =
    {
      'generated_text'       : list[str], len N
      'h_first_answer'       : np.ndarray [N, n_layers, D]
      'h_mean_answer'        : np.ndarray [N, n_layers, D]
      'logprob_gen'          : np.ndarray [N]              per-token mean log p
      'example_ids'          : np.ndarray [N] str
      'meta': {
        'model_id', 'model_key', 'dataset',
        'gen_strategy':   'greedy' | 'sample_t0.7' | 'beam',
        'n_questions':    N,
        'pool_positions': ['first_answer', 'mean_answer'],
        'probe_layer_indices': [24, 27, 30],   # actual layer indices used
        'max_new_tokens': 64,
      }
    }

GENERATION STRATEGIES:
  greedy:        do_sample=False, num_beams=1   (deterministic)
  sample_t0.7:   do_sample=True, temperature=0.7, top_p=0.95
  beam:          do_sample=False, num_beams=4

The chosen 3 layers from probe range are computed as
    early = n_total - n_probe + 1
    late  = n_total
    mid   = (early + late) // 2
For llama3_8b (32 layers, n_probe=8): [25, 28, 31].
"""
from __future__ import annotations
import os
import time

import modal


# ── Modal scaffolding (uses same volume + image as modal_app.py) ─────────────
APP = modal.App("ova-arr-generations")

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0", "transformers>=4.44.0", "accelerate>=0.28.0",
        "datasets>=2.18.0", "tqdm>=4.66.0", "numpy>=1.26.0",
        "huggingface_hub>=0.23.0",
    )
    .add_local_python_source("data", "utils")
)

hf_cache   = modal.Volume.from_name("hf-model-cache",  create_if_missing=True)
output_vol = modal.Volume.from_name("ova-arr-extract", create_if_missing=True)


# ── Configuration (matches MODEL_CONFIGS from modal_app.py) ──────────────────
MODEL_CONFIGS = {
    "llama3_8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu":     "A100-40GB", "n_gpu": 1, "dtype": "float16",
        "n_probe_layers": 8,
    },
    "mistral_7b": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "gpu":     "A100-40GB", "n_gpu": 1, "dtype": "float16",
        "n_probe_layers": 8,
    },
    "gemma2_9b": {
        "hf_name": "google/gemma-2-9b-it",
        "gpu":     "A100-40GB", "n_gpu": 1, "dtype": "bfloat16",
        "n_probe_layers": 10,
    },
    "llama3_70b": {
        "hf_name": "meta-llama/Llama-3.1-70B-Instruct",
        "gpu":     "A100-80GB", "n_gpu": 2, "dtype": "float16",
        "n_probe_layers": 12,
    },
}

ALL_DATASETS = ["truthfulqa", "halueval_qa", "triviaqa", "popqa", "bioasq"]
SAVE_EVERY = 100

MAX_NEW_TOKENS = 64
GEN_STRATEGIES = {
    "greedy":     dict(do_sample=False, num_beams=1),
    "sample_t07": dict(do_sample=True,  temperature=0.7, top_p=0.95, num_beams=1),
    "beam":       dict(do_sample=False, num_beams=4),
}


def _probe_layer_indices(n_total: int, n_probe: int) -> list[int]:
    """Pick 3 layers from the probe range: early, mid, late."""
    early = n_total - n_probe + 1   # first layer in probe range (1-indexed for clarity)
    late  = n_total
    mid   = (early + late) // 2
    return [early, mid, late]


def _generate_and_pool(
    model, tokenizer, device, question: str,
    n_probe: int, n_total: int, strategy: str,
):
    """One forward pass with generation. Returns:
      - generated_text: str
      - h_first_answer: [n_layers_returned, D]   first token of generation
      - h_mean_answer:  [n_layers_returned, D]   mean over generation tokens
      - logprob_gen:    float                    mean log p over generated tokens

    Three layers per pool position (early/mid/late from probe range).
    """
    import torch

    prompt = f"Question: {question}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids       = enc.input_ids.to(device)
    attention_mask   = enc.attention_mask.to(device)
    prompt_len = prompt_ids.shape[1]

    # Filter sampling-only kwargs out of the greedy/beam strategies so that
    # transformers does not warn about ignored flags.
    strat_kwargs = dict(GEN_STRATEGIES[strategy])
    if not strat_kwargs.get("do_sample", False):
        strat_kwargs.pop("temperature", None)
        strat_kwargs.pop("top_p", None)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        attention_mask=attention_mask,
        **strat_kwargs,
    )

    with torch.no_grad():
        out = model.generate(prompt_ids, **gen_kwargs)

    # out.sequences:      [1, prompt_len + n_gen]
    # out.hidden_states:  tuple of length n_gen, each is tuple of n_total+1 tensors
    full_ids = out.sequences[0]
    n_gen = full_ids.shape[0] - prompt_len
    if n_gen == 0:
        return None  # generated nothing

    generated_ids = full_ids[prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Pick 3 layer indices from the probe range
    layer_indices = _probe_layer_indices(n_total, n_probe)
    # NOTE: out.hidden_states[t] is a tuple of length n_total+1 (incl. embedding layer).
    # out.hidden_states[t][L] has shape [1, T_so_far, D] for L > 0.
    # We pool over the GENERATION tokens, which sit at positions prompt_len..prompt_len+n_gen-1.

    # Stack hidden states for each generation step at our target layers
    # Shape after stacking: [n_gen, n_layers, D]
    h_first = []  # first generation token only -> just out.hidden_states[0]
    h_per_step = []

    for t, layer_dict in enumerate(out.hidden_states):
        # layer_dict[L] has shape [1, T, D] where T is the current sequence length.
        # The hidden state for the t-th generated token is at position
        # T-1 of layer_dict[L]. (For t=0, T = prompt_len + 1; the new token's
        # rep sits at position prompt_len.)
        step_vecs = []
        for L in layer_indices:
            hs = layer_dict[L]    # [1, T, D]
            v = hs[0, -1, :]      # the just-generated token's rep at layer L
            step_vecs.append(v.cpu().float())
        h_per_step.append(torch.stack(step_vecs))   # [n_layers, D]

    h_per_step = torch.stack(h_per_step)   # [n_gen, n_layers, D]

    h_first_answer = h_per_step[0]                  # [n_layers, D]
    h_mean_answer  = h_per_step.mean(dim=0)         # [n_layers, D]

    # Generation log-probability (per-token mean)
    # out.scores is a tuple of length n_gen with [1, vocab_size] logits at each step
    if hasattr(out, "scores") and out.scores is not None and len(out.scores) > 0:
        lp_steps = []
        for t, logits in enumerate(out.scores):
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            tok_id = generated_ids[t].item()
            lp_steps.append(log_probs[tok_id].item())
        logprob_gen = float(sum(lp_steps) / len(lp_steps))
    else:
        logprob_gen = float("nan")

    return generated_text, h_first_answer.numpy(), h_mean_answer.numpy(), logprob_gen


# ── Shared generation loop (called inside each model-specific Modal function) ─

def _run_generations(model_key: str, datasets: list,
                     strategies: list[str] | None = None):
    """Generate from the model for each question and extract hidden states."""
    from data.registry import load_dataset
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    cfg     = MODEL_CONFIGS[model_key]
    n_probe = cfg["n_probe_layers"]
    strats  = strategies or list(GEN_STRATEGIES.keys())

    print(f"\n{'='*64}")
    print(f"  {model_key}  |  {cfg['hf_name']}")
    print(f"  GPU: {cfg['n_gpu']}× {cfg['gpu']}  |  strategies: {strats}")
    print(f"  Datasets: {datasets}")
    print(f"{'='*64}\n")

    # Load model once — amortised across all datasets and strategies
    t0        = time.time()
    dtype     = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_name"], torch_dtype=dtype, device_map="auto")
    model.eval()
    n_total = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    device  = next(model.parameters()).device
    print(f"  Loaded in {time.time()-t0:.0f}s | "
          f"{n_total} layers | d={d_model} | {device}")

    layer_indices = _probe_layer_indices(n_total, n_probe)
    print(f"  Probe layers: {layer_indices}  "
          f"(from last {n_probe} of {n_total})\n")

    for ds_name in datasets:
        records = load_dataset(ds_name)
        if not records:
            print(f"  ✗ {ds_name}: no records — skipping")
            continue

        for strategy in strats:
            out_dir = f"/extract/{model_key}/{ds_name}"
            out_path = f"{out_dir}/gen_{strategy}.pt"
            os.makedirs(out_dir, exist_ok=True)

            # Skip completed strategies
            if os.path.exists(out_path):
                saved = torch.load(out_path, map_location="cpu",
                                   weights_only=False)
                n = saved.get("meta", {}).get("n_questions", "?")
                print(f"  ✓ {ds_name}/{strategy}: already done "
                      f"({n} questions) — skipping")
                del saved
                continue

            print(f"\n── {model_key}/{ds_name}/{strategy}")

            buf = {
                "generated_text": [],
                "h_first_answer": [],
                "h_mean_answer":  [],
                "logprob_gen":    [],
                "example_ids":    [],
            }
            skipped = 0

            for i, rec in enumerate(tqdm(records,
                                         desc=f"{model_key}/{ds_name}/{strategy}")):
                result = _generate_and_pool(
                    model, tokenizer, device,
                    rec.question, n_probe, n_total, strategy,
                )
                if result is None:
                    skipped += 1
                    continue

                gen_text, h_first, h_mean, lp = result
                buf["generated_text"].append(gen_text)
                buf["h_first_answer"].append(h_first)
                buf["h_mean_answer"].append(h_mean)
                buf["logprob_gen"].append(lp)
                buf["example_ids"].append(rec.example_id)

                processed = i + 1

                # Checkpoint every SAVE_EVERY questions
                if processed % SAVE_EVERY == 0:
                    torch.save(buf, out_path)
                    output_vol.commit()
                    tqdm.write(f"  Checkpoint: {len(buf['example_ids'])} "
                               f"questions → {out_path}")

            N = len(buf["example_ids"])
            print(f"  Processed: {N}  Skipped: {skipped}")
            if N == 0:
                continue

            # Final save
            save_dict = {
                "generated_text": buf["generated_text"],
                "h_first_answer": np.stack(buf["h_first_answer"]),
                "h_mean_answer":  np.stack(buf["h_mean_answer"]),
                "logprob_gen":    np.asarray(buf["logprob_gen"],
                                             dtype=np.float32),
                "example_ids":    np.array(buf["example_ids"], dtype=object),
                "meta": {
                    "model_id":           cfg["hf_name"],
                    "model_key":          model_key,
                    "dataset":            ds_name,
                    "gen_strategy":       strategy,
                    "n_questions":        N,
                    "pool_positions":     ["first_answer", "mean_answer"],
                    "probe_layer_indices": layer_indices,
                    "max_new_tokens":     MAX_NEW_TOKENS,
                },
            }
            torch.save(save_dict, out_path)
            output_vol.commit()
            print(f"  ✓ Saved → {out_path}")
            print(f"    h_first_answer shape: "
                  f"{save_dict['h_first_answer'].shape}")
            print(f"    h_mean_answer shape:  "
                  f"{save_dict['h_mean_answer'].shape}")

    print(f"\n✓  All generation tasks complete for {model_key}")


# ── Modal functions at global scope (Modal requirement: GPU spec hardcoded) ───

_VOLUMES = {"/root/.cache/huggingface": hf_cache, "/extract": output_vol}
_SECRETS = [modal.Secret.from_name("huggingface")]


@APP.function(image=IMAGE, gpu="A100-40GB", volumes=_VOLUMES,
              timeout=28800, memory=65536, secrets=_SECRETS)
def run_llama3_8b(datasets: list = None,
                  strategies: list = None):
    _run_generations("llama3_8b", datasets or ALL_DATASETS, strategies)


@APP.function(image=IMAGE, gpu="A100-40GB", volumes=_VOLUMES,
              timeout=28800, memory=65536, secrets=_SECRETS)
def run_mistral_7b(datasets: list = None,
                   strategies: list = None):
    _run_generations("mistral_7b", datasets or ALL_DATASETS, strategies)


@APP.function(image=IMAGE, gpu="A100-40GB", volumes=_VOLUMES,
              timeout=28800, memory=65536, secrets=_SECRETS)
def run_gemma2_9b(datasets: list = None,
                  strategies: list = None):
    _run_generations("gemma2_9b", datasets or ALL_DATASETS, strategies)


@APP.function(image=IMAGE, gpu="A100-80GB:2", volumes=_VOLUMES,
              timeout=28800, memory=131072, secrets=_SECRETS)
def run_llama3_70b(datasets: list = None,
                   strategies: list = None):
    _run_generations("llama3_70b", datasets or ALL_DATASETS, strategies)


# Map model key → its Modal function
_FN_MAP = {
    "llama3_8b":  run_llama3_8b,
    "mistral_7b": run_mistral_7b,
    "gemma2_9b":  run_gemma2_9b,
    "llama3_70b": run_llama3_70b,
}


# ── Entrypoint ────────────────────────────────────────────────────────────────

@APP.local_entrypoint()
def main(
    model:        str  = None,
    dataset:      str  = None,
    datasets:     str  = None,
    strategies:   str  = None,
    all:          bool = False,
    small_models: bool = False,
    dry_run:      bool = False,
    gate_70b:     bool = True,
):
    """
    Run unconditioned generation + hidden-state extraction.

    --model llama3_8b               single model
    --model llama3_8b,mistral_7b    two models sequentially
    --small-models                  llama3_8b → mistral_7b → gemma2_9b
    --all                           all 4 models, gate before 70B
    --dataset truthfulqa            single dataset (shorthand)
    --datasets truthfulqa,popqa     dataset subset
    --strategies greedy,sample_t07,beam  generation strategies
    --dry-run                       show run order and cost estimate
    --no-gate-70b                   skip confirmation before 70B run
    """
    # Resolve datasets
    ds_str = datasets or dataset
    ds_list = ([d.strip() for d in ds_str.split(",")]
               if ds_str else ALL_DATASETS)

    bad_ds = [d for d in ds_list if d not in ALL_DATASETS]
    if bad_ds:
        print(f"Unknown datasets: {bad_ds}. Valid: {ALL_DATASETS}")
        return

    # Resolve strategies
    strat_list = ([s.strip() for s in strategies.split(",")]
                  if strategies else list(GEN_STRATEGIES.keys()))
    bad_st = [s for s in strat_list if s not in GEN_STRATEGIES]
    if bad_st:
        print(f"Unknown strategies: {bad_st}. "
              f"Valid: {list(GEN_STRATEGIES.keys())}")
        return

    # Determine models
    if all:
        models_to_run = list(MODEL_CONFIGS.keys())
    elif small_models:
        models_to_run = ["llama3_8b", "mistral_7b", "gemma2_9b"]
    elif model:
        models_to_run = [m.strip() for m in model.split(",")]
        bad = [m for m in models_to_run if m not in MODEL_CONFIGS]
        if bad:
            print(f"Unknown models: {bad}. "
                  f"Valid: {list(MODEL_CONFIGS.keys())}")
            return
    else:
        print("\nOptions:")
        print("  modal run extraction/run_generations.py "
              "--model llama3_8b --dataset truthfulqa")
        print("  modal run extraction/run_generations.py "
              "--model llama3_8b --strategies greedy")
        print("  modal run extraction/run_generations.py "
              "--small-models --dry-run")
        print("  modal run extraction/run_generations.py --all")
        return

    # Dry run
    if dry_run:
        RATES = {"A100-40GB": 2.10, "A100-80GB": 2.50}
        n_ds  = len(ds_list)
        n_st  = len(strat_list)
        print(f"\nDRY RUN | datasets ({n_ds}): {ds_list}")
        print(f"         | strategies ({n_st}): {strat_list}")
        print(f"Run order (sequential):")
        total = 0.0
        for i, m in enumerate(models_to_run):
            cfg  = MODEL_CONFIGS[m]
            rate = RATES[cfg["gpu"]] * cfg["n_gpu"]
            hrs  = (4.0 if "70b" in m else 2.0) * (n_ds / 5) * (n_st / 3)
            cost = rate * hrs
            total += cost
            print(f"\n  [{i+1}/{len(models_to_run)}] {m}")
            print(f"    {cfg['n_gpu']}× {cfg['gpu']}")
            print(f"    Est: {hrs:.1f}h × ${rate:.2f}/h = ~${cost:.0f}")
        print(f"\n  TOTAL: ~${total:.0f}")
        return

    # Sequential execution
    small  = [m for m in models_to_run if m != "llama3_70b"]
    run70b = "llama3_70b" in models_to_run
    done   = []

    for m in small:
        print(f"\n{'─'*58}")
        print(f"  [{len(done)+1}/{len(models_to_run)}]  {m}")
        print(f"  Datasets: {ds_list}  |  Strategies: {strat_list}")
        print(f"{'─'*58}")
        _FN_MAP[m].remote(datasets=ds_list, strategies=strat_list)
        done.append(m)
        print(f"  ✓ {m} complete  ({len(done)}/{len(models_to_run)})")

    if run70b:
        if gate_70b and small:
            print(f"\n{'='*58}")
            print(f"  Small models done. Verify results before "
                  f"llama3_70b (~$40).")
            print(f"{'='*58}")
            print(f"\n  Then re-run with:")
            print(f"    modal run extraction/run_generations.py "
                  f"--model llama3_70b --no-gate-70b "
                  f"--datasets {','.join(ds_list)} "
                  f"--strategies {','.join(strat_list)}")
        else:
            print(f"\n{'─'*58}")
            print(f"  [{len(done)+1}/{len(models_to_run)}]  llama3_70b")
            print(f"  Datasets: {ds_list}  |  Strategies: {strat_list}")
            print(f"{'─'*58}")
            _FN_MAP["llama3_70b"].remote(datasets=ds_list,
                                         strategies=strat_list)
            done.append("llama3_70b")
            print(f"  ✓ llama3_70b complete  "
                  f"({len(done)}/{len(models_to_run)})")

    print(f"\n{'='*58}")
    print(f"  ✓  Done — {len(done)} model(s): {done}")
    print(f"  Datasets: {ds_list}  |  Strategies: {strat_list}")
    print(f"{'='*58}")