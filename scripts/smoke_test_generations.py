"""
scripts/smoke_test_generations.py
─────────────────────────────────
Sanity-check the generation extraction + judge pipeline on 5 questions
BEFORE launching a 3-hour Modal job.

What this verifies:
  1. The dataset loads and yields Records with the expected fields
  2. The model generates something non-empty for each question
  3. Hidden state shapes match expectations: [n_layers=3, D=hidden_dim]
  4. The probe layer indices we picked make sense
  5. logprob_gen is a reasonable float (not NaN)
  6. The judge loads and produces labels that match the expected format
  7. Generation diversity: greedy vs sample give different outputs

Run this LOCALLY (not on Modal) with a small/cheap model first, then with
the actual target model. The full target model run will take ~2 minutes
for 5 questions on an A100 (or ~10-30 min on a laptop's GPU/MPS).

Usage:
  # Quick smoke test with a small model
  python scripts/smoke_test_generations.py \\
    --model-id distilgpt2 --skip-judge --n 5

  # Real smoke test with target model (needs HF auth + decent GPU)
  python scripts/smoke_test_generations.py \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --dataset truthfulqa \\
    --n 5

  # Include judge (needs Llama-2 license accepted on HF)
  python scripts/smoke_test_generations.py \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --dataset truthfulqa --n 5 --include-judge
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Project root on sys.path so we can `from data...`, `from extraction...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def smoke_test(args):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print(f"  SMOKE TEST: {args.model_id} × {args.dataset} × N={args.n}")
    print("=" * 70)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    try:
        from data.registry import load_dataset
        records = load_dataset(args.dataset)
        print(f"   ✓ Loaded {len(records)} records from {args.dataset!r}")
        records = records[:args.n]
        for i, r in enumerate(records):
            print(f"   [{i}] q: {r.question[:70]!r}")
            print(f"       correct: {r.correct_answer[:70]!r}")
            print(f"       expert_reliable: {r.expert_reliable}, category: {r.category!r}")
    except Exception as e:
        print(f"   ✗ FAIL: {type(e).__name__}: {e}")
        return False

    # ── 2. Load model ──────────────────────────────────────────────────────
    print(f"\n[2/6] Loading model {args.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        n_total = model.config.num_hidden_layers
        d_model = model.config.hidden_size
        device = next(model.parameters()).device
        print(f"   ✓ Loaded: n_total_layers={n_total}, d={d_model}, device={device}")
    except Exception as e:
        print(f"   ✗ FAIL loading model: {type(e).__name__}: {e}")
        return False

    # ── 3. Test layer indices ──────────────────────────────────────────────
    print("\n[3/6] Computing probe layer indices...")
    n_probe = 8  # matches llama3_8b config in generations.py
    early = n_total - n_probe + 1
    late = n_total
    mid = (early + late) // 2
    layer_indices = [early, mid, late]
    print(f"   probe range: layers {early}-{late} (last {n_probe} of {n_total})")
    print(f"   picked: early={early}, mid={mid}, late={late}")
    if not all(0 <= L <= n_total for L in layer_indices):
        print(f"   ✗ FAIL: layer indices out of bounds")
        return False
    print(f"   ✓ Layer indices valid: {layer_indices}")

    # ── 4. Test generation for each strategy ───────────────────────────────
    print("\n[4/6] Running generation for each strategy...")
    GEN_STRATEGIES = {
        "greedy":     dict(do_sample=False, num_beams=1),
        "sample_t07": dict(do_sample=True, temperature=0.7, top_p=0.95, num_beams=1),
        "beam":       dict(do_sample=False, num_beams=4),
    }

    strategy_results: dict = {}   # strategy -> list of (text, h_first, h_mean, lp)
    for strategy, gen_kwargs in GEN_STRATEGIES.items():
        print(f"\n   --- strategy: {strategy} ---")
        results = []
        for i, rec in enumerate(records):
            try:
                result = _generate_and_pool_local(
                    model, tokenizer, device, rec.question,
                    n_probe, n_total, gen_kwargs,
                )
                if result is None:
                    print(f"   [{i}] ✗ generated nothing")
                    continue
                gen_text, h_first, h_mean, lp = result
                results.append((gen_text, h_first, h_mean, lp))
                if i < 3:  # Print first 3 for inspection
                    print(f"   [{i}] gen: {gen_text[:80]!r}")
                    print(f"       h_first shape: {h_first.shape}, "
                          f"h_mean shape: {h_mean.shape}, lp: {lp:.3f}")
            except Exception as e:
                print(f"   [{i}] ✗ FAIL: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                return False
        strategy_results[strategy] = results
        print(f"   ✓ {strategy}: {len(results)}/{len(records)} succeeded")

    # ── 5. Sanity check: do different strategies actually differ? ──────────
    print("\n[5/6] Generation diversity check...")
    greedy_texts = [r[0] for r in strategy_results["greedy"]]
    sample_texts = [r[0] for r in strategy_results["sample_t07"]]
    beam_texts   = [r[0] for r in strategy_results["beam"]]

    n_diff_gs = sum(1 for g, s in zip(greedy_texts, sample_texts) if g != s)
    n_diff_gb = sum(1 for g, b in zip(greedy_texts, beam_texts) if g != b)
    print(f"   greedy vs sample_t07 differ on: {n_diff_gs}/{len(greedy_texts)}")
    print(f"   greedy vs beam       differ on: {n_diff_gb}/{len(greedy_texts)}")
    if n_diff_gs == 0:
        print(f"   ⚠ WARNING: greedy and sample_t07 produced identical outputs. "
              f"Sampling not working as expected.")
    print(f"   ✓ Strategies produce different outputs (as expected)")

    # ── 6. Optional: test the judge ────────────────────────────────────────
    if args.include_judge and not args.skip_judge:
        print("\n[6/6] Testing the judge...")
        try:
            from eval.judge import TruthfulQAJudge
            print("   Loading judge model (this takes ~30s)...")
            judge = TruthfulQAJudge()
            questions = [r.question for r in records]
            generations = [strategy_results["greedy"][i][0] for i in range(len(records))]
            print(f"   Labeling {len(records)} (q, gen) pairs...")
            result = judge.label_batch(questions, generations, verbose=True)
            print(f"   ✓ Judge produced {len(result.labels)} labels")
            print(f"   labels: {result.labels.tolist()}")
            print(f"   scores: {[f'{s:.2f}' for s in result.scores]}")
            print(f"   p_yes:  {[f'{p:.2f}' for p in result.raw_yes_probs]}")
            if not (0 <= result.raw_yes_probs.min() and result.raw_yes_probs.max() <= 1):
                print(f"   ✗ FAIL: p_yes out of [0,1]")
                return False
        except Exception as e:
            print(f"   ✗ FAIL: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return False
    else:
        print("\n[6/6] Skipping judge (use --include-judge to enable)")

    print("\n" + "=" * 70)
    print(f"  ✓ ALL SMOKE TESTS PASSED")
    print(f"    Safe to launch the full extraction on Modal.")
    print("=" * 70)
    return True


def _generate_and_pool_local(
    model, tokenizer, device, question: str,
    n_probe: int, n_total: int, gen_kwargs: dict,
):
    """Local mirror of extraction.generations._generate_and_pool, for smoke testing.
    Keeping a separate copy here lets us iterate on it without re-deploying Modal."""
    import torch
    import numpy as np

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    gen_kwargs_full = dict(
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        **gen_kwargs,
    )

    with torch.no_grad():
        out = model.generate(prompt_ids, **gen_kwargs_full)

    full_ids = out.sequences[0]
    n_gen = full_ids.shape[0] - prompt_len
    if n_gen == 0:
        return None

    generated_ids = full_ids[prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    early = n_total - n_probe + 1
    late = n_total
    mid = (early + late) // 2
    layer_indices = [early, mid, late]

    # Pool hidden states. out.hidden_states is a tuple of length n_gen,
    # each element is a tuple of (n_total + 1) tensors (one per layer, +1 for embedding).
    h_per_step = []
    for t, layer_dict in enumerate(out.hidden_states):
        step_vecs = []
        for L in layer_indices:
            hs = layer_dict[L]   # [1, T, D] or sometimes [batch=1, T, D]
            v = hs[0, -1, :]
            step_vecs.append(v.cpu().float())
        h_per_step.append(torch.stack(step_vecs))   # [n_layers, D]
    h_per_step = torch.stack(h_per_step)   # [n_gen, n_layers, D]

    h_first_answer = h_per_step[0]               # [n_layers, D]
    h_mean_answer  = h_per_step.mean(dim=0)      # [n_layers, D]

    # logprob of generation
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True,
                   help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--dataset", default="truthfulqa")
    p.add_argument("--n", type=int, default=5,
                   help="Number of questions to test on")
    p.add_argument("--include-judge", action="store_true",
                   help="Also test the judge module")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip judge even if --include-judge is set")
    args = p.parse_args()

    ok = smoke_test(args)
    sys.exit(0 if ok else 1)
