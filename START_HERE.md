# RSUQ — Start Here (playbook)

A random-set uncertainty library serving four papers; frames are the
interface, papers are thin drivers. Read STORY.md (Stage II) and
AAAI_STRATEGY.md for the framing; this file is how to RUN things.

## Install (M2)
    pip install torch transformers datasets scikit-learn scipy pytest
    python -m pytest tests/          # all identities I1-I8 + RS-NN compat, seconds

## The gate ladder (run in order; each can falsify the next)
    python run_EA_gate.py --model gpt2 --k 200      # E-A  DONE: PASS x2
    python run_EB_figure1.py --model gpt2           # E-B  fixed-W + turnover
    python run_EC_ED.py --model gpt2 --dataset halueval --span_mode first  # E-C/E-D
    #   --dataset {halueval,triviaqa}  --span_mode {first,span}  --tau 0.5

## What each layer is
    rsuq/core/frame.py       FixedFrame (Stage I/II/NeurIPS) + ContextFrame (AAAI)
    rsuq/core/sense_inventory.py  OD-1 measured-separation Stratum P, OD-2/3
    rsuq/core/beliefs.py     mass head, q, pignistics (per-position aware)
    rsuq/core/signals.py     W, H_tc, H(BetP), size-controlled W
    rsuq/rsnn_compat.py      RS-NN matrix pignistic + belief encoding (citeable);
                             Mobius fenced for Stage III only
    rsuq/align.py            offset-based span/word location (correctness fix)
    rsuq/extract.py          chunk-don't-pad, batched states, lazy tied logits
    rsuq/bags.py             MC-dropout (scoped) + ablation bags, one-pass
    rsuq/loading.py          gpt2 -> llama load path (fp16/4bit)
    rsuq/diagnostics/        orthogonality gate, redundancy screen, smoking gun
    papers/                  stage1_driver, stage2 (in stage2_package), aaai_driver

## Status (2026-06)
    E-A  PASS: confusion alignment ~50x baseline; sense routing Jaccard 0.156
    E-B  ready to run (fixed-W half; contextual-W after sense inventory)
    E-C/E-D  END-TO-END runnable: HaluEval/TriviaQA loaders + sense inventory +
             mass-head training + stratified AUROC/screen all wired
    OD-1 resolved: Stratum P by occurrence-state silhouette > tau (0.5)

## Library usage (audited, see UPGRADES.md)
    offset alignment, tied-head lazy logits, chunk-don't-pad, base_model states,
    scoped dropout, model-agnostic AutoModel + hidden_size, 4bit path.

## E-C/E-D end-to-end (the headline)
    python run_EC_ED.py --model gpt2 --dataset halueval --span_mode first
Pipeline: WikiText occurrences -> sense inventory (OD-1 measured separation,
tau=0.5) -> ContextFrame -> cache states -> train mass head (cluster CE) ->
load QA (balanced 0/1) -> score answer-span tokens (fixed-W, ctx-W, conf,
error, in_P) -> stratified AUROC + screen. Writes results_EC_ED.json and
results_sense_inventory.json. Auto-verdict 'surgical_claim': PASS iff ctx-W
beats fixed-W on Stratum P AND null on Stratum M.

Recommended substrate order: HaluEval first (balanced, gold labels, cleanest
screen), TriviaQA second (distractor-paired) as the robustness substrate.
span_mode=first is deferral-honest (per-token correctness convention);
span_mode=span is the all-answer-tokens robustness check.

## AAAI headline on RS-LLM's OWN benchmarks (run_rsllm_bench.py)
Uses Shireen's OBQA/CoQA prompts + answer-slot scoring as the baseline recipe,
but holds frame recipe (our MiniBatch K-means) and head (frozen) FIXED across
arms so the ONLY variable is fixed-vs-context membership.

    # laptop smoke test:
    python run_rsllm_bench.py --model gpt2 --dataset obqa --k 2000 --n 200
    # real (their setup):
    python run_rsllm_bench.py --model meta-llama/Llama-2-7b-hf --load_in_4bit \
        --dataset obqa --k 8000 --tau 0.5

Sense inventory is built from the BENCHMARK's own facts/stories (on-task
context), so Stratum P = answer tokens whose competitor set shifts with the
task context — cleaner than generic WikiText polysemes. OBQA leads (one
unambiguous option-token position/question); CoQA robustness (answer span).
Framing rule: claim context-on-frozen-frame > fixed-frame, NOT our-head >
their-LoRA-head (they trained more params; that's their advantage, not the
contest).

## The deferral layer (rsuq/deferral.py) — what makes it a DEFERRAL paper
The bench now emits, per stratum AND overall:
  - risk-coverage AURC for confidence / fixed-W / ctx-W (lower=better)
  - composition: conf vs conf+W_ctx vs conf+W_fixed (the headline contrast)
  - deferral_claim PASS iff conf+W_ctx beats conf AND conf+W_fixed does not
    (the positive mirror of the deferral paper's negative result: same width
    formula, only the frame differs; ctx composes, fixed is redundant).
Headline table = risk-coverage / AURC (deferral-led, per the locked intro).
AUROC detection is the supporting lemma, not the main number.
Theory backing: W ⊥ BetP (impossibility result) => the width has a STRUCTURAL
reason to be non-redundant — unlike the exogenous proxies that failed.
