# Transformers-Library Upgrades (audit applied)

1. chunk-don't-pad preprocessing (rsuq/extract.py: chunk_texts) — uniform
   blocks, no masks, full batching (run_clm idiom)
2. batched state collection via model.base_model (no 13-layer tuple unless
   last4_mean pooling requested) — rsuq/extract.py: collect_states
3. logits cache eliminated: lazy z = lm_head(h) via get_output_embeddings()
   (~5GB -> ~150MB for 50k positions) — rsuq/extract.py: logits_from_states
4. MC-dropout bag in ONE forward via batch-repeat (independent elementwise
   masks); embed-noise and token-perturb ablations batched likewise
5. portability: FixedFrame.from_model uses get_input_embeddings();
   no model-class-specific attributes anywhere; AutoModelForCausalLM-ready
6. inference_mode over no_grad; optional dtype for bf16 on A100

Deliberately NOT adopted: HF Trainer (18k-param head over cached states),
generate/KV-cache (teacher-forced scoring only).

## Fast-tokenizer alignment (rsuq/align.py) — correctness fix, not just speed

7. Polyseme location now uses OFFSET MAPPING (return_offsets_mapping) instead
   of encode(" "+tok)+len==1. The old check silently DROPPED every multi-piece
   polyseme and assumed GPT-2's leading-space convention. align.py finds the
   word in-context by character span -> token index, handles multi-subword
   words (scores the slot position), and works for any tokenizer/model.
   Wired into run_EA_gate.py (E-A.2) and run_EB_figure1.py.
8. map_span_to_tokens / find_answer_span: char-span -> token-index mapping for
   E-C answer-span attribution (TriviaQA/HaluEval). Eliminates the off-by-one
   token-matching bug class when attributing a per-token width to an answer.
9. Guard: align.py raises if tokenizer.is_fast is False — loud failure instead
   of silent wrong-position scoring.

## E-C/E-D pipeline + OD-1 resolution (measured separation)

10. rsuq/core/sense_inventory.py: OD-1 resolved data-drivenly — a token enters
    Stratum P iff occurrence-state silhouette > tau (validated poly~0.9/mono~0.06).
    OD-2 sense count by best silhouette; OD-3 host cluster via hidden-space
    prototypes. collect_occurrences batched via base_model.
11. run_EC_ED.py: stratified AUROC (EC) + redundancy screen (ED), paired
    bootstrap, answer-span attribution via offset mapping (rsuq.align). Surgical
    claim auto-checked: ctx-W > fixed-W on Stratum P AND null on Stratum M.
12. E-D foil = max-BetP_R confidence (parameter-free), per the locked decision.

## RS-LLM benchmark integration (run_rsllm_bench.py)
13. OBQA/CoQA prompts mirror Shireen's notebooks exactly (citeable baseline
    recipe). Frame=our MiniBatch K-means, head=frozen, for BOTH arms => the
    only variable is fixed-vs-context membership.
14. Sense inventory built from the benchmark's OWN facts/stories (on-task
    context), so Stratum P is task-relevant, not generic-corpus polysemy.
15. OBQA single-option-token scoring (unambiguous label/position); CoQA
    answer-span via offset alignment. Frozen Llama-2-7B 4-bit via rsuq.loading.

## Deferral layer (rsuq/deferral.py) — turns detection into deferral
16. risk_coverage + AURC: selective-prediction metric (the deferral-led headline).
17. compose_aurc + deferral_battery: two-signal composition (conf vs conf+W),
    fit/eval on disjoint halves. The claim conf+W_ctx<conf AND conf+W_fixed~=conf
    is the POSITIVE MIRROR of the deferral paper's negative result.
18. Wired into run_rsllm_bench.analyse(): every run emits deferral_overall +
    per-stratum deferral tables; surgical+deferral verdicts both reported.
