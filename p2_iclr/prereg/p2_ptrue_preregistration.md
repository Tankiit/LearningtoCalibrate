# Pre-registration — P(True) report arm

Frozen 9 September 2026, before any P(True) forward pass.

This is a P2/ICLR preregistration. It is not a P1/AISTATS rule file.

## Purpose

Test a legend-free, binary confidence readout on the same teacher-forced
candidate pairs used by the letter11 arm. This is a second elicited report
scheme, not a replacement for the letter11 result and not a correctness label.

## Arm and readout

- Cells: the three body cells, Llama/Mistral/Qwen × TruthfulQA.
- Same items, candidate answers, fixed partitions, seeds, and question-only
  baseline as the letter11 analysis.
- For each supplied candidate, append the fixed question:
  `Is this answer correct? Answer Yes or No.`
- Let `T` be the exact rendered template prefix ending immediately before the
  model's answer token. Tokenize `T + "Yes"` and `T + "No"` as continuations;
  each must add exactly one token. Record those two continuation IDs per
  model as the read set. Do not infer the spacing from a standalone token
  audit.
- At the read position, let `p_yes` and `p_no` be the full-vocabulary softmax
  probabilities of those two IDs. The primary normalized readout is
  `P(Yes) = p_yes / (p_yes + p_no)`. Record raw pair mass
  `p_yes + p_no` as the availability diagnostic, analogous to letter11's
  full-vocabulary alphabet mass.
- The deployable score is normalized `P(Yes)`; the paired score is
  `P(Yes)_pos - P(Yes)_neg`.
- Record `p_yes`, `p_no`, normalized `P(Yes)`, raw pair mass, continuation IDs,
  prompt hashes, row IDs, and model revisions.

## Tokenization gate

Before extraction, verify `Yes` and ` No` as single tokens for all three
tokenizers, with and without a leading space. The exact prompt surface form
used by the extractor must be recorded. If any required continuation is not
single-token, stop the arm and report it as not measurable; do not silently
switch spacing or extraction position.

## Frozen analysis rules (`app:rules`)

- score: `p_yes_pos - p_yes_neg`, paired, same as `V⁺ - V⁻`;
- deployable score: `p_yes_pos`, reported beside the paired score;
- read position: the position preceding the emitted `Yes`/`No` token;
- adequacy gate: the existing 5-percentage-point score bins and at least two
  effective bins;
- comparator: `AURC_q`, identical folds, seeds, tie convention, and target
  definition;
- outcome per model: `improves`, `does not improve`, or `not measurable`;
- report one row per model in the report table and one row per model in the
  selection-stability table;
- P(True) cells have their own scheme count and do not change the letter11
  statement `one of three cells`;
- no calibration, AUROC, semantic-entropy, or sampled-generation analysis is
  added to this arm;
- a null P(True) result does not establish that the report channel is absent;
  it is a result for this binary readout.

## Stop and provenance rules

The arm is not interpretable if IDs do not align with the canonical caches, if
the prompt hash differs between candidates or models unexpectedly, or if the
tokenization gate fails. Any such outcome is recorded as a provenance or
measurability failure, not converted into a result by changing the prompt.
