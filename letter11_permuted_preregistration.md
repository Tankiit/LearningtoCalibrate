# PREREG letter11-permuted — 4 September 2026

Frozen before any `letter11-permuted` forward pass. The existing
`letter11-v1` and `numeric101` caches may be inspected for pairing, gate
calculation, and control definitions; no new-arm outcome is used to change
these rules.

## Arm

- `CONF_SCHEME = "letter11-permuted"`.
- The alphabet is the same eleven single-token letters as `letter11-v1`:
  `A B C D E F G H J K L` (no `I`).
- The prompt template is identical to `letter11-v1` except for the mapping
  sentence.
- The permutation is fixed and adversarial, not redrawn per item:

  ```text
  A=100%  B=90%  C=80%  D=70%  E=60%  F=50%
  G=40%   H=30%  J=20%  K=10%  L=0%
  ```

  A fixed reversal tests whether the model preserves the denoted value when
  the glyph-to-value order is inverted. A fresh permutation per item would
  estimate a different, averaged instrument and is not this test.
- Cells: all six cells, three models × `{truthfulqa, pavlick_nli}`, using the
  same items and fixed partitions as `letter11-v1`.
- The numeric101 Llama caches are a clean-instrument control for the
  readable, direct numeric arm. They are not assumed to have perfect item
  ordering agreement with either letter arm; their observed agreement is
  reported as a reference, not used as a success threshold.

## Quantities, in the order they will be read

1. **Readability and measurability gate.** Verify that all eleven permuted
   labels are injective single-token continuations under each tokenizer. Then
   apply the fixed 5-percentage-point binning rule and require at least two
   effective bins for the score used in the analysis. A failed gate is
   reported as `not measurable` before any substantive interpretation.
2. **Per item.** Compute `V⁺_perm` and `V⁻_perm` from the new-arm `Vdist` using
   the denoted values in the reversed legend. Pair them by `example_id` with
   `V⁺_v1`, `V⁻_v1`, and, where available, the matched `numeric101` values.
3. **Primary.** Compute Spearman
   `rho_plus = rho(V⁺_v1, V⁺_perm)` per cell. Recompute the `letter11-v1`
   alternate-form reliability per cell from its prespecified alternate
   wording; do not import the Llama/TruthfulQA value of `.401`.
4. **Secondary.** Compute the paired legend difference
   `D_gap = (V⁺_perm - V⁻_perm) - (V⁺_v1 - V⁻_v1)` per item, reporting its
   signed mean and an item-bootstrap interval.
5. **Control.** Report `rho(V⁺_v1, V⁺_numeric101)` and
   `rho(V⁺_perm, V⁺_numeric101)` on the paired Llama cells. This is a
   descriptive clean-readout control, not a new decision rule.

All correlations are Spearman. Pearson substitutions, level-based means
across formats, and rescaling-based comparisons are not permitted.

## Decision rules

The primary classification is per model and dataset. “Well below” is fixed as
the upper endpoint of the two-sided 95% item-bootstrap interval being below
`0.50 × R_v1`, where `R_v1` is the recomputed letter11-v1 alternate-form
reliability for that cell. If `R_v1` is unavailable or non-positive, the cell
is inconclusive rather than reclassified using another threshold.

- **Value-tracking result:** the lower endpoint of the 95% interval for
  `rho_plus` is at least `0.75 × R_v1`. Claim: *the permuted legend preserves
  item ordering to the level supported by the v1 instrument, consistent with
  the report tracking the denoted value rather than the letter glyph alone*.
- **Glyph/position result:** the upper endpoint is below `0.50 × R_v1`.
  Claim: *the report is not invariant to the letter legend; the result is
  consistent with glyph- or legend-position-sensitive readout*. This is a
  mechanism-localisation result, not proof that one of those two causes is
  individually responsible.
- **Intermediate result:** neither rule is met. Claim: *the legend
  sensitivity is unresolved in this cell*.
- **Gate failure:** `not measurable`; no correlation or mechanism claim is
  assigned to that cell.

For paper-level generalisation, the same classification must occur in at least
two of three models on TruthfulQA, with no within-dataset sign reversal. A
single cell or a single model is reported as cell-level evidence only.

## What must not replace a null or an inconclusive result

- No post-hoc remapping of letters to recover a correlation.
- No switch to Pearson if Spearman is weak.
- No selecting `V⁻` because `V⁺` is null; `V⁺` is the deployable object.
- No treating the numeric101 control as proof that a letter result must have
  high correlation; its purpose is to document the clean direct readout.
- No causal claim distinguishing glyph from legend position. The reversal
  varies both and only the future permuted-legend design can separate them.

## Withdrawal condition

If the new arm fails the readability/measurability gate in two or more models,
or if the TruthfulQA result is intermediate or sign-reversing in all but one
model, restrict the sentence “verbalized confidence is not invariant to the
elicitation vocabulary” to the already observed letter/numeric comparison.
Do not generalise it to a legend-specific mechanism, and do not put the
permuted-arm result in the abstract.

## Reporting commitment

Report the fixed reversal, gate status, `V⁺` ordering correlation and interval,
recomputed reliability, paired gap difference and interval, and the numeric101
control values for every cell. State explicitly that `letter11-permuted` versus
`letter11-v1` varies the legend and glyph/position together; it can establish
legend sensitivity, but it cannot by itself identify which surface component
causes it.
