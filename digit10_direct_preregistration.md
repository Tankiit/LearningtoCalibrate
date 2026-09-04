# Pre-registration — `digit10-direct` format arm

Written 4 September 2026, before the run. This file is the frozen analysis
plan for the arm and must not be edited after the outcome is inspected.

## Why this run

The format effect is currently established on one model. `digits101-v1` is
unreadable on Mistral and Qwen, whose tokenizers split digit runs. A
single-digit alphabet is single-token under both tokenizer families and is
therefore readable on all three models. This run decides whether §1 attributes
the format claim to one model or to three.

The direct arm comes first. It establishes whether the effect exists beyond
Llama; the permuted arm remains the mechanism/indirection follow-up.

## Arm

- `CONF_SCHEME = "digit10-direct"`, levels 0–9 mapped to 0–90%.
- No multi-digit level; no tokenizer digit-run split.
- Prompt template identical to `letter11-v1` apart from the level list and
  mapping sentence.
- Six cells: three models × `{truthfulqa, pavlick_nli}`; same items, fixed
  partition, and seeds as the letter arm.

## Statistic

The paired format contrast on the held-out fold is

    C = ΔAURC(letter11) − ΔAURC(digit10)

`s_q` is common to both arms and cancels exactly. Report per cell with seed SD
and an item-bootstrap interval.

- Primary: `C` on `V⁺`.
- Secondary: `C` on `V⁺ − V⁻`.
- Also report effective bins per cell under each format.

The endpoint differs (90% versus 100%), so no level-based cross-format means,
sums, or additivity comparisons may be computed. The primary statistic is
rank-based.

## Predictions

| cell | predicted sign of `C` | magnitude anchor |
|---|---:|---:|
| llama / truthfulqa | positive | approximately +0.12 |
| llama / pavlick | negative | approximately −0.30 |

For the untested models, Mistral is predicted to show a large effect because
the letter readout is diffuse (`H_norm .89–.93`); Qwen is predicted to show a
small effect because its letter readout is sharp (`H_norm .15–.29`).

## Decision rules

- If the `C` interval excludes zero with consistent sign in at least 2 of 3
  models on TruthfulQA, establish the format effect beyond one model. State it
  as established, not characterised.
- If `C` is indistinguishable from zero on Mistral and Qwen, treat the effect as
  Llama-specific; downgrade the claim to one model and move the paragraph out
  of §1 into Results.
- If Qwen's `|C|` is largest, the concentration account is falsified. Remove
  the mechanism sentence; do not fit a replacement account to these numbers.
- If sign reverses across models within a dataset, report the strongest
  non-identification finding and make no mechanism claim.
- If either Llama cell reverses sign relative to `digits101`, withdraw the
  alphabet-family reading and report the numeric formats separately.

## Confounds and open claims

K differs (10 versus 11), so matching is approximate and controlled by the
pre-existing support-restriction argument, not by design. The scale endpoint
differs (90% versus 100%); this invalidates level-based comparisons across
formats. Glyph and mapping remain entangled. Separating them requires the
permuted-legend arm. The shuffled-legend control is blocked because the legend
is inside `CONF_PROMPT` and permutation requires a new forward pass.

This run does not explain the effect, establish that one format is correct, or
close the single-`a⁻`, off-policy teacher-forcing, or fixed-partition
concessions.

## Frozen run metadata

```json
{
  "conf_scheme": "digit10-direct",
  "conf_values": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
  "models": ["llama3_8b", "mistral_7b", "qwen2_5_7b"],
  "datasets": ["truthfulqa", "pavlick_nli"],
  "seeds": [0, 1, 2],
  "primary_readout": "V_pos",
  "secondary_readout": "V_pos_minus_V_neg",
  "primary_fold": "test",
  "bootstrap": {"unit": "item", "interval": "percentile", "level": 0.95},
  "endpoint_cross_format_comparison": "forbidden",
  "shuffled_legend": "blocked_requires_new_forward_pass"
}
```
