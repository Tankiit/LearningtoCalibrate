# `letter11-permuted` results

This is the fixed adversarial reversal preregistered in
[`letter11_permuted_preregistration.md`](../../letter11_permuted_preregistration.md):
`A=100% ... L=0%`, run on all three models and both datasets. The new arm was
run as a separate forward pass and paired to the canonical `letter11-v1` cache
by `example_id`.

All eleven labels were injective single-token continuations in all six cells.
The fixed 5pp/at-least-two-effective-bins gate passed for both `V⁺` and the
paired gap in every cell.

The decision rule was frozen before analysis: value-tracking requires the
lower bootstrap endpoint to be at least `0.75 ×` the recomputed v1
alternate-form Spearman reliability; non-invariance requires the upper
endpoint to be below `0.50 ×` that reliability.

The primary correlation is computed in **denoted-value space**, not token
space. `load_V` reads each arm's `Vdist` and forms `V⁺ = Vdist · levels/100`
using that arm's legend. Thus the observed near-zero values are not evidence
for glyph tracking: pure value tracking would predict `rho = +1`, while a
pure unchanged-token/reversed-legend account would predict `rho = -1`.
The observed result is neither account.

| model | dataset | rho(V⁺ v1, V⁺ perm) | 95% interval | v1 reliability | verdict |
|---|---|---:|---:|---:|---|
| Llama | TruthfulQA | 0.020 | [-0.048, 0.093] | 0.351 | legend-noninvariant |
| Llama | Pavlick NLI | -0.004 | [-0.093, 0.086] | 0.268 | legend-noninvariant |
| Mistral | TruthfulQA | -0.026 | [-0.095, 0.045] | 0.365 | legend-noninvariant |
| Mistral | Pavlick NLI | 0.055 | [-0.032, 0.138] | 0.866 | legend-noninvariant |
| Qwen | TruthfulQA | 0.067 | [-0.006, 0.134] | 0.468 | legend-noninvariant |
| Qwen | Pavlick NLI | -0.054 | [-0.145, 0.035] | 0.207 | legend-noninvariant |

The result is consistent across all three models on TruthfulQA and Pavlick NLI:
the reversed legend does not preserve `V⁺` item ordering. The result is a
six-of-six failure of invariance to admissible re-encoding. It does not identify
what replaces the original ordering: neither pure denoted-value tracking nor
pure unchanged-token/glyph tracking predicts the near-zero correlations. The
reversal holds the glyph sequence and list positions fixed, so no mechanism
claim about those components is licensed.

The paired gap changes were small; three of the six item-bootstrap intervals
excluded zero, while the other three included zero. The full values are in
`per_cell.csv`. The numeric101 control is descriptive only:

| dataset | rho(v1, numeric101) | rho(permuted, numeric101) |
|---|---:|---:|
| TruthfulQA | 0.237 | 0.010 |
| Pavlick NLI | 0.239 | 0.005 |

Artifacts:

- `per_cell.csv`: gates, primary correlations, intervals, and paired-gap results
- `numeric101_controls.csv`: Llama clean-readout controls
- `manifest.json`: run and bootstrap metadata
