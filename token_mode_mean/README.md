# TruthfulQA mean-token pooling

> **Standing correction (2026-09-01):** values in `direction_cosines.csv` use
> the withdrawn target-selected construction `h_a = where(y, h_pos, h_neg)` and
> must not be interpreted as shared pre/post computation. The `AURC_a` values in
> `summary.csv` and `per_seed.csv` use the fixed-order paired feature
> `[h_pos, h_neg, h_pos-h_neg]` and remain valid. See
> [`paper/session_2026-09-01.md`](../../paper/session_2026-09-01.md).

GPU extraction completed for Llama 3.1 8B, Mistral 7B, and Qwen 2.5 7B with
`token_mode="mean"`: 817/817 aligned items per model and zero skips. The variant
artifacts live under `outputs/step1_extract_variants/token_mode_mean/`; canonical
first-token caches were not overwritten.

The matched comparison uses the same cached mean-logprob outcome and the same
three held-out splits for both pooling modes. As a hard negative control, the
mean-pooling artifacts copy `h_q` from the canonical cache after checking item
IDs and shapes. `h_q` is therefore bitwise identical across modes, and the full
per-seed `AURC_q` arrays are numerically identical (`np.array_equal`), not merely
close.

| model | `AURC_q` | first-token gap | mean-token gap | increase |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 0.481326 | 0.027010 | 0.145831 | +0.118821 |
| Mistral 7B | 0.401690 | 0.008608 | 0.103749 | +0.095141 |
| Qwen 2.5 7B | 0.414234 | 0.063102 | 0.154352 | +0.091250 |

Here `gap = AURC_q - AURC_a`, so positive values mean the answer-state probe
improves on the question-only probe. Mean-token pooling strengthens the gap for
all three models.

## Across-position direction geometry

Directions use the balanced difference of class centroids. The split-half
attenuation ceiling is `sqrt(r_q * r_a)`, where each reliability is the cosine
between directions estimated on disjoint halves of the same labels. The final
column reports observed cosine divided by this ceiling.

| model | mode | `cos(d_q,d_a)` | ceiling | fraction of ceiling | permutation p |
|---|---|---:|---:|---:|---:|
| Llama 3.1 8B | first | 0.258 | 0.434 | 59.5% | 0.001 |
| Llama 3.1 8B | mean | 0.384 | 0.443 | 86.6% | 0.001 |
| Mistral 7B | first | 0.419 | 0.602 | 69.6% | 0.001 |
| Mistral 7B | mean | 0.544 | 0.631 | 86.2% | 0.001 |
| Qwen 2.5 7B | first | 0.229 | 0.554 | 41.2% | 0.001 |
| Qwen 2.5 7B | mean | 0.301 | 0.574 | 52.4% | 0.001 |

Mean pooling increases both the selective-prediction gain and the recoverable
pre/post direction alignment in every model. The answer therefore refines a
direction already established by the question, with the largest residual
geometric change in Qwen.

Reproduce with:

```bash
python scripts/compare_truthfulqa_token_modes.py
python scripts/compare_truthfulqa_cosines.py
```

Detailed outputs are in `per_seed.csv`; TruthfulQA aggregate values are in
`summary.csv`, and the state-channel summary containing both TruthfulQA and
PopQA is in `summary_all_datasets.csv`. The matched state/report comparison is
in `../cached_results/truthfulqa_channel_dissociation.csv`; only TruthfulQA is
shared by the two channel inventories. The geometry analysis is in
`direction_cosines.csv`. Both scripts require
bitwise-identical `h_q`; the token-mode comparison additionally fails unless
the complete per-seed `AURC_q` arrays are exactly equal.
