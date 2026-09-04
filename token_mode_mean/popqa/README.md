# PopQA paired first/mean pooling

The extraction contains 7,377/7,377 deduplicated PopQA items for each of Llama
3.1 8B, Mistral 7B, and Qwen 2.5 7B, with zero skips. First-token and mean-token
answer states were saved from the same forward passes. They therefore share one
`h_q`, one outcome array, and one set of item IDs by construction. The analysis
also asserts exact equality of every per-seed `AURC_q` value.

| model | `AURC_q` | first-token gap | mean-token gap | increase |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 0.136561 | 0.028856 | 0.070773 | +0.041917 |
| Mistral 7B | 0.148181 | 0.033632 | 0.072495 | +0.038863 |
| Qwen 2.5 7B | 0.200180 | 0.038869 | 0.091701 | +0.052832 |

Here `gap = AURC_q - AURC_a`. Mean pooling improves the answer-state reader in
all three models, reproducing the performance direction seen on TruthfulQA.

| model | mode | `cos(d_q,d_a)` | ceiling | fraction of ceiling | permutation p |
|---|---|---:|---:|---:|---:|
| Llama 3.1 8B | first | 0.571 | 0.914 | 62.5% | 0.001 |
| Llama 3.1 8B | mean | 0.423 | 0.923 | 45.8% | 0.001 |
| Mistral 7B | first | 0.443 | 0.898 | 49.4% | 0.001 |
| Mistral 7B | mean | 0.421 | 0.931 | 45.2% | 0.001 |
| Qwen 2.5 7B | first | 0.413 | 0.929 | 44.5% | 0.001 |
| Qwen 2.5 7B | mean | 0.376 | 0.944 | 39.8% | 0.001 |

Unlike TruthfulQA, mean pooling reduces pre/post direction alignment on PopQA
despite improving AURC. The high ceilings (0.90--0.94) rule out sampling noise as
the explanation. Mean answer states are more predictive, but their correctness
direction contains a larger position-specific component. The paper should not
generalize TruthfulQA's increased alignment under mean pooling across free-text
datasets.

Exact rows are in `per_seed.csv`, `summary.csv`, and `direction_cosines.csv`.
The full paired activation caches remain on the `ova-arr-extract` Modal volume at
`/token_mode_mean/{model}/popqa/`; only the numeric results are stored here.
