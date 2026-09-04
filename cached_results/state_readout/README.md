# Shared-state readout test

This cached test uses the same `h_pos` answer state to predict `V_pos` under
`letter11` and `digits101`. Each dataset/fold has one shared scaler/PCA basis;
only the two ridge heads differ. Predictions are held out under seeds 0, 1, 2.

The ordering statistics are Spearman rank correlations, not the Pearson values
used in the earlier reliability/disattenuation table:

| dataset | Spearman(V_letter, V_digit) | Pearson |
|---|---:|---:|
| truthfulqa | 0.237 | 0.185 |
| pavlick_nli | 0.239 | 0.228 |

Held-out state-readout performance is asymmetric:

| dataset | letter11 R² | digits101 R² |
|---|---:|---:|
| truthfulqa | 0.138 ± 0.006 | 0.319 ± 0.028 |
| pavlick_nli | 0.233 ± 0.015 | 0.508 ± 0.114 |

The split-half within-format weight cosines are unstable. Their means are
0.067 (letter11) and −0.019 (digits101) on TruthfulQA, and 0.112 and 0.097 on
Pavlick NLI, with substantial seed variation. The cross-format weight cosine is
therefore not used as evidence.

The correlations between the two cross-fitted held-out prediction vectors are
also modest: Pearson 0.209 ± 0.039 and Spearman 0.167 ± 0.051 on TruthfulQA;
Pearson 0.353 ± 0.030 and Spearman 0.367 ± 0.039 on Pavlick NLI. Thus the
cached test does not support either a simple shared-scalar retrieval account or
a strong, stable claim that the two formats recover cleanly distinct state
directions.
