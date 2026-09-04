# Superseded TRAIN-fold partition-invariance test

This exploratory test is superseded by the preregistered
`letter11-permuted` arm. Its reliability-ceiling comparison is retained for
provenance only and is not used for the paper's invariance claim.

Exploratory, seed-0 TRAIN fold only. The decision rule was frozen in
[`partition_invariance_test.py`](../../partition_invariance_test.py) before the
script printed a result: `RATIO_THRESHOLD = 0.50`.

The numeric `.796` reliability in the earlier record is the Pearson
correlation between `V_pos` and the cached numeric alternate form `Valt_pos`.
For this Spearman-only test, the same alternate-form cache gives a TRAIN-fold
Spearman reliability of `0.791`; it is not a split-half reliability. The letter
TRAIN-fold Spearman reliability is `0.261`, giving a geometric reliability
ceiling of `0.454`.

| quantity | value |
|---|---:|
| cross-format Spearman, `V⁺` | 0.205 |
| letter alternate-form Spearman | 0.261 |
| numeric alternate-form Spearman | 0.791 |
| geometric ceiling | 0.454 |
| ratio | 0.452 |
| frozen threshold | 0.500 |
| verdict | fails |

The verdict is therefore that the two item orderings disagree beyond the
prespecified half-ceiling threshold on this TRAIN fold. The quantile-transform
sanity check passed: it left AURC unchanged. It did not produce the verdict;
the verdict comes from the cross-format Spearman correlation relative to the
reliability ceiling.
