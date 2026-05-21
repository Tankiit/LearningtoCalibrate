# Reviewer Question Action Tracker

This tracker converts predictable reviewer objections into paper prerequisites.
The goal is to answer high-severity questions in the submission itself and save
only low-severity or unpredictable questions for rebuttal.

## Bucket A: Must Defend In The Paper

| ID | Reviewer question | Paper location | Required artifact |
|---|---|---|---|
| A1 | Is set size just thresholded semantic entropy? | Method + results | Empirical correlation between set width and semantic entropy, plus a decoupling explanation showing beta-adaptive width is not only cluster-count thresholding. |
| A2 | Is the coverage theorem just split conformal? | Method theorem box | State the finite-sample split-conformal guarantee and clarify what is instantiated: semantic clustering plus beta-parametrized nonconformity. |
| A3 | Why base vs instruct distributions? | Experimental setup | One sentence motivating instruct/base divergence as the epistemic signal; cite the relevant base/instruct divergence reference. |
| A4 | Why LM-as-judge correctness? | Experimental setup | One paragraph explaining judge use, consistency, and any cited reliability precedent. |
| A5 | Why DeBERTa-MNLI for clustering? | Experimental setup + appendix | State it is the semantic-uncertainty default and include a small clustering-judge ablation if time permits. |
| A6 | Why these datasets? | Experimental setup | Describe low, medium, and high ambiguity strata by design. |
| A7 | Does coverage hold empirically? | Results | Coverage verification table at target levels 0.80, 0.90, and 0.95. |

## Bucket B: Mention Briefly, Expand If Pushed

| ID | Reviewer question | In-paper treatment | Rebuttal-ready evidence |
|---|---|---|---|
| B1 | Why `n=20`, not 5 or 40? | One sentence: semantic-uncertainty precedent, `T=1`, `n=20`. | Appendix n-sample ablation showing saturation. |
| B2 | What about Quach-style sequence conformal baselines? | Related work: cluster-level sets are complementary to sequence-level conformal prediction. | Optional post-hoc comparison if Tier-1 experiments finish early. |
| B3 | What is the inference cost? | Limitations sentence. | Point to amortization or sample-efficient semantic prediction as future work. |
| B4 | Why no 70B or API models? | Footnote: open-weight comparable-size models. | Compute-budget and revision-scope answer. |
| B5 | Does NLI clustering preserve exchangeability? | Theorem caveat: deterministic scoring preserves split-conformal coverage; clustering randomness is fixed or audited. | Clustering-seed variance table. |

## Bucket C: Rebuttal Queue

| ID | Reviewer question | Short response |
|---|---|---|
| C1 | AmbigQA annotations are noisy or small. | Report bootstrap confidence intervals and show low-ambiguity negative controls. |
| C2 | Decision-rule kappa is heuristic. | Mention kappa sensitivity if needed. |
| C3 | PopQA has recency bias. | The claim is uncertainty calibration, not recency; optionally show long-tail subset. |
| C4 | What about adversarial prompts? | Out of scope; adversarial uncertainty is a separate setting. |
| C5 | What about multilingual or non-English? | Out of scope for this submission; English QA demonstrates the method. |
| C6 | Why these nonconformity scores? | Appendix ablation across frequency, entropy, and likelihood scores. |
| C7 | What about long-form generation? | Out of scope; cluster-level short-form QA is the controlled testbed. |

## Sprint Prerequisites

1. Generate beta for one pilot cell first.
2. Compute `spearmanr(max(0, -beta), nonconformity_score)` on that pilot.
3. Continue only if the sign and magnitude support adaptive width.
4. Run full beta extraction across model, dataset, and seed matrix.
5. Rerun `scripts/step6_credal_sets.py` and fill coverage-width tables.
6. Add the temperature/n sentence from `paper/temperature_n_sampling_note.md`.
7. Add the Bucket-A defenses before drafting the final four-page version.

## Critical Go/No-Go Metric

Before investing in full extraction, run the pilot-cell check:

```python
spearmanr(np.maximum(0.0, -beta), nonconformity_score)
```

If this correlation is positive and large enough to reduce width at fixed
coverage, the beta-adaptive paper can proceed as planned. If it is negative,
the paper needs either a different beta transform, a different nonconformity
score, or a different empirical claim.
