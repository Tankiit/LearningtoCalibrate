# Task G1 - SelfCheckGPT Baseline

## Goal

Implement and run a SelfCheckGPT baseline on ChaosNLI and TruthfulQA, then
compare it against semantic entropy and the gap method.

Use either:

- SelfCheckGPT-NLI, preferred if the NLI model is easy to run on your cluster.
- SelfCheckGPT-BERTScore or embedding-consistency variant, acceptable if NLI is
  blocked.

## Why this matters

This defends against the reviewer concern that the paper compares against only
one LLM-uncertainty baseline.

## Inputs

- `nsamples10.pt` for ChaosNLI and TruthfulQA.
- Greedy generations and judge labels.
- Existing semantic-entropy outputs if available.

## Method sketch

For each example:

1. Take the greedy answer.
2. Take the 10 sampled answers from `nsamples10.pt`.
3. Score consistency between the greedy answer and samples.
4. Convert inconsistency into an uncertainty score.
5. Compute AURC using the same `y_correct` labels as the main evaluation.

For NLI SelfCheckGPT, use contradiction probability as inconsistency. For
embedding/BERTScore SelfCheckGPT, use `1 - mean_similarity`.

## Output schema

Write `results_G1.parquet` with:

```text
task_id = G1
model
dataset
seed
method
metric = aurc
value
n
split
notes
```

Required methods:

- `gap`, if available from the cache or pipeline outputs.
- `semantic_entropy`, if available.
- `selfcheckgpt_nli` or `selfcheckgpt_similarity`.

## Appendix prose

Write a two-paragraph `appendix_G1.tex`:

1. Explain the SelfCheckGPT variant and implementation choice.
2. Summarize the table and state whether the gap remains competitive.

Be precise about the variant. Do not call an embedding-consistency proxy
"SelfCheckGPT-NLI".

## Success criteria

- `results_G1.parquet` exists for ChaosNLI and TruthfulQA.
- `appendix_G1.tex` contains Table G.1 and two paragraphs.
- `appendix_G1.pdf` renders.
- `NOTES_G1.md` records model names, prompt formatting, NLI/embedding model,
  and any generation-cache mismatch.

