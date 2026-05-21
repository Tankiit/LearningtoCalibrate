# Temperature and Sample Count for Semantic Cluster Sampling

This note fixes the default sampling choice for semantic cluster prediction
experiments and explains why temperature and sample count should not be treated
as interchangeable knobs.

## Core Distinction

Sampling temperature controls the distribution being sampled. Sample count
controls how well that distribution is approximated.

At each generation step, the model produces logits `l` and samples from
`softmax(l / T)`.

- `T = 0`: greedy decoding. All samples collapse to the same generation, so
  repeated sampling gives one semantic cluster and is not useful for semantic
  cluster prediction.
- `T < 1`: sharpens the model distribution. This suppresses lower-probability
  semantic alternatives and makes the model look more certain than its native
  predictive distribution.
- `T = 1`: samples from the model's actual predictive distribution. As `n`
  grows, empirical semantic-cluster frequencies converge to that distribution.
- `T > 1`: flattens the distribution. This increases cluster diversity, but the
  additional clusters increasingly reflect temperature-induced token noise
  rather than model uncertainty.

Therefore `n=20, T=1` is not equivalent to `n=10, T=2`, even if both produce a
similar number of clusters. The former estimates the model's own semantic
predictive distribution more accurately; the latter changes the distribution
before estimating it.

## Implication for Nonconformity Scores

Semantic-cluster nonconformity scores such as

```text
s_freq(x, y) = -log p_hat(C_y | x)
```

depend directly on empirical cluster frequency. If temperature is raised, rare
clusters become artificially common. A split-conformal threshold can compensate
for marginal coverage, but the resulting sets are less interpretable as the
model's own semantic uncertainty.

## Default Experimental Choice

Use:

```text
temperature = 1.0
n_samples = 20
```

This matches the standard semantic-uncertainty setting and keeps the experiment
interpretable: only `n_samples` controls Monte Carlo resolution, while
temperature is fixed to the model's native predictive distribution.

## Instruct vs Base Models

Instruct-tuned models may produce fewer clusters than their base counterparts at
the same `T=1` because alignment training can concentrate the output
distribution. This should not be treated as a decoding bug. The conformal
threshold is fit per model, so coverage is recalibrated for each predictive
distribution. Smaller mean set sizes on instruct models are expected when the
instruct distribution is genuinely sharper.

## Paper-Ready Sentence

We sample `n=20` continuations at `T=1.0`, following the semantic-uncertainty
default: temperature fixes the model distribution being estimated, while `n`
only controls Monte Carlo resolution. Instruct models can yield fewer semantic
clusters than base models at the same temperature; our split-conformal threshold
is fit per model and therefore calibrates this difference directly.
