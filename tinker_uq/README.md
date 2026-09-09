# Tinker UQ — shared ICLR / AISTATS codebase

Version 0.2.0 · 7 September 2026

A plain-Python package with a command-line interface; no notebooks required.

Train question-only and answer-aware binary correctness reporters with Tinker,
then reuse the frozen scores in two distinct analyses. This package extends the
standalone `tinker_confidence.py`; it does not replace your running frozen-model
experiments. No model weights or private research data are bundled.

## What is implemented

| Component | What it does |
|---|---|
| `tinker_uq/training.py` | Independent q/qa LoRA adapters, full-label custom binary log loss, base/trained scoring, normal/reversed legends, checkpoint paths |
| `tinker_uq/data.py` | Shared input schema, question-level split checks, cross-arm alignment, optional one-candidate sampling |
| `tinker_uq/analysis.py` | Brier/log loss/AUROC, pair accuracy, tie-averaged AURC, paired question-bootstrap Brier gain, legend retention comparisons |
| `tinker_uq/conformal.py` | Fixed anchor, pre-calibration partitions, finite-sample cell quantiles, K sweep, coverage and set-size exports |
| `tinker_uq/demo.py` | Simulated scores for an offline end-to-end demonstration |
| `tests/test_core.py` | Leakage, scoring, sampling and finite-sample edge cases |

## Install

Unzip this project and enter its directory:

```bash
cd tinker_uq
pip install -e .
python -m unittest discover -s tests -v
```

The analysis/demo path needs no Tinker credentials or GPU. For remote training:

```bash
pip install -e '.[tinker]'
tinker auth login
python -m tinker_uq models
```

Use an available instruct model with a Hugging Face chat template. The runner
passes `enable_thinking=False` (Qwen-style); review the rendered prompt audit
before scaling to another family. Base models without a chat template require a
renderer implementation and are rejected. The model ID is deliberately required:
availability is checked against your account instead of hard-coding a stale list.
No model download of full weights is required; the client loads its tokenizer.

## Input contract

One JSON object per line in `cached_answers.jsonl`:

```json
{"row_id":"example-0001","question_id":"question-0001","question":"What is the capital of France?","answer":"Paris","correct":1,"split":"train"}
```

Required fields: `row_id`, `question_id`, `question`, `answer`, `correct`, `split`.
Use globally unique strings for IDs and binary, independently graded correctness.
All four splits must be present. Include the context/options needed to assess the
answer inside `question`. Never insert the gold answer or positive/negative
branch identifier into model inputs. IDs and correctness are not shown to models.

| Split | Permitted role |
|---|---|
| `train` | Train reporters; fit partition edges from frozen train scores |
| `val` | Choose training settings before final analysis; optional independent partition data |
| `cal` | Fit the conformal quantiles after all model/partition choices are fixed |
| `test` | Final diagnostics and prediction-set evaluation |

Reuse the question-level splits of the frozen experiments. Every answer variant,
legend and duplicate of a question must remain in the same split. Text checks
catch exact/whitespace duplicates, not paraphrases: curate those IDs upstream.
Use separate runs per dataset and answer-generating model/decoding policy.
Optional `dataset` and `generator_model` fields must be constant within a run.
Namespace question IDs upstream when needed. The manifest records input hash,
configuration, dependency versions, and the tokenizer template hash.

## Run the offline demo first

```bash
python -m tinker_uq demo --out demo --n 250 --seed 0
python -m tinker_uq audit --data demo/cached_answers.jsonl
python -m tinker_uq iclr --predictions demo/predictions.jsonl --out demo/iclr
python -m tinker_uq fit-conformal \
  --predictions demo/predictions.jsonl --out demo/calibration \
  --k 1 2 4 8 16 --alpha 0.1
python -m tinker_uq evaluate-conformal \
  --predictions demo/predictions.jsonl \
  --calibration demo/calibration/calibration.json --out demo/aistats
```

These are **simulated scores**, not experimental evidence and not trained-model
outputs. `demo_run/` contains an executed example. The synthetic generator is a
software demonstration, not the finite-corpus causal experiment from your paper.
All output directories must be new. This prevents accidental run overwrites.

## Train the shared signals

```bash
python -m tinker_uq audit --data cached_answers.jsonl
python -m tinker_uq train \
  --data cached_answers.jsonl --model YOUR_AVAILABLE_MODEL_ID \
  --out runs/seed0 --seed 0 --epochs 1 --rank 16 --batch-size 8 --lr 0.0001
```

Begin with a small, separate pilot dataset containing all splits. `train` starts
remote training and scoring. One epoch and this learning rate are pilot defaults,
not validated optimal settings. Repeat independent runs with seeds 1 and 2 after
the pilot. Keep the data and splits unchanged across training seeds.

The q and qa adapters both start from the same base model with the same seed,
rank, optimizer and fixed epoch budget. q sees only the question; qa also sees the
cached candidate answer. Only the report-string likelihoods enter the training
loss, but LoRA updates shared model weights. Evaluating the same cached answers
does not prove that the adapted model's answer-generation policy is unchanged.

The normalized probability is

    p_correct = exp(ell_correct) / (exp(ell_A) + exp(ell_B))
    ell_label = sum of log probabilities of every token in label + newline

Training minimizes binary log loss of that same normalized probability. It uses
Tinker's custom loss API, not a regression target of 0 or 100. Prompt/answer tokens
are context only and receive no direct token loss. Completion likelihoods still
backpropagate through that context. All label tokens, including the delimiter,
are scored; there is no first-token approximation or length normalization.

**Interpretation:** this is a binary correctness reporter. The probability is
extracted from label likelihoods, not a generated numeric confidence statement.
The normalization conditions on the two allowed report sequences. It does not
ensure free-generation format compliance; `allowed_label_logmass` exposes how
much mass the model assigns to that restricted set. The normal legend is trained;
the reversed legend is a transfer diagnostic, not proof of legend comprehension.

Before scaling, inspect `prompt_audit.json` for the exact rendered q/qa prompts.
Its first example can contain private data. The code refuses silent truncation
when a sequence exceeds `--max-tokens`.

Outputs:

- `predictions.jsonl`: all four splits × q/qa × base/trained × two legends.
- `manifest.json`: configuration and provenance.
- `prompt_audit.json`: token IDs and rendered example prompts.
- `training.jsonl`: per-step losses and progress.
- `checkpoints.jsonl`: per-epoch optimizer-state and final sampler paths.
- `metrics.json`: basic score diagnostics.

Each training row uses two teacher-forced sequences. Evaluation likewise scores
two complete report strings per prompt, with q-only prompt caching. Custom loss
calls involve extra passes; this first runner prioritizes transparent correctness
over throughput. Checkpoint paths are saved, but automatic interruption resume
and a budget estimator are not implemented. Tinker manages checkpoint retention;
export needed adapters before their configured retention expires.

## ICLR analysis

```bash
python -m tinker_uq iclr \
  --predictions runs/seed0/predictions.jsonl --out runs/seed0/iclr
```

Read `answer_gain.json` first. Positive Brier gain means qa improves over q.
Question-bootstrap intervals resample questions and average per-question gains;
they do not include training-seed variability. `metrics.json` also reports
within-question positive-versus-negative pair accuracy with half credit for ties.
`legend_transfer.json` reports rank agreement and retained-set overlap/error at
50%, 80%, and 90% retention. Retention ties use row ID, never correctness; interpret
retention results with care if confidence ties are common. AURC averages over
random within-tie ordering. Train metrics are descriptive only.

For a balanced contrastive dataset with one positive and one negative per question,
the q-only target is 0.5 by construction; q-only pair accuracy is exactly 0.5.
Such data tests candidate discrimination, not natural question difficulty.
Use naturally sampled, independently graded answers for the difficulty question.
Improvement beyond q is predictive gain, not an identified information fraction
or proof of introspection. These supervised reporters complement the existing
frozen hidden-state probes; training changes the object being measured.

## AISTATS: how many cells?

```bash
python -m tinker_uq fit-conformal \
  --predictions runs/seed0/predictions.jsonl --out runs/seed0/calibration \
  --k 1 2 4 8 16 --alpha 0.1
python -m tinker_uq evaluate-conformal \
  --predictions runs/seed0/predictions.jsonl \
  --calibration runs/seed0/calibration/calibration.json --out runs/seed0/aistats
```

The default anchor is trained qa under the normal legend. Hold it fixed while
comparing partitions built from q versus qa. This isolates the partition signal
from changes to the nonconformity score. `K=1` is global split conformal; its
results must be identical for the two partition signals. Use `--anchor q` or
`--stage base` for additional prespecified comparisons, in separate output folders.

For input x=(question,candidate), the outcome z is candidate correctness:

    R(x,0) = p_anchor(correct|x)
    R(x,1) = 1 - p_anchor(correct|x)
    C(x) = {z in {0,1}: R(x,z) <= qhat_cell(x)}

Partition edges use training scores only. Equal scores stay together, so effective
K can be lower than requested K. Calibration scores never determine the edges.
`--partition-split val` uses the validation scores instead, and supports the old
standalone export that omitted training predictions. In either case, freeze the
partition before using calibration labels.

The cell threshold is the exact `ceil((n_cell+1)*(1-alpha))`-th order statistic.
If that rank exceeds n_cell (including an empty cell), the threshold is infinity
and the set is `{0,1}`. JSON stores infinity as `null`. No global fallback is
substituted into a small cell. Inclusive boundaries preserve the conservative
handling of ties. Empty prediction sets are allowed and reported honestly.

Outputs: `summary.csv`/`summary.json` (coverage and mean size versus K),
`cells.jsonl` (cell sizes, thresholds, coverage), and `sets.jsonl` (per-test sets).
Wilson intervals describe test coverage uncertainty conditional on the fitted
calibrator under i.i.d. test sampling; they do not cover calibration-run variability
and are not simultaneous across cells. Worst observed cell coverage can be noisy.

The sweep is a **fixed-K comparison**, not a learned K selector. Do not pick K
using final calibration/test performance and then attach an unadjusted guarantee.
A data-driven selector needs its own pre-calibration development protocol. Repeat
calibration draws using a separate prespecified experiment to study variability;
this version does not estimate quantization error or prove an optimal-K rate.

### Sampling unit and scope of the guarantee

Default: exactly one candidate per question for conformal analysis. If there are
multiple candidates, the command stops. To explicitly sample one candidate per
question, identically across arms and splits, add:

```bash
--candidate-policy sample --seed 0
```

Sampling is uniform within question and uses IDs/seed only, never labels or
scores. It changes the estimand to that sampling scheme; sampling from artificial
balanced pairs does not reconstruct natural generated-answer accuracy. The usual
cell-conditional lower coverage statement requires exchangeable calibration/test
units and a fixed predictor/partition. Code checks alone cannot establish those
assumptions. There is no guarantee for arbitrary dataset shift, no pointwise
conditional-coverage claim, and no simultaneous coverage of all answer variants.

These sets concern `{incorrect, correct}` for a supplied answer. They are not
sets of possible answers, numerical confidence intervals, or deferral risk
bounds. In particular, a singleton `{correct}` is not automatically a 90%-precision
acceptance policy. Answer-set prediction and selective-risk control need their
own target and score definitions.

## Validation status

Offline tests passed for data leakage, score alignment, missing logprobs, reversed
legends, order-statistic ranks, ties, small/empty cells, label-blind sampling, and
cross-arm metadata alignment. The synthetic demo ran through both paper paths.
The optional torch gradient test is skipped when torch is absent. No live Tinker
training or real-data research result was produced in this environment. Run the
small live pilot and inspect its saved prompts/losses before scaling.

## References checked for implementation

- [Tinker custom loss API](https://tinker-docs.thinkingmachines.ai/tinker/losses/custom/)
- [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
- [Sampling and token log probabilities](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/)
- [Angelopoulos and Bates: conformal tutorial, including group-balanced coverage](https://arxiv.org/abs/2107.07511)
