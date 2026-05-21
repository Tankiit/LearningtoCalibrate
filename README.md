# OVA_ARR

Code for *LLMs Know What They Won't Say: Turning Internal-External Disagreement into Calibrated Deferral* (ARR May 2026).

## What this repository tests

This repo tests whether a signed probe gap can serve as a deferral score:

```text
gap(x) = f_pred(x) - f_defer(x)
```

where `f_pred` predicts whether the model answer is correct and `f_defer`
predicts the dataset's expert-reliability / ambiguity signal. The current
matrix results are important: across the completed cells, the gap generally
underperforms `probe_pred_only`. The diagnostics show why: `f_defer` often
learns its `y_expert` target, but `y_expert` is mostly uncorrelated with model
error.

## Where To Find What

```
├── extraction/    # Step 1: pull hidden states + generation logprobs from LLMs (Modal)
├── probes/        # Step 2: train OVA heads (f_pred, f_defer) on frozen reps
├── data/          # Dataset adapters and the dataset registry
├── gap/           # The actual Δ(x) signal + conformal calibration
├── eval/          # AURC, risk-coverage curves, baseline comparisons
└── utils/         # Logging, config loading, RNG, type helpers

configs/           # YAML configs (one per model, dataset, experiment)
scripts/           # Thin CLI entry points — see `scripts/README.md`
tests/             # Sanity tests on synthetic data (run before any real experiment)
outputs/           # Where results land (gitignored, never commit)
results/           # Diagnostics and matrix summaries
```

## Core Code Path

The main implementation pieces are:

- **Dataset registry:** `data/registry.py`
- **Dataset schema:** `data/schema.py`
- **Dataset adapters:** `data/adapters/*.py`
- **OVA heads:** `probes/ova.py`
- **Probe training:** `probes/training.py`
- **Gap signal:** `gap/signal.py`
- **Conformal calibration:** `gap/conformal.py`
- **AURC / risk-coverage:** `eval/risk_coverage.py`
- **Baseline score wrappers:** `eval/baselines.py`
- **End-to-end per-cell evaluation:** `scripts/step4_evaluate.py`

The most important lines are:

```python
# gap/signal.py
gap = logit_pred - logit_defer
sigma_pred = sigmoid(logit_pred)
sigma_defer = sigmoid(logit_defer)
```

and:

```python
# eval/baselines.py
baseline_gap(signals)        -> signals.gap
baseline_sigma_pred(signals) -> signals.sigma_pred
```

`scripts/step4_evaluate.py` compares `gap`, `probe_pred_only`, `logit_pred`,
`probe_amb_only`, `phillips_sum`, and optional generation logprob baselines on
the same evaluation rows.

## Dataset Adapters

Medical / Interpretation A additions:

- `data/adapters/pubmedqa.py`
- `data/adapters/medqa.py`

Main showcase / diagnostic adapters:

- `data/adapters/chaosnli.py`
- `data/adapters/pavlick_nli.py`
- `data/adapters/ambigqa_kge2.py`
- `data/adapters/truthfulqa.py`
- `data/adapters/halueval_qa.py`
- `data/adapters/triviaqa.py`
- `data/adapters/popqa.py`

Dataset configs live in `configs/datasets/`. Experiment configs are generated
or stored under `configs/experiments/`.

## Pipeline overview

```
configs/        Modal volume         local artifacts          tables/figures
   │                  │                     │                      │
   ▼                  ▼                     ▼                      ▼
   ├─►  step1_extract.py  ─►  hidden_states.pt + logprobs.pt
   │
   ├─►  step2_train_probes.py  ─►  ova_heads.pt + signals.pt
   │
   ├─►  step3_calibrate.py  ─►  conformal_thresholds.pt
   │
   └─►  step4_evaluate.py  ─►  results.parquet (AURC, risk-coverage curves)
```

Each step is idempotent: re-running with the same config skips already-done work.

## Matrix Runner

Use Modal for extraction, then local scripts for probe training and evaluation:

```bash
# Modal extraction
modal run extraction/modal_app.py --model llama3_8b,mistral_7b,qwen2_5_7b \
  --datasets chaosnli,pavlick_nli,ambigqa_kge2,pubmedqa,truthfulqa,medqa

# Local matrix once extraction artifacts are cached
TORCH_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python scripts/run_main_matrix.py --tasks p0 \
  --models llama3_8b mistral_7b qwen2_5_7b \
  --datasets chaosnli pavlick_nli ambigqa_kge2 pubmedqa truthfulqa medqa \
  --seeds 0 1 2 --skip-extract
```

The thread caps keep local probe/diagnostic work from consuming all CPU cores.

## Important Diagnostics

These diagnostics were added because the headline gap score underperformed.

### Correlation With Correctness

`scripts/inspect_signals.py` now prints:

```python
corr(p_pred, y_correct)
corr(gap,    y_correct)
```

Matrix-level outputs:

- `results/gap_correctness_correlations.parquet`
- `results/gap_correctness_correlations_summary.csv`

Observed pattern: `p_pred` correlates more strongly with correctness than `gap`
in every completed cell.

### Defer-Head Diagnostics

`scripts/defer_head_diagnostics.py` reports:

```python
f_defer accuracy on y_expert
corr(f_defer predictions, y_correct)
corr(model error, y_expert)
y_correct == 1 count in the eval set
```

Outputs:

- `results/defer_head_diagnostics.parquet`
- `results/defer_head_diagnostics_summary.csv`

Observed pattern: `f_defer` often learns `y_expert`, but `y_expert` is not
meaningfully correlated with model error. Therefore subtracting `f_defer` from
`f_pred` usually hurts selective prediction.

## Quick start

```bash
# Set up environment
pip install -e .

# Run synthetic-data sanity check (5 minutes, no GPU)
pytest tests/

# Run on one (model, dataset) end-to-end
python scripts/step1_extract.py    --config configs/experiments/llama3_8b_truthfulqa.yaml
python scripts/step2_train_probes.py --config configs/experiments/llama3_8b_truthfulqa.yaml
python scripts/step3_calibrate.py    --config configs/experiments/llama3_8b_truthfulqa.yaml
python scripts/step4_evaluate.py     --config configs/experiments/llama3_8b_truthfulqa.yaml

# Sweep selected models × datasets
python scripts/run_full_sweep.py --skip-extract
```

## Current Result Files

- Aggregated headline AURC table: `outputs/aggregated_results.parquet`
- Main matrix status: `results/main_matrix_status.parquet`
- SCOD diagnostics: `results/main_matrix/scod/`
- y-expert semantic audits: `results/main_matrix/y_expert_audit/`
- Gap correctness correlations: `results/gap_correctness_correlations_summary.csv`
- Defer-head diagnostics: `results/defer_head_diagnostics_summary.csv`

## Current Empirical Status

Lower AURC is better. In the completed runs, `gap` is worse than
`probe_pred_only` on the key showcase and medical cells:

- ChaosNLI, Pavlick-NLI, AmbigQA-k>=2: `gap` underperforms `probe_pred_only`.
- PubMedQA: `gap` underperforms `probe_pred_only` for all three models.
- MedQA: `gap` strongly underperforms, consistent with the diagnostic-regime
  prediction.
- HaluEval-QA and TriviaQA: `gap` is nearly tied but slightly worse.
- PopQA: `gap` is much worse.

This means the honest current interpretation is not "the gap dominates"; it is:
the mechanism learns the configured `y_expert` signal, but that signal does not
track model error well enough for the gap to improve risk-coverage.
