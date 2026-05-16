# OVA_ARR

Code for *LLMs Know What They Won't Say: Turning Internal-External Disagreement into Calibrated Deferral* (ARR May 2026).

## What this paper is about, in one sentence

The signed gap between two independently calibrated linear probes on frozen LLM hidden states — one trained to predict model correctness, one trained to predict expert reliability — is a calibrated deferral signal that strictly outperforms either probe alone on risk-coverage metrics.

## Where to find what

```
├── extraction/    # Step 1: pull hidden states + generation logprobs from LLMs (Modal)
├── probes/        # Step 2: train OVA heads (f_pred, f_defer) on frozen reps
├── data/          # Dataset adapters: TruthfulQA, HaluEval, TriviaQA, PopQA, BioASQ
├── gap/           # The actual Δ(x) signal + conformal calibration
├── eval/          # AURC, risk-coverage curves, baseline comparisons
└── utils/         # Logging, config loading, RNG, type helpers

configs/           # YAML configs (one per model, dataset, experiment)
scripts/           # Thin CLI entry points — see `scripts/README.md`
tests/             # Sanity tests on synthetic data (run before any real experiment)
outputs/           # Where results land (gitignored, never commit)
```

## Reviewer fast-path

A reviewer asking "where is the gap signal computed?" should be able to find it in 30 seconds:

- **Δ(x) definition:** `gap/signal.py` — the `gap_signal()` function.
- **OVA heads:** `probes/ova.py` — the `OVAHeads` class.
- **Conformal calibration:** `gap/conformal.py` — the `ConformalDeferral` class.
- **AURC:** `eval/risk_coverage.py` — the `aurc()` function.

If you can't find what you need, open an issue.

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

# Sweep all 4 models × 5 datasets
python scripts/run_full_sweep.py
```

## Status

- Extraction pipeline: ported from existing Modal code
- Probe training: ported from existing scripts
- Conformal calibration: implemented
- Evaluation: AURC implemented; baselines pending
- Experiments: ⬜ ⬜ ⬜ ⬜ ⬜ (5 datasets × 4 models — partial; see `outputs/STATUS.md`)
