# Task D2 - SCOD-Optimal Selector Comparison

## Goal

Run the SCOD-optimal selector comparison across all available 15
`(model, dataset)` appendix combinations and produce Appendix Table D.2.

Compare:

- `probe_only`
- `gap`
- `scod_optimal_raw`
- `scod_optimal_cal`
- `learned_2d`

## Why this matters

This answers whether the gap signal is close to the best selector available in
the two-dimensional `(p_pred, p_defer)` space. If the gap is close to the
learned 2D upper bound on showcase datasets, it strongly defends the design.

## Inputs

- Cached extraction files under `collaborator_kit/cached_extractions/`, or
  equivalent `outputs/` artifacts from the main pipeline.
- `scripts/scod_optimal_selector.py`.
- `eval/risk_coverage.py`.

## Suggested commands

For a single repository-native config:

```bash
python scripts/scod_optimal_selector.py \
  --config configs/experiments/llama3_8b_truthfulqa.yaml \
  --seed 100 \
  --out collaborator_kit/results_D2_llama3_8b_truthfulqa.parquet
```

For the appendix matrix, run one job per cell and concatenate into
`results_D2.parquet`. Prefer task-local orchestration over editing shared
scripts.

## Output schema

Write one long-format row per fold and method:

```text
task_id = D2
model
dataset
seed
fold
method
metric = aurc
value
n
split = cv_fold
notes
```

Also include a summary table in `appendix_D2.tex` with mean and standard
deviation by `(model, dataset, method)`.

## Interpretation rules

- Lower AURC is better.
- `learned_2d` is the empirical upper bound among simple 2D selectors.
- `scod_optimal_cal` is the calibrated additive SCOD rule.
- The key comparison is `gap` vs `learned_2d`, not just `gap` vs
  `probe_only`.

Report negative results. If gap is not close to learned 2D, that is an
important limitation, not a formatting problem.

## Success criteria

- `results_D2.parquet` exists.
- `appendix_D2.tex` contains Appendix Table D.2 and a short caption paragraph.
- `appendix_D2.pdf` renders.
- `NOTES_D2.md` documents missing cells, calibration failures, and any
  deviations from the canonical representation slice.

