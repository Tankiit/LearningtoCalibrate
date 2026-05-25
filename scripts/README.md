# scripts/

Thin CLI entry points. Each `stepN_*.py` is idempotent: rerunning with the
same config skips already-completed work. They all read a single YAML
config file via `--config configs/experiments/<name>.yaml`.

| script                    | reads from                         | writes to                                         |
|---------------------------|------------------------------------|---------------------------------------------------|
| `step1_extract.py`        | configs/, Modal volume             | `outputs/step1_extract/{model}/{dataset}/`        |
| `step2_train_probes.py`   | step 1 outputs + dataset registry  | `outputs/step2_probes/{model}/{dataset}/seed*/`   |
| `step3_calibrate.py`      | step 2 signals                     | `outputs/step3_conformal/{model}/{dataset}/seed*/`|
| `step4_evaluate.py`       | steps 2 + 3                        | `outputs/step4_eval/{model}/{dataset}/seed*/`     |
| `aggregate_results.py`    | all step 4 results.parquet         | `outputs/aggregated_results.parquet`              |
| `run_full_sweep.py`       | configs/                           | runs steps 1-4 across the (model × dataset × seed) grid |

## Single (model, dataset)

```bash
python scripts/step1_extract.py        --config configs/experiments/llama3_8b_truthfulqa.yaml
python scripts/step2_train_probes.py   --config configs/experiments/llama3_8b_truthfulqa.yaml
python scripts/step3_calibrate.py      --config configs/experiments/llama3_8b_truthfulqa.yaml
python scripts/step4_evaluate.py       --config configs/experiments/llama3_8b_truthfulqa.yaml
```

## Full sweep

```bash
python scripts/run_full_sweep.py
python scripts/aggregate_results.py
```

Use `--skip-extract` once Modal extraction is done; use `--up-to-step N` to
stop early (e.g. `--up-to-step 2` to train probes only).
