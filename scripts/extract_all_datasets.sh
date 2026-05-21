#!/usr/bin/env bash
# scripts/extract_all_datasets.sh
# Run hidden-state extraction for every dataset on Modal (sequential).
#
# Usage:
#   bash scripts/extract_all_datasets.sh                            # all datasets, small models
#   bash scripts/extract_all_datasets.sh --all-models                # all datasets, all 4 models
#   bash scripts/extract_all_datasets.sh --model llama3_8b           # single model
#   bash scripts/extract_all_datasets.sh --datasets popqa            # single dataset
#   bash scripts/extract_all_datasets.sh --datasets popqa,bioasq     # multiple datasets
#   bash scripts/extract_all_datasets.sh --model llama3_8b --datasets popqa
#   bash scripts/extract_all_datasets.sh --dry-run                   # preview only
set -euo pipefail

MODELS_FLAG="--small-models"
DATASETS="truthfulqa,halueval_qa,triviaqa,popqa,bioasq"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-models)  MODELS_FLAG="--all"        ; shift ;;
        --model)       MODELS_FLAG="--model $2"   ; shift 2 ;;
        --datasets)    DATASETS="$2"               ; shift 2 ;;
        --dry-run)     DRY_RUN=1                  ; shift ;;
        -h|--help)
            echo "Usage: $0 [--all-models | --model <key>] [--datasets <csv>] [--dry-run]"
            exit 0 ;;
        *)             echo "Unknown flag: $1"    ; exit 1 ;;
    esac
done

CMD="modal run extraction/modal_app.py $MODELS_FLAG --datasets $DATASETS"
[[ -n "${DRY_RUN:-}" ]] && CMD="$CMD --dry-run"

echo "Running: $CMD"
eval "$CMD"
