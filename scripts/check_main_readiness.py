"""
Check readiness for the main-paper model x dataset matrix.

This is a fast, local blocker report. It does not download datasets or launch
Modal jobs; it only checks whether code paths and local artifacts exist.

Usage:
  python scripts/check_main_readiness.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.registry import list_datasets  # noqa: E402
from utils.paths import hidden_states_path  # noqa: E402


P0_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
P0_DATASETS = ["chaosnli", "pavlick_nli", "ambigqa_kge2", "pubmedqa", "truthfulqa", "medqa"]


def _extract_full_registry() -> tuple[set[str], set[str]]:
    mod = importlib.import_module("extraction.extract_full")
    return set(mod.MODEL_REGISTRY), set(mod.DATASETS)


def main() -> None:
    adapter_datasets = set(list_datasets())
    modal_models, modal_datasets = _extract_full_registry()

    print("Main-paper readiness\n")

    print("Models")
    for model in P0_MODELS:
        modal = "yes" if model in modal_models else "no"
        print(f"  {model:<14} modal_registry={modal}")

    print("\nDatasets")
    for dataset in P0_DATASETS:
        adapter = "yes" if dataset in adapter_datasets else "no"
        modal = "yes" if dataset in modal_datasets else "no"
        print(f"  {dataset:<14} adapter={adapter:<3} modal_registry={modal}")

    print("\nLocal hidden-state cache")
    for model in P0_MODELS:
        for dataset in P0_DATASETS:
            status = "cached" if hidden_states_path(model, dataset).exists() else "missing"
            print(f"  {model:<14} {dataset:<14} {status}")

    blockers = []
    for model in P0_MODELS:
        if model not in modal_models:
            blockers.append(f"missing extraction model registry entry: {model}")
    for dataset in P0_DATASETS:
        if dataset not in adapter_datasets:
            blockers.append(f"missing data adapter registry entry: {dataset}")
        if dataset not in modal_datasets:
            blockers.append(f"missing extraction dataset registry entry: {dataset}")

    print("\nBlockers")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
