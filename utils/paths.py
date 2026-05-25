"""Canonical output paths.

Every script writes through this module so the layout is consistent and
discoverable from one place. Layout under outputs/:

    outputs/
      step1_extract/{model}/{dataset}/hidden_states.pt
      step1_extract/{model}/{dataset}/logprobs.pt
      step2_probes/{model}/{dataset}/seed{seed}/ova_heads.pt
      step2_probes/{model}/{dataset}/seed{seed}/signals.pt
      step3_conformal/{model}/{dataset}/seed{seed}/conformal.pt
      step4_eval/{model}/{dataset}/seed{seed}/results.parquet
      aggregated_results.parquet

`OUTPUTS_ROOT` is overridable via the OVA_ARR_OUTPUTS env var, which
is useful when running on a shared filesystem where outputs live elsewhere.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path(
    os.environ.get("OVA_ARR_OUTPUTS", REPO_ROOT / "outputs")
)


def ensure_parent(path: Path) -> Path:
    """Create parent directory if missing and return the path unchanged."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)


# ── Step 1 (extraction) ───────────────────────────────────────────────────
def hidden_states_path(model_key: str, dataset: str) -> Path:
    return OUTPUTS_ROOT / "step1_extract" / model_key / dataset / "hidden_states.pt"


def logprobs_path(model_key: str, dataset: str) -> Path:
    return OUTPUTS_ROOT / "step1_extract" / model_key / dataset / "logprobs.pt"


# ── Step 2 (probes) ───────────────────────────────────────────────────────
def _seed_dir(stage: str, model_key: str, dataset: str, seed: int) -> Path:
    return OUTPUTS_ROOT / stage / model_key / dataset / f"seed{seed}"


def ova_heads_path(model_key: str, dataset: str, seed: int) -> Path:
    return _seed_dir("step2_probes", model_key, dataset, seed) / "ova_heads.pt"


def signals_path(model_key: str, dataset: str, seed: int) -> Path:
    return _seed_dir("step2_probes", model_key, dataset, seed) / "signals.pt"


# ── Step 3 (conformal) ────────────────────────────────────────────────────
def conformal_path(model_key: str, dataset: str, seed: int) -> Path:
    return _seed_dir("step3_conformal", model_key, dataset, seed) / "conformal.pt"


# ── Step 4 (evaluation) ───────────────────────────────────────────────────
def results_path(model_key: str, dataset: str, seed: int) -> Path:
    return _seed_dir("step4_eval", model_key, dataset, seed) / "results.parquet"


def aggregated_results_path() -> Path:
    return OUTPUTS_ROOT / "aggregated_results.parquet"
