"""Outils partagés par les scripts de naim/ : caches Modal, identifiants, statistiques.

Les caches viennent du volume Modal ``ova-arr-extract`` (workspace cril-nlp) et sont
téléchargés une seule fois dans ``naim/cached_results/volume/``. Les sorties vont dans
``naim/results/``. Les deux dossiers sont ignorés par git (règles ``cached_results/`` et
``results/``). Lancer les scripts avec ``~/Documents/hf/bin/python``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

NAIM = Path(__file__).resolve().parent
REPO = NAIM.parent
RESULTS = NAIM / "results"
CACHE = NAIM / "cached_results" / "volume"
VOLUME = "ova-arr-extract"
MODELS = ("llama3_8b", "mistral_7b", "qwen2_5_7b")
LEDGER = REPO / "cached_results" / "letter11_provenance_v1" / "summary.csv"
SPLITS_CSV = REPO / "cached_results" / "state_readout" / "cross_fitted_predictions.csv"
BOOT_SEED = 20260915
N_BOOT = 2000

_USED: set[Path] = set()


def _modal_cli() -> str:
    local = Path(sys.executable).with_name("modal")
    if local.exists():
        return str(local)
    found = shutil.which("modal")
    if found is None:
        raise RuntimeError("CLI modal introuvable : lancer avec ~/Documents/hf/bin/python")
    return found


def fetch(remote: str) -> Path:
    """Chemin local d'un fichier du volume, téléchargé au premier appel."""
    local = CACHE / remote
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [_modal_cli(), "volume", "get", VOLUME, "/" + remote, str(local.parent)],
            capture_output=True, text=True)
        if proc.returncode != 0 or not local.exists():
            raise RuntimeError(f"échec du téléchargement de {remote} : {proc.stderr.strip()[-500:]}")
    _USED.add(local)
    return local


def load_pt(remote: str):
    import torch
    # Caches de l'équipe (pickles contenant des tableaux numpy) : weights_only=False nécessaire.
    return torch.load(fetch(remote), map_location="cpu", weights_only=False)


def step1(model: str, name: str, dataset: str = "truthfulqa"):
    """Caches d'origine : hidden_states, logprobs, fine_conf."""
    return load_pt(f"{model}/{dataset}/{name}.pt")


def provenance(model: str, arm: str, dataset: str = "truthfulqa"):
    """Readout conditionnel letter11-v1 ; arm = forward ou reversed."""
    return load_pt(f"letter11_provenance_v1/{model}/{dataset}/{arm}.pt")


def example_ids(blob) -> np.ndarray:
    return np.asarray(blob["example_ids"]).astype(str)


def as_array(x, dtype=np.float64) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=dtype)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes() -> dict[str, str]:
    """SHA-256 des fichiers du volume effectivement lus par le script."""
    return {str(p.relative_to(CACHE)): sha256_file(p) for p in sorted(_USED)}


def spearman(a, b) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def ci(values, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    tail = 50 * (1 - level)
    return float(np.percentile(x, tail)), float(np.percentile(x, 100 - tail))


def read_ledger() -> list[dict]:
    with open(LEDGER) as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
