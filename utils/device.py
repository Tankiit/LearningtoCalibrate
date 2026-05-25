"""Torch device selection helpers."""
from __future__ import annotations

import os

import torch


def get_torch_device(requested: str = "auto") -> str:
    """Return a usable torch device string, preferring accelerators."""
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported torch device: {requested}")

    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_torch_threads(default_threads: int = 2) -> None:
    """Keep torch CPU helper work from using every core."""
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", default_threads))
    torch.set_num_threads(max(1, n_threads))
    try:
        torch.set_num_interop_threads(max(1, n_threads))
    except RuntimeError:
        pass
