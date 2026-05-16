"""
Import all-layer/all-pool Modal extraction artifacts into local outputs/.

`extraction/extract_full.py` writes rich caches to the Modal volume
`ova-arr-extract-full`:

  /{model}/{dataset}/contrastive_h.pt
  /{model}/{dataset}/contrastive_meta.pt
  /{model}/{dataset}/gen_greedy_meta.pt
  /{model}/{dataset}/judge_greedy.pt

The existing local training/eval scripts expect the older compact layout:

  outputs/step1_extract/{model}/{dataset}/hidden_states.pt
  outputs/step1_extract/{model}/{dataset}/logprobs.pt
  outputs/step4_eval/{model}/{dataset}/seed{seed}/judge_labels.pt

This script bridges those formats by slicing one canonical representation from
the all-layer cache. Defaults match the paper lock: response_mean, layer -4.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.paths import ensure_parent, hidden_states_path, logprobs_path  # noqa: E402


VOLUME_NAME = "ova-arr-extract-full"
POOL_TO_IDX = {"last": 0, "response_mean": 1, "response_max": 2}


def _read_volume_file(volume: Any, remote_path: str, local_path: Path) -> bool:
    """Download a Modal volume file. Return False if it is absent."""
    try:
        chunks = volume.read_file(remote_path)
        with local_path.open("wb") as f:
            for chunk in chunks:
                f.write(chunk)
    except FileNotFoundError:
        return False
    return True


def _load_remote_pt(volume: Any, remote_path: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(remote_path).name
        if not _read_volume_file(volume, remote_path, local):
            return None
        blob = torch.load(local, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"{remote_path} must contain a dict, got {type(blob).__name__}")
    return blob


def _slice_h(x: Any, layer_rel: int, pool: str) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim == 2:
        return t.float()
    if t.ndim != 4:
        raise ValueError(f"expected hidden states with ndim 2 or 4, got shape {tuple(t.shape)}")
    pool_idx = POOL_TO_IDX[pool]
    return t[:, layer_rel, pool_idx, :].float().contiguous()


def import_cell(
    volume: Any,
    model: str,
    dataset: str,
    *,
    layer_rel: int,
    pool: str,
    strategy: str,
    seed: int,
    force: bool,
    compact: bool,
) -> str:
    h_out = hidden_states_path(model, dataset)
    lp_out = logprobs_path(model, dataset)
    judge_out = Path("outputs") / "step4_eval" / model / dataset / f"seed{seed}" / "judge_labels.pt"

    if h_out.exists() and not force:
        return f"skip existing {model}/{dataset}"

    if compact:
        base = f"/compact/{model}/{dataset}"
        h_blob = _load_remote_pt(volume, f"{base}/hidden_states.pt")
        if h_blob is None:
            return f"missing compact cache {model}/{dataset}"
        torch.save(h_blob, ensure_parent(h_out))

        lp_blob = _load_remote_pt(volume, f"{base}/logprobs.pt")
        if lp_blob is not None:
            torch.save(lp_blob, ensure_parent(lp_out))

        judge_blob = _load_remote_pt(volume, f"{base}/judge_labels.pt")
        if judge_blob is not None:
            torch.save(judge_blob, ensure_parent(judge_out))

        return f"imported compact {model}/{dataset} -> {h_out}"

    base = f"/{model}/{dataset}"
    h_blob = _load_remote_pt(volume, f"{base}/contrastive_h.pt")
    meta_blob = _load_remote_pt(volume, f"{base}/contrastive_meta.pt")
    if h_blob is None or meta_blob is None:
        return f"missing contrastive cache {model}/{dataset}"

    h_pos = _slice_h(h_blob["h_pos"], layer_rel=layer_rel, pool=pool)
    h_neg = _slice_h(h_blob["h_neg"], layer_rel=layer_rel, pool=pool)
    example_ids = np.asarray(meta_blob.get("example_ids") or [
        r.get("example_id") for r in meta_blob.get("records", [])
    ])

    torch.save({
        "h_pos": h_pos,
        "h_neg": h_neg,
        "example_ids": example_ids,
        "meta": {
            "source_volume": VOLUME_NAME,
            "source_layout": "extract_full",
            "model_key": model,
            "dataset": dataset,
            "pool": pool,
            "layer_rel": layer_rel,
            "pool_names": meta_blob.get("meta", {}).get("pool_names"),
            "n_layers": meta_blob.get("meta", {}).get("n_layers"),
        },
    }, ensure_parent(h_out))

    gen_meta = _load_remote_pt(volume, f"{base}/gen_{strategy}_meta.pt")
    if gen_meta is not None:
        torch.save({
            "example_ids": np.asarray(gen_meta.get("example_ids", [])),
            "logprob_pos": np.asarray(gen_meta.get("logprob", []), dtype=float),
            "logprob_neg": np.asarray(gen_meta.get("logprob", []), dtype=float),
            "seq_entropy_pos": np.asarray(gen_meta.get("seq_entropy", []), dtype=float),
            "seq_entropy_neg": np.asarray(gen_meta.get("seq_entropy", []), dtype=float),
            "meta": {
                "source_volume": VOLUME_NAME,
                "source_file": f"gen_{strategy}_meta.pt",
                "note": "Generation logprobs are mirrored to pos/neg slots for legacy evaluator compatibility.",
            },
        }, ensure_parent(lp_out))

    judge_blob = _load_remote_pt(volume, f"{base}/judge_{strategy}.pt")
    if judge_blob is not None:
        labels = judge_blob.get("labels", judge_blob.get("y_correct"))
        torch.save({
            "example_ids": np.asarray(judge_blob.get("example_ids", [])),
            "labels": np.asarray(labels).astype(int),
            "scores": judge_blob.get("scores"),
            "p_yes": judge_blob.get("p_yes"),
            "source_volume": VOLUME_NAME,
            "source_file": f"judge_{strategy}.pt",
        }, ensure_parent(judge_out))

    return f"imported {model}/{dataset} -> {h_out}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import extract_full Modal artifacts locally.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--layer-rel", type=int, default=-4)
    parser.add_argument("--pool", choices=sorted(POOL_TO_IDX), default="response_mean")
    parser.add_argument("--strategy", default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compact", action="store_true",
                        help="Import /compact/{model}/{dataset} artifacts instead of full all-layer tensors.")
    args = parser.parse_args()

    import modal

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    for model in args.models:
        for dataset in args.datasets:
            msg = import_cell(
                volume,
                model,
                dataset,
                layer_rel=args.layer_rel,
                pool=args.pool,
                strategy=args.strategy,
                seed=args.seed,
                force=args.force,
                compact=args.compact,
            )
            print(msg)


if __name__ == "__main__":
    main()
