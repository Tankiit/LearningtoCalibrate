"""
Compact extract_full Modal artifacts in-place.

The full extraction cache stores all layers and all pools:

  /{model}/{dataset}/contrastive_h.pt

Those files are multi-GB for the larger datasets. This script runs on Modal,
slices the locked representation (default: response_mean / layer -4), and
writes compact artifacts back to the same volume:

  /compact/{model}/{dataset}/hidden_states.pt
  /compact/{model}/{dataset}/logprobs.pt
  /compact/{model}/{dataset}/judge_labels.pt

Local import can then download only the compact files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from itertools import product

import modal


app = modal.App("ova-arr-compact-extract-full")
volume = modal.Volume.from_name("ova-arr-extract-full", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1.0",
    "numpy>=1.26.0",
)

POOL_TO_IDX = {"last": 0, "response_mean": 1, "response_max": 2}
DEFAULT_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DEFAULT_DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "truthfulqa",
    "halueval_qa",
    "triviaqa",
    "popqa",
]


@app.function(image=image, volumes={"/extract_full": volume}, timeout=60 * 60)
def compact_cell(
    model: str,
    dataset: str,
    layer_rel: int = -4,
    pool: str = "response_mean",
    strategy: str = "greedy",
    force: bool = False,
) -> str:
    import numpy as np
    import torch

    if pool not in POOL_TO_IDX:
        raise ValueError(f"unknown pool {pool!r}; choices={sorted(POOL_TO_IDX)}")

    src = Path("/extract_full") / model / dataset
    dst = Path("/extract_full") / "compact" / model / dataset
    dst.mkdir(parents=True, exist_ok=True)

    h_out = dst / "hidden_states.pt"
    lp_out = dst / "logprobs.pt"
    judge_out = dst / "judge_labels.pt"
    if h_out.exists() and not force:
        return f"skip existing compact {model}/{dataset}"

    h_file = src / "contrastive_h.pt"
    meta_file = src / "contrastive_meta.pt"
    if not h_file.exists() or not meta_file.exists():
        return f"missing contrastive cache {model}/{dataset}"

    h_blob = torch.load(h_file, map_location="cpu", weights_only=False)
    meta_blob = torch.load(meta_file, map_location="cpu", weights_only=False)
    pool_idx = POOL_TO_IDX[pool]

    def slice_h(x):
        t = torch.as_tensor(x)
        if t.ndim == 2:
            return t.float().contiguous()
        if t.ndim != 4:
            raise ValueError(f"expected hidden states ndim 2 or 4, got {tuple(t.shape)}")
        return t[:, layer_rel, pool_idx, :].float().contiguous()

    records = meta_blob.get("records", [])
    example_ids = np.asarray(meta_blob.get("example_ids") or [r.get("example_id") for r in records])
    expert_reliable = np.asarray([bool(r.get("expert_reliable", False)) for r in records])
    torch.save(
        {
            "h_pos": slice_h(h_blob["h_pos"]),
            "h_neg": slice_h(h_blob["h_neg"]),
            "example_ids": example_ids,
            "expert_reliable": expert_reliable,
            "meta": {
                "source_volume": "ova-arr-extract-full",
                "source_layout": "extract_full_compact",
                "model_key": model,
                "dataset": dataset,
                "pool": pool,
                "layer_rel": layer_rel,
                "pool_names": meta_blob.get("meta", {}).get("pool_names"),
                "n_layers": meta_blob.get("meta", {}).get("n_layers"),
            },
        },
        h_out,
    )

    gen_meta_file = src / f"gen_{strategy}_meta.pt"
    if gen_meta_file.exists():
        gen_meta = torch.load(gen_meta_file, map_location="cpu", weights_only=False)
        logprob = np.asarray(gen_meta.get("logprob", []), dtype=float)
        seq_entropy = np.asarray(gen_meta.get("seq_entropy", []), dtype=float)
        torch.save(
            {
                "example_ids": np.asarray(gen_meta.get("example_ids", [])),
                "logprob_pos": logprob,
                "logprob_neg": logprob,
                "seq_entropy_pos": seq_entropy,
                "seq_entropy_neg": seq_entropy,
                "meta": {
                    "source_volume": "ova-arr-extract-full",
                    "source_file": f"gen_{strategy}_meta.pt",
                    "note": "Generation scores mirrored to pos/neg slots for legacy evaluator compatibility.",
                },
            },
            lp_out,
        )

    judge_file = src / f"judge_{strategy}.pt"
    if judge_file.exists():
        judge_blob = torch.load(judge_file, map_location="cpu", weights_only=False)
        labels = judge_blob.get("labels", judge_blob.get("y_correct"))
        torch.save(
            {
                "example_ids": np.asarray(judge_blob.get("example_ids", [])),
                "labels": np.asarray(labels).astype(int) if labels is not None else None,
                "scores": judge_blob.get("scores"),
                "p_yes": judge_blob.get("p_yes"),
                "source_volume": "ova-arr-extract-full",
                "source_file": f"judge_{strategy}.pt",
            },
            judge_out,
        )

    volume.commit()
    return f"compacted {model}/{dataset} -> compact/{model}/{dataset}"


@app.local_entrypoint()
def main(
    only_models: Optional[str] = None,
    only_datasets: Optional[str] = None,
    layer_rel: int = -4,
    pool: str = "response_mean",
    strategy: str = "greedy",
    force: bool = False,
):
    models = DEFAULT_MODELS if only_models is None else [m.strip() for m in only_models.split(",")]
    datasets = DEFAULT_DATASETS if only_datasets is None else [d.strip() for d in only_datasets.split(",")]
    pairs = list(product(models, datasets))
    for msg in compact_cell.map(
        [m for m, _ in pairs],
        [d for _, d in pairs],
        kwargs={
            "layer_rel": layer_rel,
            "pool": pool,
            "strategy": strategy,
            "force": force,
        },
        order_outputs=True,
    ):
        print(msg)
