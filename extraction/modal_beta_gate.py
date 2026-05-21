"""Gate test for beta before full Modal extraction.

Usage:
    modal run extraction/modal_beta_gate.py
    modal run extraction/modal_beta_gate.py --model llama3_8b --dataset truthfulqa
    modal run extraction/modal_beta_gate.py --overwrite

The script computes item-level beta for a small pilot set and writes beta plus
beta_valid into any matching signals.pt files found on the Modal output volume.

For full extraction, use `--full-matrix --n-items 0`; `n_items=0` means all
items in each dataset adapter.
"""
from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("beta-gate-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.28.0",
        "numpy>=1.26.0",
        "scipy",
        "scikit-learn",
        "datasets>=2.18.0",
        "huggingface_hub>=0.23.0",
        "tqdm",
    )
    .add_local_python_source("data", "utils", "beta_psr")
    .add_local_dir(
        "/Users/tanmoy/research/data/chaosNLI_v1.0",
        remote_path="/root/local_data/chaosNLI_v1.0",
    )
)

OUTPUT_VOLUME_NAME = "ova-arr-outputs"
OUTPUTS_MOUNT = Path("/outputs")
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MODEL_PAIRS = {
    "llama3_8b": (
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ),
    "mistral_7b": (
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "qwen2_5_7b": (
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
}

DEFAULT_GATE_CELLS = [
    ("llama3_8b", "truthfulqa"),
    ("llama3_8b", "chaosnli"),
    ("mistral_7b", "truthfulqa"),
    ("mistral_7b", "chaosnli"),
]

DEFAULT_FULL_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DEFAULT_FULL_DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "pubmedqa",
    "medqa",
    "truthfulqa",
    "halueval_qa",
    "triviaqa",
    "popqa",
]

REGIME1_AUROC_GATE = 0.65
REGIME2_SPEARMAN_GATE = 0.10


def _hf_token() -> str:
    """Resolve HF token from Modal secret / env (supports common var names)."""
    import os

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise RuntimeError(
        "Hugging Face token not found. Refresh the Modal secret:\n"
        '  modal secret create huggingface HF_TOKEN="$(cat ~/.cache/huggingface/token)" '
        'HUGGING_FACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token)" --force\n'
        "Also accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B"
    )


def _score_continuation(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    """Teacher-forced log P_model(continuation | prompt)."""
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    answer_ids = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=128,
    ).input_ids.to(device)
    if answer_ids.shape[1] == 0:
        return 0.0
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :]
    targets = full_ids[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    start = max(prompt_ids.shape[1] - 1, 0)
    cont_lp = lp[0, start : start + answer_ids.shape[1], :]
    cont_targets = targets[0, start : start + answer_ids.shape[1]]
    return float(cont_lp.gather(1, cont_targets[:, None]).sum().item())


def _generate_continuations(model, tokenizer, prompt: str, n: int, max_new: int, temp: float, device: str) -> list[str]:
    """Sample n continuations from a model."""
    outputs: list[str] = []
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    for _ in range(n):
        out = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new,
            temperature=max(temp, 1e-3),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0, inputs["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return outputs


def _dominant_cluster(conts: list[str], prompt: str, model, tokenizer, layer: int, device: str, eps: float = 0.15):
    """Cluster continuations by final-token hidden-state cosine distance."""
    import numpy as np
    import torch
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity

    hidden = []
    lps = []
    for cont in conts:
        ids = tokenizer(prompt + cont, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hidden.append(out.hidden_states[layer][0, -1, :].float().cpu().numpy())
        lps.append(_score_continuation(model, tokenizer, prompt, cont, device))

    h = np.stack(hidden)
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    dist = np.clip(1.0 - cosine_similarity(h), 0.0, 2.0)
    labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit(dist).labels_

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        key = 10000 + i if label == -1 else int(label)
        clusters.setdefault(key, []).append(i)

    lp = np.asarray(lps, dtype=float)
    weights = np.exp(lp - lp.max())
    best = max(clusters, key=lambda k: float(weights[clusters[k]].sum()))
    return [(conts[i], lps[i]) for i in clusters[best]]


def _compute_beta(prompt: str, base_model, inst_model, tokenizer, layer: int, device: str, n: int, max_new: int, temp: float) -> float:
    """
    Compute beta(C*, x) as a proper-scoring-rule log score differential.

    C* is the dominant cluster under the base model. beta compares the log
    score awarded to the instruct model and base model on the same realised
    cluster outcome:

        beta = S_log(delta_C*, P_instruct) - S_log(delta_C*, P_base)
             = log P_instruct(C*) - log P_base(C*)
    """
    from beta_psr import compute_beta_psr

    conts = _generate_continuations(base_model, tokenizer, prompt, n, max_new, temp, device)
    cluster = _dominant_cluster(conts, prompt, base_model, tokenizer, layer, device)
    lp_base_members = [lp for _, lp in cluster]
    lp_inst_members = [
        _score_continuation(inst_model, tokenizer, prompt, cont, device)
        for cont, _ in cluster
    ]
    return compute_beta_psr(lp_base_members, lp_inst_members)


def _regime1_gate(betas, labels):
    """AUROC(-beta, hallucination_label)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    betas = np.asarray(betas, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = np.isfinite(betas)
    if valid.sum() < 20 or len(np.unique(labels[valid])) < 2:
        return float("nan"), "insufficient data", False
    auroc = float(roc_auc_score(labels[valid], -betas[valid]))
    passed = auroc > REGIME1_AUROC_GATE
    verdict = f"{'PASS' if passed else 'FAIL'} (AUROC={auroc:.3f}, gate>{REGIME1_AUROC_GATE})"
    return auroc, verdict, passed


def _regime2_gate(betas, entropies):
    """Spearman r(abs(beta), annotation_entropy)."""
    import numpy as np
    from scipy.stats import spearmanr

    betas = np.asarray(betas, dtype=float)
    entropies = np.asarray(entropies, dtype=float)
    valid = np.isfinite(betas) & np.isfinite(entropies)
    if valid.sum() < 20:
        return float("nan"), "insufficient data", False
    res = spearmanr(np.abs(betas[valid]), entropies[valid])
    r = float(res.statistic)
    p = float(res.pvalue)
    passed = r > REGIME2_SPEARMAN_GATE
    verdict = f"{'PASS' if passed else 'FAIL'} (r={r:.3f}, p={p:.4g}, gate>{REGIME2_SPEARMAN_GATE})"
    return r, verdict, passed


def _candidate_signal_paths(model_key: str, dataset_name: str) -> list[Path]:
    paths = []
    for seed in (0, 1, 2):
        paths.append(OUTPUTS_MOUNT / "step2_probes" / model_key / dataset_name / f"seed{seed}" / "signals.pt")
    paths.append(OUTPUTS_MOUNT / model_key / dataset_name / "signals.pt")
    return paths


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=21600,
    memory=80000,
    volumes={
        str(OUTPUTS_MOUNT): output_vol,
        "/root/.cache/huggingface": hf_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_cell(
    model_key: str,
    dataset_name: str,
    n_items: int = 200,
    n_continuations: int = 10,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    overwrite: bool = False,
    seed: int = 42,
    shard_index: int = -1,
    num_shards: int = 1,
) -> dict:
    import numpy as np
    import torch
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data.annotation_entropy import load_entropy_for_dataset
    from data.registry import DATASET_REGISTRY

    hf_token = _hf_token()
    login(token=hf_token, add_to_git_credential=False)

    if model_key not in MODEL_PAIRS:
        return {"status": "error", "error": f"unknown model {model_key}"}
    if dataset_name not in DATASET_REGISTRY:
        return {"status": "error", "error": f"unknown dataset {dataset_name}"}

    items = DATASET_REGISTRY[dataset_name].load()
    n_total = len(items)
    rng = np.random.default_rng(seed)
    manifest_selected = _selected_from_manifest(model_key, dataset_name, items)
    if manifest_selected is not None:
        selected = manifest_selected
    elif n_items <= 0 or n_items >= n_total:
        selected = np.arange(n_total)
    else:
        selected = np.sort(rng.choice(n_total, size=min(n_items, n_total), replace=False))
    if num_shards > 1:
        if shard_index < 0 or shard_index >= num_shards:
            return {"status": "error", "error": f"invalid shard {shard_index}/{num_shards}"}
        selected = selected[shard_index::num_shards]
    selected_ids = np.asarray([items[i].example_id for i in selected], dtype=str)
    prompts = [items[i].question for i in selected]
    full_run = manifest_selected is not None or len(selected) == n_total

    sidecar_path = _beta_sidecar_path(
        model_key,
        dataset_name,
        full=full_run,
        checkpoint=False,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if sidecar_path.exists() and not overwrite:
        print(f"[SKIP] existing beta sidecar: {sidecar_path}")
        return {
            "status": "ok",
            "model": model_key,
            "dataset": dataset_name,
            "n_computed": -1,
            "beta_mean": float("nan"),
            "beta_std": float("nan"),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "gate_verdict": "existing sidecar",
            "gate_passed": True,
        }

    existing_paths = [p for p in _candidate_signal_paths(model_key, dataset_name) if p.exists()]
    if existing_paths and not overwrite:
        already_done = True
        for path in existing_paths:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if "beta" not in raw:
                already_done = False
                break
        if already_done:
            print(f"[SKIP] beta already present for {model_key}/{dataset_name}")

    base_name, inst_name = MODEL_PAIRS[model_key]
    load_kw = {"token": hf_token, "torch_dtype": torch.bfloat16, "device_map": "auto"}

    print(f"Loading tokenizer: {inst_name}")
    tokenizer = AutoTokenizer.from_pretrained(inst_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model: {base_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_name, **load_kw).eval()
    print(f"Loading instruct model: {inst_name}")
    inst_model = AutoModelForCausalLM.from_pretrained(inst_name, **load_kw).eval()
    layer = max(1, int(0.75 * base_model.config.num_hidden_layers))
    print(f"Using layer {layer}/{base_model.config.num_hidden_layers}")

    betas = np.full(len(selected), np.nan, dtype=float)
    for j, prompt in enumerate(prompts):
        if j % 10 == 0:
            print(f"[{model_key}/{dataset_name}] {j}/{len(prompts)} beta_mean={np.nanmean(betas):.3f}")
        try:
            betas[j] = _compute_beta(
                prompt,
                base_model,
                inst_model,
                tokenizer,
                layer,
                device,
                n=n_continuations,
                max_new=max_new_tokens,
                temp=temperature,
            )
        except Exception as exc:
            print(f"[WARN] {model_key}/{dataset_name} item={j}: {exc}")
        if (j + 1) % 25 == 0 or (j + 1) == len(prompts):
            checkpoint = _checkpoint_result(model_key, dataset_name, betas, j + 1, len(prompts))
            _save_beta_sidecar(
                model_key,
                dataset_name,
                selected_ids,
                betas,
                checkpoint | {"shard_index": shard_index, "num_shards": num_shards},
                full=full_run,
                checkpoint=True,
                shard_index=shard_index,
                num_shards=num_shards,
            )
            output_vol.commit()

    result = {
        "status": "ok",
        "model": model_key,
        "dataset": dataset_name,
        "n_computed": int(np.isfinite(betas).sum()),
        "beta_mean": float(np.nanmean(betas)),
        "beta_std": float(np.nanstd(betas)),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }

    if dataset_name == "chaosnli":
        entropies = load_entropy_for_dataset("chaosnli", items)[selected]
        score, verdict, passed = _regime2_gate(betas, entropies)
        result.update({"regime": 2, "spearman_r": score, "gate_verdict": verdict, "gate_passed": passed})
        print(f"REGIME 2 GATE — {verdict}")
        _print_entropy_quartiles(betas, entropies)
    elif dataset_name == "truthfulqa":
        labels = _truthfulqa_labels_from_signals(existing_paths, selected_ids)
        if labels is not None:
            auroc, verdict, passed = _regime1_gate(betas, labels)
            result.update({"regime": 1, "auroc": auroc, "gate_verdict": verdict, "gate_passed": passed})
            print(f"REGIME 1 GATE — {verdict}")
        else:
            result.update({"regime": 1, "gate_verdict": "missing item-level hallucination labels", "gate_passed": False})
            print("REGIME 1 GATE — missing item-level hallucination labels")

    if num_shards <= 1:
        _save_beta_to_signals(existing_paths, selected_ids, betas, overwrite=overwrite)
    _save_beta_sidecar(
        model_key,
        dataset_name,
        selected_ids,
        betas,
        result,
        full=full_run,
        checkpoint=False,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    output_vol.commit()
    return result


def _checkpoint_result(model_key: str, dataset_name: str, betas, n_seen: int, n_total: int) -> dict:
    import numpy as np

    return {
        "status": "checkpoint",
        "model": model_key,
        "dataset": dataset_name,
        "n_seen": int(n_seen),
        "n_total": int(n_total),
        "n_computed": int(np.isfinite(betas).sum()),
        "beta_mean": float(np.nanmean(betas)),
        "beta_std": float(np.nanstd(betas)),
    }


def _selected_from_manifest(model_key: str, dataset_name: str, items) -> object:
    import numpy as np

    manifest = OUTPUTS_MOUNT / "beta_manifests" / model_key / dataset_name / "example_ids.txt"
    if not manifest.exists():
        return None
    wanted = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    by_id = {str(item.example_id): i for i, item in enumerate(items)}
    selected = []
    missing = []
    for example_id in wanted:
        idx = by_id.get(str(example_id))
        if idx is None:
            missing.append(example_id)
        else:
            selected.append(idx)
    if missing:
        print(f"[WARN] manifest {manifest}: {len(missing)} IDs not found; first={missing[:3]}")
    print(f"Using manifest {manifest}: {len(selected)}/{len(wanted)} IDs matched")
    return np.asarray(selected, dtype=int)


def _truthfulqa_labels_from_signals(paths: list[Path], selected_ids) -> object:
    """Best-effort labels: requires a row-level y_correct key, not just is_pos."""
    import numpy as np
    import torch

    if not paths:
        return None
    raw = torch.load(paths[0], map_location="cpu", weights_only=False)
    if "y_correct" not in raw:
        return None
    ids = np.asarray(raw["example_ids"], dtype=str)
    y = np.asarray(raw["y_correct"], dtype=int)
    by_id = {}
    for example_id, label in zip(ids, y):
        by_id.setdefault(str(example_id), int(label))
    try:
        correct = np.asarray([by_id[str(example_id)] for example_id in selected_ids], dtype=int)
    except KeyError:
        return None
    return 1 - correct


def _save_beta_to_signals(paths: list[Path], selected_ids, betas, overwrite: bool) -> None:
    import numpy as np
    import torch

    beta_by_id = {str(example_id): float(beta) for example_id, beta in zip(selected_ids, betas)}
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if "beta" in raw and not overwrite:
            continue
        ids = np.asarray(raw["example_ids"], dtype=str)
        beta = np.full(len(ids), np.nan, dtype=float)
        valid = np.zeros(len(ids), dtype=bool)
        for i, example_id in enumerate(ids):
            if str(example_id) in beta_by_id:
                beta[i] = beta_by_id[str(example_id)]
                valid[i] = np.isfinite(beta[i])
        raw["beta"] = torch.tensor(np.nan_to_num(beta, nan=0.0), dtype=torch.float32)
        raw["beta_valid"] = torch.tensor(valid, dtype=torch.bool)
        torch.save(raw, path)
        print(f"Saved beta to {path} ({int(valid.sum())}/{len(valid)} rows valid)")


def _save_beta_sidecar(
    model_key: str,
    dataset_name: str,
    selected_ids,
    betas,
    result: dict,
    full: bool = False,
    checkpoint: bool = False,
    shard_index: int = -1,
    num_shards: int = 1,
) -> None:
    import numpy as np
    import torch

    out_path = _beta_sidecar_path(
        model_key,
        dataset_name,
        full=full,
        checkpoint=False,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    checkpoint_path = _beta_sidecar_path(
        model_key,
        dataset_name,
        full=full,
        checkpoint=True,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = "beta_full" if full else "beta_gate"
    valid = np.isfinite(betas)
    payload = {
        "model": model_key,
        "dataset": dataset_name,
        "example_ids": np.asarray(selected_ids, dtype=str),
        "beta": np.nan_to_num(np.asarray(betas, dtype=float), nan=0.0).astype("float32"),
        "beta_valid": valid.astype(bool),
        "result": result,
    }
    torch.save(payload, checkpoint_path)
    if not checkpoint:
        torch.save(payload, out_path)
    target = checkpoint_path if checkpoint else out_path
    print(f"Saved beta sidecar to {target} ({int(valid.sum())}/{len(valid)} items valid)")


def _beta_sidecar_path(
    model_key: str,
    dataset_name: str,
    full: bool,
    checkpoint: bool,
    shard_index: int,
    num_shards: int,
) -> Path:
    root = "beta_full" if full else "beta_gate"
    if num_shards > 1:
        root = f"{root}_shards"
    out_dir = OUTPUTS_MOUNT / root / model_key / dataset_name
    suffix = (
        f"shard_{shard_index:04d}_of_{num_shards:04d}.pt"
        if num_shards > 1
        else "beta.pt"
    )
    if checkpoint:
        suffix = f"checkpoint_{suffix}" if num_shards > 1 else "checkpoint.pt"
    return out_dir / suffix


def _print_entropy_quartiles(betas, entropies) -> None:
    import numpy as np

    valid = np.isfinite(betas) & np.isfinite(entropies)
    if valid.sum() < 40:
        return
    abs_beta = np.abs(betas[valid])
    ent = entropies[valid]
    qs = np.percentile(ent, [25, 50, 75])
    print("|beta| by annotation entropy quartile:")
    bounds = [(-np.inf, qs[0], "Q1 low"), (qs[0], qs[1], "Q2"), (qs[1], qs[2], "Q3"), (qs[2], np.inf, "Q4 high")]
    for lo, hi, label in bounds:
        mask = (ent >= lo) & (ent < hi)
        if mask.any():
            print(f"  {label}: mean|beta|={float(abs_beta[mask].mean()):.3f} n={int(mask.sum())}")


@app.local_entrypoint()
def main(
    model: str | None = None,
    dataset: str | None = None,
    n_items: int = 200,
    overwrite: bool = False,
    full_matrix: bool = False,
    models: str | None = None,
    datasets: str | None = None,
    num_shards: int = 1,
):
    if model and dataset:
        cells = [(model, dataset)]
    elif full_matrix:
        model_list = _split_arg(models) if models else DEFAULT_FULL_MODELS
        dataset_list = _split_arg(datasets) if datasets else DEFAULT_FULL_DATASETS
        cells = [(m, d, s) for m in model_list for d in dataset_list for s in range(max(1, num_shards))]
    else:
        cells = [(m, d, -1) for m, d in DEFAULT_GATE_CELLS]

    print(f"Running {len(cells)} beta {'full-matrix' if full_matrix else 'gate'} cells:")
    print(f"n_items: {'all' if n_items <= 0 else n_items}")
    print(f"num_shards: {num_shards}")
    for m, d, s in cells:
        shard_text = f" shard={s}/{num_shards}" if num_shards > 1 else ""
        print(f"  {m} x {d}{shard_text}")

    results = list(run_cell.starmap([
        (m, d, n_items, 10, 50, 0.7, overwrite, 42, s, max(1, num_shards))
        for m, d, s in cells
    ]))
    print("\nBETA GATE TEST RESULTS")
    print("=" * 65)
    for result in results:
        if result.get("status") != "ok":
            print(result)
            continue
        print(f"{result['model']:12s} {result['dataset']:10s} {result.get('gate_verdict', 'no gate')}")

    passed = [r.get("gate_passed", False) for r in results if r.get("status") == "ok"]
    if passed and all(passed):
        print("BOTH GATES PASSED for all completed cells.")
    else:
        print("At least one gate failed or lacked labels; inspect results before full extraction.")


def _split_arg(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]
