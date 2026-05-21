"""
Modal Llama-3.1-70B judge for Semantic Conformal Prediction correctness.

This grades every (prompt, sample) pair in an SCP generations JSONL and writes
the correctness JSONL expected by ``scp.grade.aggregate_per_prompt``.

Modal volume layout uses an SCP root, separate from the OVA extraction caches:

    /scp/generations/{model}__{dataset}.jsonl
    /scp/correctness/{model}__{dataset}.jsonl

Input row schema:
    prompt_id, question, gold, generations

Output row schema:
    prompt_id, sample_idx, generation, gold, judge_raw, correct, parse_ok

Usage:
    modal run extraction/grade_scp.py --dry-run --small-models
    modal run extraction/grade_scp.py --model llama3-8b --dataset ambigqa
    modal run extraction/grade_scp.py --small-models
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import modal


APP = modal.App("ova-arr-scp-judge")

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.28.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.23.0",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
scp_vol = modal.Volume.from_name("ova-arr-scp", create_if_missing=True)

_VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/scp": scp_vol,
}
_SECRETS = [modal.Secret.from_name("huggingface")]


JUDGE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
SCP_ROOT = Path("/scp")
GEN_DIR = SCP_ROOT / "generations"
CORRECTNESS_DIR = SCP_ROOT / "correctness"

PRIMARY_MODELS = ["llama3-8b", "mistral-7b", "qwen2.5-14b"]
PRIMARY_DATASETS = ["triviaqa", "nq-open", "ambigqa"]

BATCH_SIZE = 32
FLUSH_EVERY = 50
MAX_NEW_TOKENS = 8

JUDGE_PROMPT_TEMPLATE = (
    "You are grading a generated answer against the gold answer(s). "
    "Reply with only YES or NO.\n\n"
    "Question: {question}\n"
    "Gold answer(s): {gold}\n"
    "Generated answer: {prediction}\n\n"
    "Does the generated answer match the gold answer in meaning? "
    "Reply YES or NO."
)

YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_verdict(raw: str) -> bool | None:
    """Parse judge output into True/False/None. First YES/NO wins."""
    yes_m = YES_RE.search(raw or "")
    no_m = NO_RE.search(raw or "")
    if yes_m and (not no_m or yes_m.start() < no_m.start()):
        return True
    if no_m and (not yes_m or no_m.start() < yes_m.start()):
        return False
    return None


def _self_test_parse_verdict() -> None:
    cases = {
        "YES": True,
        "yes": True,
        "Yes, the answer matches.": True,
        "NO": False,
        "No, these differ.": False,
        "Yes. But also no.": True,
        "No. But maybe yes.": False,
        "NONE": None,
        "The answer is unclear": None,
        "": None,
    }
    for raw, expected in cases.items():
        got = parse_verdict(raw)
        assert got is expected, f"{raw!r}: got {got!r}, expected {expected!r}"


def _cell_paths(model_name: str, dataset_name: str) -> tuple[Path, Path]:
    stem = f"{model_name}__{dataset_name}.jsonl"
    return GEN_DIR / stem, CORRECTNESS_DIR / stem


def _done_keys(correctness_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not correctness_path.exists():
        return done
    with correctness_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((str(row["prompt_id"]), int(row["sample_idx"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def _iter_to_grade(
    generations_path: Path,
    done: set[tuple[str, int]],
):
    with generations_path.open() as f:
        for line in f:
            row = json.loads(line)
            prompt_id = str(row["prompt_id"])
            question = str(row["question"])
            gold = row.get("gold", [])
            if isinstance(gold, str):
                gold = [gold]
            generations = row.get("generations", [])
            for sample_idx, generation in enumerate(generations):
                key = (prompt_id, sample_idx)
                if key in done:
                    continue
                yield {
                    "prompt_id": prompt_id,
                    "sample_idx": sample_idx,
                    "question": question,
                    "generation": str(generation),
                    "gold": [str(x) for x in gold],
                }


def _build_prompt(tokenizer, question: str, generation: str, gold: list[str]) -> str:
    gold_str = " / ".join(gold) if gold else "(no gold answer provided)"
    user_msg = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        prediction=generation,
        gold=gold_str,
    )
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_judge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def _generate_verdicts(tokenizer, model, prompts: list[str]) -> list[str]:
    import torch

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[:, prompt_len:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


def _run_cells_for_model(
    model_name: str,
    dataset_names: list[str],
    batch_size: int = BATCH_SIZE,
    resume: bool = True,
) -> None:
    tokenizer, judge_model = _load_judge()
    print(f"[judge] loaded {JUDGE_MODEL} for sample model {model_name}")

    for dataset_name in dataset_names:
        generations_path, correctness_path = _cell_paths(model_name, dataset_name)
        if not generations_path.exists():
            print(f"[skip] missing generations: {generations_path}")
            continue
        correctness_path.parent.mkdir(parents=True, exist_ok=True)

        done = _done_keys(correctness_path) if resume else set()
        work = list(_iter_to_grade(generations_path, done))
        print(
            f"[cell] {model_name}/{dataset_name}: "
            f"{len(done)} done, {len(work)} to grade"
        )
        if not work:
            continue

        t0 = time.time()
        written = 0
        with correctness_path.open("a") as f_out:
            for batch_idx, start in enumerate(range(0, len(work), batch_size)):
                batch = work[start:start + batch_size]
                prompts = [
                    _build_prompt(
                        tokenizer,
                        row["question"],
                        row["generation"],
                        row["gold"],
                    )
                    for row in batch
                ]
                try:
                    raws = _generate_verdicts(tokenizer, judge_model, prompts)
                except Exception as exc:
                    print(
                        f"[batch-error] {model_name}/{dataset_name} "
                        f"batch={batch_idx}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                for row, raw in zip(batch, raws):
                    verdict = parse_verdict(raw)
                    out_row = {
                        "prompt_id": row["prompt_id"],
                        "sample_idx": row["sample_idx"],
                        "generation": row["generation"],
                        "gold": row["gold"],
                        "judge_raw": raw,
                        "correct": bool(verdict) if verdict is not None else False,
                        "parse_ok": verdict is not None,
                    }
                    f_out.write(json.dumps(out_row) + "\n")
                    written += 1

                if (batch_idx + 1) % FLUSH_EVERY == 0:
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    scp_vol.commit()
                    rate = written / max(time.time() - t0, 1.0)
                    print(
                        f"  [{written}/{len(work)}] {rate:.1f} grades/s",
                        flush=True,
                    )

            f_out.flush()
            os.fsync(f_out.fileno())
        scp_vol.commit()

        parse_ok = 0
        total = 0
        with correctness_path.open() as f_check:
            for line in f_check:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                parse_ok += int(bool(row.get("parse_ok")))
        rate = parse_ok / max(total, 1)
        print(
            f"[done] {model_name}/{dataset_name}: wrote {written}, "
            f"parse_ok={parse_ok}/{total} ({100 * rate:.1f}%) -> "
            f"{correctness_path}"
        )


@APP.function(
    image=IMAGE,
    gpu="A100-80GB:2",
    volumes=_VOLUMES,
    timeout=60 * 60 * 8,
    memory=131072,
    secrets=_SECRETS,
)
def grade_llama3_8b(datasets: list[str], batch_size: int, resume: bool) -> None:
    _run_cells_for_model("llama3-8b", datasets, batch_size, resume)


@APP.function(
    image=IMAGE,
    gpu="A100-80GB:2",
    volumes=_VOLUMES,
    timeout=60 * 60 * 8,
    memory=131072,
    secrets=_SECRETS,
)
def grade_mistral_7b(datasets: list[str], batch_size: int, resume: bool) -> None:
    _run_cells_for_model("mistral-7b", datasets, batch_size, resume)


@APP.function(
    image=IMAGE,
    gpu="A100-80GB:2",
    volumes=_VOLUMES,
    timeout=60 * 60 * 8,
    memory=131072,
    secrets=_SECRETS,
)
def grade_qwen2_5_14b(datasets: list[str], batch_size: int, resume: bool) -> None:
    _run_cells_for_model("qwen2.5-14b", datasets, batch_size, resume)


_FN_MAP = {
    "llama3-8b": grade_llama3_8b,
    "mistral-7b": grade_mistral_7b,
    "qwen2.5-14b": grade_qwen2_5_14b,
}


@APP.local_entrypoint()
def main(
    model: str | None = None,
    dataset: str | None = None,
    datasets: str | None = None,
    small_models: bool = False,
    batch_size: int = BATCH_SIZE,
    no_resume: bool = False,
    dry_run: bool = False,
    self_test: bool = False,
) -> None:
    if self_test:
        _self_test_parse_verdict()
        print("OK: parse_verdict self-test passed")
        return

    ds_arg = datasets or dataset
    dataset_list = (
        [d.strip() for d in ds_arg.split(",") if d.strip()]
        if ds_arg else list(PRIMARY_DATASETS)
    )
    model_list = (
        list(PRIMARY_MODELS)
        if small_models else
        [m.strip() for m in model.split(",") if m.strip()] if model else []
    )

    if not model_list:
        print("Choose --model <name> or --small-models.")
        print(f"Known small models: {PRIMARY_MODELS}")
        return

    unknown = [m for m in model_list if m not in _FN_MAP]
    if unknown:
        print(f"Unknown model(s): {unknown}. Valid: {list(_FN_MAP)}")
        return

    if dry_run:
        n_cells = len(model_list) * len(dataset_list)
        print("SCP judge dry run")
        print(f"  root:     {SCP_ROOT}")
        print(f"  models:   {model_list}")
        print(f"  datasets: {dataset_list}")
        print(f"  cells:    {n_cells}")
        print(f"  input:    /scp/generations/{{model}}__{{dataset}}.jsonl")
        print(f"  output:   /scp/correctness/{{model}}__{{dataset}}.jsonl")
        print(f"  judge:    {JUDGE_MODEL} bf16 on A100-80GB:2")
        print(f"  batch:    {batch_size}, max_new_tokens={MAX_NEW_TOKENS}")
        print(f"  estimate: ~40 min/cell, ~{40 * n_cells / 60:.1f} GPU-hours wall")
        return

    for model_name in model_list:
        print(f"[launch] SCP judge for {model_name}: {dataset_list}")
        _FN_MAP[model_name].remote(
            datasets=dataset_list,
            batch_size=batch_size,
            resume=not no_resume,
        )

