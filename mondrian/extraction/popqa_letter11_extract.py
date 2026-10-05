"""E9: PopQA letter-legend confidence reports with the frozen Run 1 elicitation (read-only config).

Launch from mondrian/:
  modal run --detach extraction/popqa_letter11_extract.py --run-id RUN_ID [--limit 50]
One worker per model (A100-40GB). Each loads its model once and writes
/extract/popqa_letter11/RUN_ID/<model>/<form>/<legend>.pt with the Run 1 schema
(V_pos, V_neg, Vdist_*, first_token_letter_mass_*, example_ids, prompt/input hashes).
The frozen Run 1 extractor is not modified; prompts, letter tokens, model revisions and
runtime pins are read from inputs/popqa_letter11/run1_config_readonly.json.
"""
import datetime as dt
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
INPUTS = HERE.parent.parent / "inputs" / "popqa_letter11"
if not INPUTS.exists():
    INPUTS = Path("/root/popqa_letter11")
CONFIG = json.loads((INPUTS / "run1_config_readonly.json").read_text())
C = CONFIG["scientific"]
BATCH = 1   # amendment: bs=32 failed the padding spot check (up to 0.016)
app = modal.App("p1-e9-popqa-letter11")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(*[f"{p}=={v}" for p, v in C["VERSIONS"].items()])
         .add_local_dir(str(INPUTS), remote_path="/root/popqa_letter11")
         .add_local_file(str(HERE), remote_path="/root/popqa_letter11_extract.py"))
volume = modal.Volume.from_name("ova-arr-extract", create_if_missing=False)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prompt(row, form, legend, side):
    values = list(range(0, 101, 10)) if legend == "fwd" else list(range(100, -1, -10))
    line = "  ".join(f"{letter}={v}%" for letter, v in zip(C["LETTERS"], values))
    return C["TEMPLATES"][form].format(q=row["question"], a=row[side], legend=line)


def validate_continuations(tokenizer, text, expected):
    base = tokenizer(text, add_special_tokens=True).input_ids
    for letter, token_id in zip(C["LETTERS"], expected["candidate_token_ids"]):
        full = tokenizer(text + " " + letter, add_special_tokens=True).input_ids
        if full != base + expected["common_prefix_ids"] + [token_id]:
            raise ValueError("Letter-token/prompt-prefix identity failed")
    return base + expected["common_prefix_ids"]


@app.function(image=image, gpu="A100-40GB", timeout=21600, retries=0,
              volumes={"/extract": volume, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def extract_model(model_key, run_id, limit):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = now()
    versions = {p: importlib.metadata.version(p) for p in C["VERSIONS"]}
    if versions != C["VERSIONS"]:
        raise ValueError(f"Pinned runtime mismatch: {versions}")
    qbytes = Path("/root/popqa_letter11/questions.jsonl").read_bytes()
    questions = [json.loads(line) for line in qbytes.decode().splitlines()]
    if limit:
        questions = questions[:limit]
    mid, revision = C["MODELS"][model_key]
    tokenizer = AutoTokenizer.from_pretrained(mid, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    texts, tokens = {}, {}
    for f in C["FORMS"]:
        for l in C["LEGENDS"]:
            for s in C["CANDS"]:
                key = (f, l, s)
                texts[key] = [prompt(q, f, l, s) for q in questions]
                tokens[key] = [validate_continuations(tokenizer, t, CONFIG["tokens"][model_key][l]) for t in texts[key]]
    model = AutoModelForCausalLM.from_pretrained(mid, revision=revision, torch_dtype=torch.float16, device_map="auto",
                                                 attn_implementation=CONFIG["attention_implementation"]).eval()
    device = next(model.parameters()).device
    root = Path("/extract/popqa_letter11") / run_id / model_key
    root.mkdir(parents=True, exist_ok=False)
    meta = dict(artifact="p1_e9_popqa_letter11", dataset="popqa", run_id=run_id, model_key=model_key, model_id=mid,
                model_revision=revision, dtype="float16", padding_side="left", chat_template=False,
                add_special_tokens=True, versions=versions, extractor_sha256=sha(Path("/root/popqa_letter11_extract.py").read_bytes()),
                config_sha256=sha(json.dumps(CONFIG, sort_keys=True).encode()), candidate_source_sha256=sha(qbytes),
                limit=limit, gpu=torch.cuda.get_device_name(device), started_utc=started, letters=list(C["LETTERS"]),
                read_position=CONFIG["read_position"], job_id=modal.current_function_call_id())

    def collect(key, indices):
        f, l, s = key
        enc = tokenizer([texts[key][i] for i in indices], padding=True, return_tensors="pt").to(device)
        prefix_ids = CONFIG["tokens"][model_key][l]["common_prefix_ids"]
        if prefix_ids:
            prefix = torch.tensor(prefix_ids, device=device).expand(len(indices), -1)
            enc["input_ids"] = torch.cat((enc["input_ids"], prefix), dim=1)
            enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(prefix)), dim=1)
        for j, i in enumerate(indices):
            if enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist() != tokens[key][i]:
                raise ValueError("Batch tokenization differs from validated unpadded input")
        ids = torch.tensor(CONFIG["tokens"][model_key][l]["candidate_token_ids"], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits.index_select(1, ids), dim=1).cpu().numpy()
        mass = torch.softmax(logits, dim=1).index_select(1, ids).sum(1).cpu().numpy()
        vals = np.arange(0, 101, 10, dtype=np.float32) if l == "fwd" else np.arange(100, -1, -10, dtype=np.float32)
        return p @ (vals / 100), p, mass

    # determinism check (amendment): first 8 questions twice at bs=1 must agree within SPOT_TOL
    spot_idx = list(range(min(C["N_SPOT_QUESTIONS"], len(questions))))
    worst = 0.0
    for key in texts:
        a = np.concatenate([collect(key, [i])[0] for i in spot_idx])
        b = np.concatenate([collect(key, [i])[0] for i in spot_idx])
        worst = max(worst, float(np.max(np.abs(a - b))))
    (root / "spot_check.json").write_text(json.dumps(dict(meta=meta, max_abs_diff_repeat_bs1=worst, tol=C["SPOT_TOL"])))
    volume.commit()
    if worst > C["SPOT_TOL"]:
        raise ValueError(f"Determinism check failed: {worst}")
    batches = [list(range(i, min(i + BATCH, len(questions)))) for i in range(0, len(questions), BATCH)]
    for f in C["FORMS"]:
        for l in C["LEGENDS"]:
            arrays = {s: [collect((f, l, s), idx) for idx in batches] for s in C["CANDS"]}
            blob = dict(meta=dict(meta, form=f, legend=l, prompt_template=C["TEMPLATES"][f], batch_size=BATCH,
                                  finished_utc=now(), **CONFIG["tokens"][model_key][l]),
                        example_ids=[q["question_id"] for q in questions],
                        conf_values=np.arange(0, 101, 10) if l == "fwd" else np.arange(100, -1, -10))
            for s in C["CANDS"]:
                for j, field in enumerate(("V", "Vdist", "first_token_letter_mass")):
                    blob[f"{field}_{s}"] = np.concatenate([a[j] for a in arrays[s]], axis=0)
                blob[f"prompt_hashes_{s}"] = [sha(t.encode()) for t in texts[(f, l, s)]]
                blob[f"candidate_hashes_{s}"] = [q[f"{s}_sha256"] for q in questions]
            path = root / f / f"{l}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(blob, path)
            volume.commit()
            print(f"Saved {model_key}/{f}/{l}: {len(questions)} questions", flush=True)
    return dict(model=model_key, spot_max_abs_diff=worst, started=started, finished=now())


@app.local_entrypoint()
def main(run_id: str, limit: int = 0):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    calls = {m: extract_model.spawn(m, run_id, limit) for m in C["MODELS"]}
    for m, call in calls.items():
        print(m, json.dumps(call.get()), flush=True)
