"""Étape 4 (things2do.md §1) : émissions réelles de confiance sur les cellules originales.

Cellules : Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B (révisions de letter11_provenance_extract.py),
817 questions TruthfulQA, les deux candidats mc1, légendes forward/reversed letter11-v1
(A–L sans I). Même prompt de confiance que le readout conditionnel, mais le modèle génère
librement (greedy, pas de masquage du vocabulaire) et on lit la lettre écrite.

Le protocole est figé dans PROTOCOL et son hash est écrit dans chaque sortie. S'il change après
le shard, supprimer results/etape4/shard et recollecter le shard.

1. Vérification locale, sans GPU : ~/Documents/hf/bin/python naim/etape4_verifier_protocole.py
2. Shard, 96 réponses :            ~/Documents/hf/bin/modal run naim/etape4_emissions_modal.py --mode shard
3. Collecte complète, 9 804 :      ~/Documents/hf/bin/modal run naim/etape4_emissions_modal.py --mode complet
4. Analyse :                       ~/Documents/hf/bin/python naim/etape4_analyse_emissions.py --run shard

Les étapes 2 et 3 utilisent des GPU A100 sur le compte Modal de l'équipe.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

MODELS = {
    "llama3_8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "mistral_7b": ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    "qwen2_5_7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
LETTERS = tuple("ABCDEFGHJKL")
LEGENDS = {"forward": tuple(range(0, 101, 10)), "reversed": tuple(range(100, -1, -10))}
BASE_PROMPT = (
    "Question: {q}\nAnswer: {a}\n"
    "How confident are you that this answer is correct?\n"
    "{legend}\nAnswer with a single letter.\nConfidence:"
)
N_QUESTIONS = 817
SHARD_QUESTIONS = 8
BATCH_SIZE = 8

PROTOCOL = {
    "name": "naim-emissions-letter11-v2",
    "changements": "v2 (2026-09-15, après le shard v1) : les blancs et sauts de ligne en tête de réponse "
                   "sont ignorés avant de lire la première ligne. Motif : Mistral écrit souvent '\\nA' "
                   "et v1 le classait 'vide'. Décidé sur la validité du format seulement, sans regarder "
                   "les inversions ; shard v1 archivé dans results/etape4/shard_v1 et shard recollecté.",
    "dataset": "HF truthful_qa/multiple_choice validation ; pos = réponse correcte mc1, "
               "neg = première réponse incorrecte mc1 ; questions sans les deux ignorées",
    "question_id": "sha1('truthfulqa||' + question)[:16], ordre identique au cache provenance-v1",
    "shard": f"les {SHARD_QUESTIONS} premières questions dans cet ordre",
    "prompt_template": BASE_PROMPT,
    "legend_format": "'lettre=valeur%' séparés par deux espaces",
    "letters": LETTERS,
    "legends": LEGENDS,
    "chat_template": False,
    "add_special_tokens": True,
    "padding_side": "left",
    "dtype": "float16",
    "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 8},
    "arret": "EOS du modèle ou premier saut de ligne après le début de la réponse ; la suite est "
             "conservée mais non lue",
    "grammaire": "blancs initiaux ignorés ; première ligne sans espaces autour = exactement une lettre "
                 "permise, suivie au plus d'un '.'",
    "parser": "naim-letter11-strict-v2 (principal) ; lecture tolérante conservée à part",
}
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

_STRICT = re.compile(r"^([A-Z])\.?$")
_LENIENT = re.compile(r"^\s*([A-Z])(?![A-Za-z])")


def prompt_for(question: str, answer: str, legend: str) -> str:
    legend_text = "  ".join(f"{x}={v}%" for x, v in zip(LETTERS, LEGENDS[legend]))
    return BASE_PROMPT.format(q=question, a=answer, legend=legend_text)


def parse_response(text: str, ended: bool) -> tuple[str | None, str | None, str | None]:
    """(lettre stricte ou None, raison d'échec ou None, lettre tolérante ou None)."""
    first_line, newline, _ = text.lstrip().partition("\n")
    lenient = _LENIENT.match(first_line)
    lenient_letter = lenient.group(1) if lenient and lenient.group(1) in LETTERS else None
    stripped = first_line.strip()
    if not stripped:
        return None, "vide", lenient_letter
    match = _STRICT.match(stripped)
    if match is None:
        return None, "texte_en_trop", lenient_letter
    if match.group(1) not in LETTERS:
        return None, "lettre_hors_legende", lenient_letter
    if not (newline or ended):
        return None, "tronque", lenient_letter
    return match.group(1), None, lenient_letter


def decoded_value(letter: str | None, legend: str) -> float | None:
    return None if letter is None else LEGENDS[legend][LETTERS.index(letter)] / 100


def load_truthfulqa() -> list[dict]:
    from datasets import load_dataset

    records = []
    for row in load_dataset("truthful_qa", "multiple_choice", split="validation"):
        targets = row["mc1_targets"]
        correct = [c for c, l in zip(targets["choices"], targets["labels"]) if l == 1]
        wrong = [c for c, l in zip(targets["choices"], targets["labels"]) if l == 0]
        if not correct or not wrong:
            continue
        q = row["question"]
        records.append({"question_id": hashlib.sha1(f"truthfulqa||{q}".encode()).hexdigest()[:16],
                        "question": q, "pos": correct[0], "neg": wrong[0]})
    return records


app = modal.App("naim-emissions-letter11-v1")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.14.0", "transformers==5.16.1", "accelerate==1.14.0", "datasets==5.0.1", "numpy==2.4.6")
hf_cache = modal.Volume.from_name("hf-model-cache")
extract = modal.Volume.from_name("ova-arr-extract")
out_vol = modal.Volume.from_name("naim-emissions", create_if_missing=True)


@app.function(image=image, gpu="A100-40GB", timeout=6 * 3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/extract": extract, "/out": out_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def collect(model_key: str, n_questions: int, run_name: str) -> dict:
    import importlib.metadata

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = MODELS[model_key]
    records = load_truthfulqa()
    cache = torch.load(f"/extract/letter11_provenance_v1/{model_key}/truthfulqa/forward.pt",
                       map_location="cpu", weights_only=False)
    if [r["question_id"] for r in records] != [str(x) for x in cache["example_ids"]]:
        raise RuntimeError("jointure impossible : identifiants différents du cache provenance-v1")
    records = records[:n_questions]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16, device_map="auto").eval()
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tokenizer.eos_token_id}

    jobs = [(r, side, legend) for r in records for side in ("pos", "neg") for legend in LEGENDS]
    rows, started = [], time.time()
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        prompts = [prompt_for(r["question"], r[side], legend) for r, side, legend in batch]
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**enc, **PROTOCOL["generation"], pad_token_id=tokenizer.eos_token_id)
        generated = out[:, enc["input_ids"].shape[1]:].tolist()
        for k, ((r, side, legend), prompt) in enumerate(zip(batch, prompts)):
            ids = generated[k]
            cut = next((i for i, t in enumerate(ids) if t in eos_ids), None)
            kept = ids if cut is None else ids[:cut]
            text = tokenizer.decode(kept, skip_special_tokens=True)
            letter, reason, lenient = parse_response(text, ended=cut is not None)
            mask = enc["attention_mask"][k].tolist()
            rows.append({
                "run": run_name, "protocol_sha256": PROTOCOL_SHA256,
                "model_key": model_key, "model_id": model_id, "revision": revision,
                "question_id": r["question_id"], "candidate_side": side, "correct": int(side == "pos"),
                "legend": legend,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_ids": [t for t, m in zip(enc["input_ids"][k].tolist(), mask) if m],
                "generated_token_ids": kept, "ended_eos": cut is not None, "raw_text": text,
                "parsed_letter": letter, "decoded_value": decoded_value(letter, legend),
                "valid": letter is not None, "failure_reason": reason,
                "lenient_letter": lenient, "lenient_value": decoded_value(lenient, legend),
            })

    path = Path("/out") / PROTOCOL["name"] / run_name / f"{model_key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    out_vol.commit()
    return {
        "model_key": model_key, "rows": rows, "runtime_s": time.time() - started,
        "versions": {p: importlib.metadata.version(p)
                     for p in ("torch", "transformers", "accelerate", "datasets", "numpy")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "example_prompt": prompt_for(records[0]["question"], records[0]["pos"], "forward"),
    }


@app.local_entrypoint()
def main(mode: str = "shard", models: str = ",".join(MODELS)):
    if mode not in ("shard", "complet"):
        raise SystemExit("--mode doit valoir shard ou complet")
    keys = [m.strip() for m in models.split(",") if m.strip()]
    n_questions = SHARD_QUESTIONS if mode == "shard" else N_QUESTIONS
    run_dir = Path(__file__).resolve().parent / "results" / "etape4" / mode
    if (run_dir / "emissions.jsonl").exists():
        raise SystemExit(f"{run_dir}/emissions.jsonl existe déjà : le supprimer pour recollecter")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    results = list(collect.starmap([(k, n_questions, mode) for k in keys]))
    rows = [row for res in results for row in res["rows"]]
    with open(run_dir / "emissions.jsonl", "w") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    expected = len(keys) * n_questions * 2 * len(LEGENDS)
    manifest = {
        "mode": mode, "debut_utc": started, "fin_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
        "parser_sha256": hashlib.sha256(inspect.getsource(parse_response).encode()).hexdigest(),
        "reponses_attendues": expected, "reponses_obtenues": len(rows),
        "par_modele": {res["model_key"]: {k: res[k] for k in ("runtime_s", "versions", "gpu", "example_prompt")}
                       for res in results},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    valid = sum(r["valid"] for r in rows)
    print(f"{len(rows)}/{expected} réponses, {valid} valides ; temps par modèle (s) : "
          + ", ".join(f"{r['model_key']}={r['runtime_s']:.0f}" for r in results))
    print(f"Écrit dans {run_dir}")
