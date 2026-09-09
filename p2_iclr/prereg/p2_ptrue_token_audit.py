"""Audit template-produced P(True) continuation IDs before extraction."""
from __future__ import annotations

from transformers import AutoTokenizer

MODELS = {
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
}

# The extractor must use the exact surface form selected after this audit.
TEMPLATE = "Question: {q}\nAnswer: {a}\nIs this answer correct?\nAnswer:"
CONTINUATIONS = ("Yes", "No")


def main() -> int:
    failed = False
    for name, model_id in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as exc:  # fail closed; do not infer tokenization
            print(f"{name}: UNAVAILABLE ({type(exc).__name__}: {exc})")
            failed = True
            continue
        print(name)
        prefix = TEMPLATE.format(q="q", a="a")
        base = tok(prefix, add_special_tokens=False)["input_ids"]
        print(f"  template suffix: {prefix[-24:]!r}")
        read_ids = {}
        for text in CONTINUATIONS:
            full = tok(prefix + text, add_special_tokens=False)["input_ids"]
            continuation = full[len(base):] if full[:len(base)] == base else []
            ok = len(continuation) == 1
            read_ids[text] = continuation[0] if ok else None
            tokens = tok.convert_ids_to_tokens(continuation) if continuation else []
            print(f"  {text!r}: continuation_ids={continuation} tokens={tokens} single={ok}")
            failed |= not ok
        if len(set(v for v in read_ids.values() if v is not None)) != 2:
            print(f"  read set is not injective: {read_ids}")
            failed = True
        else:
            print(f"  read_set={read_ids}")
    if failed:
        print("P(True) tokenization gate: FAIL/INCOMPLETE")
        return 1
    print("P(True) tokenization gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
