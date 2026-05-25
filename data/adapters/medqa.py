"""
MedQA adapter.

Uses a US English 4-option MedQA/USMLE subset. MedQA has no per-item
annotator-disagreement field, so y_expert is a metadata-derived diagnostic
proxy based on question length.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.schema import Record, stable_id

DEFAULT_MAX_ROWS = 12000


@dataclass
class MedQAItem:
    item_id: str
    question_stem: str
    options: dict[str, str]
    gold_option_key: str
    gold_option_text: str
    distractor_key: str
    distractor_text: str
    question_length: int
    y_expert: float
    y_expert_binary: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question_stem": self.question_stem,
            "options": self.options,
            "gold_option_key": self.gold_option_key,
            "gold_option_text": self.gold_option_text,
            "distractor_key": self.distractor_key,
            "distractor_text": self.distractor_text,
            "question_length": self.question_length,
            "y_expert": self.y_expert,
            "y_expert_binary": self.y_expert_binary,
        }


def _inspect_dataset(ds_split) -> dict[str, Any]:
    cols = list(ds_split.column_names)
    info: dict[str, Any] = {"columns": cols, "n_items": len(ds_split)}

    if "question" in cols:
        info["question_column"] = "question"
    else:
        raise KeyError(f"No question column. Available: {cols}")

    if "options" in cols:
        info["options_column"] = "options"
        info["options_strategy"] = "options"
    elif "choices" in cols:
        info["options_column"] = "choices"
        info["options_strategy"] = "choices"
    else:
        raise KeyError(f"No options/choices column. Available: {cols}")

    if "answer_idx" in cols:
        info["answer_column"] = "answer_idx"
        info["answer_strategy"] = "key_string"
    elif "answer" in cols:
        info["answer_column"] = "answer"
        info["answer_strategy"] = "text_or_key"
    else:
        raise KeyError(f"No answer/answer_idx column. Available: {cols}")
    return info


def _extract_options(row: dict[str, Any], schema: dict[str, Any]) -> dict[str, str]:
    raw = row[schema["options_column"]]

    if isinstance(raw, dict):
        if "key" in raw and "value" in raw:
            return {
                str(k).strip().upper(): str(v).strip()
                for k, v in zip(raw["key"], raw["value"])
            }
        return {str(k).strip().upper(): str(v).strip() for k, v in raw.items()}

    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            out = {}
            for opt in raw:
                key = opt.get("key") or opt.get("label")
                value = opt.get("value") or opt.get("text")
                if key is not None and value is not None:
                    out[str(key).strip().upper()] = str(value).strip()
            return out
        keys = ["A", "B", "C", "D", "E"][:len(raw)]
        return {k: str(v).strip() for k, v in zip(keys, raw)}

    raise ValueError(f"Unsupported options format: {type(raw).__name__}")


def _find_gold_key(row: dict[str, Any], schema: dict[str, Any], options: dict[str, str]) -> str:
    raw = row[schema["answer_column"]]
    answer = str(raw).strip()
    answer_upper = answer.upper()
    if answer_upper in options:
        return answer_upper

    if schema["answer_strategy"] in {"text", "text_or_key"}:
        for key, text in options.items():
            if text == answer or text.lower() == answer.lower():
                return key
    raise ValueError(f"Could not match answer {answer!r} to options {options!r}")


def _load_source(split: str):
    from datasets import load_dataset

    try:
        return load_dataset(
            "bigbio/med_qa",
            "med_qa_en_4options_source",
            split=split,
        )
    except Exception:
        return load_dataset("GBaker/MedQA-USMLE-4-options", split=split)


def load_medqa(
    split: str = "train",
    max_items: int | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> list[MedQAItem]:
    if verbose:
        print(f"Loading MedQA US English 4-option subset ({split})...")

    ds = _load_source(split)
    schema = _inspect_dataset(ds)
    rng = np.random.default_rng(seed)

    items: list[MedQAItem] = []
    skipped = 0
    lengths: list[int] = []

    for i, row in enumerate(ds):
        if max_items is not None and len(items) >= max_items:
            break
        try:
            row = dict(row)
            question = str(row[schema["question_column"]]).strip()
            if not question:
                skipped += 1
                continue
            options = _extract_options(row, schema)
            if len(options) != 4:
                skipped += 1
                continue
            gold_key = _find_gold_key(row, schema, options)
            non_gold = [key for key in options if key != gold_key]
            distractor_key = str(rng.choice(non_gold))
        except (KeyError, ValueError):
            skipped += 1
            continue

        q_len = len(question)
        lengths.append(q_len)
        items.append(MedQAItem(
            item_id=str(row.get("id", f"medqa_{i}")),
            question_stem=question,
            options=options,
            gold_option_key=gold_key,
            gold_option_text=options[gold_key],
            distractor_key=distractor_key,
            distractor_text=options[distractor_key],
            question_length=q_len,
            y_expert=0.0,
            y_expert_binary=0,
        ))

    if lengths:
        median_len = float(np.median(lengths))
        max_len = float(np.max(lengths))
        for item in items:
            item.y_expert = float(item.question_length) / max_len
            item.y_expert_binary = int(item.question_length >= median_len)

    if verbose:
        print(f"  Items: {len(items)}, skipped: {skipped}")
        if lengths:
            print(f"  Question length: median={np.median(lengths):.0f}, max={np.max(lengths):.0f}")
            print(f"  y_expert_binary=1: {sum(i.y_expert_binary for i in items)} / {len(items)}")
    return items


def format_prompt(question_stem: str, options: dict[str, str], answer_key: str) -> str:
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"Question: {question_stem}\n\nOptions:\n{options_text}\n\nAnswer: {answer_key}"


def build_contrastive_prompts(items: Iterable[MedQAItem]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "question_stem": item.question_stem,
            "options": item.options,
            "a_plus": item.gold_option_key,
            "a_minus": item.distractor_key,
            "a_plus_text": item.gold_option_text,
            "a_minus_text": item.distractor_text,
            "y_expert": item.y_expert,
            "y_expert_binary": item.y_expert_binary,
        }
        for item in items
    ]


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    items = load_medqa(max_items=max_rows, verbose=False)
    return [
        Record(
            example_id=stable_id("medqa", item.item_id, item.question_stem),
            question=format_prompt(item.question_stem, item.options, "").rstrip(),
            correct_answer=item.gold_option_key,
            wrong_answer=item.distractor_key,
            category="MedQA-US-4option",
            expert_reliable=bool(item.y_expert_binary),
            meta=item.to_dict(),
        )
        for item in items
    ]


def _smoke_test(max_items: int = 50) -> int:
    print("=" * 60)
    print("Smoke test: MedQA adapter")
    print("=" * 60)
    items = load_medqa(max_items=max_items, verbose=True)
    if not items:
        raise RuntimeError("MedQA smoke test loaded zero items")
    for item in items[:2]:
        print(f"\n  item_id: {item.item_id}")
        print(f"  question_stem: {item.question_stem[:200]}...")
        for key, value in sorted(item.options.items()):
            marker = " <- gold" if key == item.gold_option_key else (
                " <- distractor" if key == item.distractor_key else ""
            )
            print(f"    {key}. {value[:100]}{marker}")
        print(f"  y_expert: {item.y_expert:.4f}, binary: {item.y_expert_binary}")
    records = load(max_rows=max_items)
    print(f"\nRecord smoke: {len(records)} records")
    print("\n=== Smoke test passed ===")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    if args.smoke:
        sys.exit(_smoke_test(max_items=args.max_items or 50))
    items = load_medqa(split=args.split, max_items=args.max_items, verbose=True)
    print(f"\nLoaded {len(items)} items.")
