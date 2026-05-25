"""
PubMedQA adapter.

Uses the pqa_labeled subset: the expert-annotated PubMedQA split with
yes/no/maybe decisions. The "maybe" label is treated as high expert
uncertainty.
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

PUBMEDQA_LABELS = ["yes", "no", "maybe"]
PUBMEDQA_LABEL_TO_ID = {label: i for i, label in enumerate(PUBMEDQA_LABELS)}
PUBMEDQA_ID_TO_LABEL = {i: label for i, label in enumerate(PUBMEDQA_LABELS)}
DEFAULT_MAX_ROWS = 1000


@dataclass
class PubMedQAItem:
    item_id: str
    context: str
    question: str
    gold_answer: str
    gold_answer_id: int
    distractor: str
    distractor_id: int
    y_expert: float
    y_expert_binary: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "context": self.context,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "gold_answer_id": self.gold_answer_id,
            "distractor": self.distractor,
            "distractor_id": self.distractor_id,
            "y_expert": self.y_expert,
            "y_expert_binary": self.y_expert_binary,
        }


def _inspect_dataset(ds_split) -> dict[str, Any]:
    cols = list(ds_split.column_names)
    info: dict[str, Any] = {"columns": cols, "n_items": len(ds_split)}

    if "final_decision" in cols:
        info["decision_column"] = "final_decision"
    elif "answer" in cols:
        info["decision_column"] = "answer"
    else:
        raise KeyError(f"No final_decision/answer column. Available: {cols}")

    if "context" in cols:
        info["context_column"] = "context"
        info["context_strategy"] = "dict_with_contexts"
    elif "abstract" in cols:
        info["context_column"] = "abstract"
        info["context_strategy"] = "plain_string"
    else:
        info["context_column"] = None
        info["context_strategy"] = "no_context"
    return info


def _extract_context(row: dict[str, Any], schema: dict[str, Any]) -> str:
    if schema["context_strategy"] == "no_context":
        return ""

    raw = row.get(schema["context_column"])
    if raw is None:
        return ""
    if schema["context_strategy"] == "plain_string":
        return str(raw).strip()

    if isinstance(raw, dict):
        contexts = raw.get("contexts") or raw.get("texts") or []
        if isinstance(contexts, list):
            return " ".join(str(c).strip() for c in contexts if str(c).strip())
        return str(contexts).strip()
    if isinstance(raw, list):
        return " ".join(str(c).strip() for c in raw if str(c).strip())
    return str(raw).strip()


def _normalize_decision(decision: Any) -> str:
    if isinstance(decision, (int, np.integer)):
        if 0 <= int(decision) <= 2:
            return PUBMEDQA_ID_TO_LABEL[int(decision)]
        raise ValueError(f"Decision integer out of range: {decision}")

    if isinstance(decision, str):
        d = decision.strip().lower()
        if d in PUBMEDQA_LABEL_TO_ID:
            return d
        if d == "y":
            return "yes"
        if d == "n":
            return "no"
        if d in {"m", "unsure", "uncertain"}:
            return "maybe"
    raise ValueError(f"Could not normalize decision: {decision!r}")


def _load_source(split: str):
    from datasets import load_dataset

    try:
        return load_dataset(
            "bigbio/pubmed_qa",
            "pubmed_qa_labeled_source",
            split=split,
        )
    except Exception:
        return load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            split="train",
        )


def load_pubmedqa(
    split: str = "train",
    max_items: int | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> list[PubMedQAItem]:
    if verbose:
        print(f"Loading PubMedQA pqa_labeled ({split})...")

    ds = _load_source(split)
    schema = _inspect_dataset(ds)
    rng = np.random.default_rng(seed)

    items: list[PubMedQAItem] = []
    skipped = 0
    counts = {label: 0 for label in PUBMEDQA_LABELS}

    for i, row in enumerate(ds):
        if max_items is not None and len(items) >= max_items:
            break
        try:
            row = dict(row)
            decision = _normalize_decision(row[schema["decision_column"]])
            question = str(row["question"]).strip()
            if not question:
                skipped += 1
                continue
            context = _extract_context(row, schema)
        except (KeyError, ValueError):
            skipped += 1
            continue

        non_gold = [label for label in PUBMEDQA_LABELS if label != decision]
        distractor = str(rng.choice(non_gold))
        y_expert = 1.0 if decision == "maybe" else 0.0
        item_id = str(row.get("pubid") or row.get("id") or f"pubmedqa_{i}")
        items.append(PubMedQAItem(
            item_id=item_id,
            context=context,
            question=question,
            gold_answer=decision,
            gold_answer_id=PUBMEDQA_LABEL_TO_ID[decision],
            distractor=distractor,
            distractor_id=PUBMEDQA_LABEL_TO_ID[distractor],
            y_expert=y_expert,
            y_expert_binary=int(y_expert),
        ))
        counts[decision] += 1

    if verbose:
        print(f"  Items: {len(items)}, skipped: {skipped}")
        for label, count in counts.items():
            pct = 100.0 * count / max(len(items), 1)
            print(f"  {label:5s}: {count:4d} ({pct:.1f}%)")
    return items


def format_prompt(context: str, question: str, answer: str) -> str:
    if context:
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    return f"Question: {question}\n\nAnswer: {answer}"


def build_contrastive_prompts(items: Iterable[PubMedQAItem]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "context": item.context,
            "question": item.question,
            "a_plus": item.gold_answer,
            "a_minus": item.distractor,
            "y_expert": item.y_expert,
            "y_expert_binary": item.y_expert_binary,
        }
        for item in items
    ]


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    items = load_pubmedqa(max_items=max_rows, verbose=False)
    return [
        Record(
            example_id=stable_id("pubmedqa", item.item_id, item.question),
            question=format_prompt(item.context, item.question, "").rstrip(),
            correct_answer=item.gold_answer,
            wrong_answer=item.distractor,
            category=f"PubMedQA-{item.gold_answer}",
            expert_reliable=bool(item.y_expert_binary),
            meta=item.to_dict(),
        )
        for item in items
    ]


def _smoke_test(max_items: int = 50) -> int:
    print("=" * 60)
    print("Smoke test: PubMedQA adapter")
    print("=" * 60)
    items = load_pubmedqa(max_items=max_items, verbose=True)
    if not items:
        raise RuntimeError("PubMedQA smoke test loaded zero items")
    for item in items[:3]:
        print(f"\n  item_id: {item.item_id}")
        print(f"  question: {item.question[:120]}")
        print(f"  gold: {item.gold_answer}, distractor: {item.distractor}")
        print(f"  y_expert: {item.y_expert}, binary: {item.y_expert_binary}")
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
    items = load_pubmedqa(split=args.split, max_items=args.max_items, verbose=True)
    print(f"\nLoaded {len(items)} items.")
