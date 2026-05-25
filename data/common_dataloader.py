"""Common dataset-source helpers for Record-producing adapters.

The project-level adapter contract is intentionally narrower than a generic
PyTorch dataloader: adapters must return ``list[data.schema.Record]`` for the
LLM extraction pipeline. This module centralizes the repeated Hugging Face /
JSONL loading and small normalization utilities while leaving each dataset's
scientific mapping explicit in its adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from data.schema import Record, stable_id


@dataclass(frozen=True)
class HFSource:
    path: str
    name: str | None = None
    split: str = "train"
    trust_remote_code: bool = True


@dataclass(frozen=True)
class JsonlSource:
    url: str
    split: str = "train"


def load_hf_source(source: HFSource):
    """Load a Hugging Face dataset split from a small declarative spec."""
    from datasets import load_dataset

    if source.name:
        return load_dataset(source.path, name=source.name, split=source.split)
    return load_dataset(source.path, split=source.split)


def load_jsonl_source(source: JsonlSource):
    """Load a JSONL URL through Hugging Face datasets for caching/Modal reuse."""
    from datasets import load_dataset

    return load_dataset("json", data_files={source.split: source.url}, split=source.split)


def load_local_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load local JSONL into a list of dictionaries."""
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_text(value: Any) -> str:
    """Return the first non-empty string inside common scalar/list shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            text = first_text(item)
            if text:
                return text
    return str(value).strip()


def shifted_distractors(answers: list[str]) -> list[str]:
    """Construct deterministic wrong answers by shifting the answer column."""
    if not answers:
        return []
    if len(answers) == 1:
        return [answers[0]]
    return answers[1:] + [answers[0]]


def records_from_question_answers(
    *,
    dataset: str,
    rows: Iterable[dict[str, Any]],
    question_key: str,
    answer_key: str,
    category: str,
    expert_reliable: bool = True,
    max_rows: int | None = None,
) -> list[Record]:
    """Build Records for QA datasets that need shift-by-one distractors."""
    raw: list[tuple[str, str, dict[str, Any]]] = []
    for row in rows:
        q = first_text(row.get(question_key))
        a = first_text(row.get(answer_key))
        if q and a:
            raw.append((q, a, dict(row)))
        if max_rows is not None and len(raw) >= max_rows:
            break

    wrong = shifted_distractors([a for _, a, _ in raw])
    return [
        Record(
            example_id=stable_id(dataset, q),
            question=q,
            correct_answer=a,
            wrong_answer=w,
            category=category,
            expert_reliable=expert_reliable,
            meta={"source_row": row},
        )
        for (q, a, row), w in zip(raw, wrong)
        if a != w
    ]
