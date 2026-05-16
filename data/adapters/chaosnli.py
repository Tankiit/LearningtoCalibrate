"""ChaosNLI adapter.

Uses the official ChaosNLI JSONL files from the project repository through the
common JSONL loader. Each row has 100 human NLI labels. We map the majority
label to the positive answer and a minority label to the contrastive answer.

``expert_reliable`` is true when the human majority is strong enough to treat
the NLI label as unambiguous; otherwise it is false. This gives the defer head
per-example ambiguity structure rather than a constant expert label.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from data.common_dataloader import (
    JsonlSource,
    first_text,
    load_jsonl_source,
    load_local_jsonl,
)
from data.schema import Record, stable_id

DEFAULT_MAX_ROWS = 8000
AGREEMENT_THRESHOLD = 0.80

LABEL_TEXT = {
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction",
    0: "entailment",
    1: "neutral",
    2: "contradiction",
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
}

SOURCES = [
    JsonlSource(
        "https://raw.githubusercontent.com/easonnie/ChaosNLI/master/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl",
        split="mnli_m",
    ),
    JsonlSource(
        "https://raw.githubusercontent.com/easonnie/ChaosNLI/master/data/chaosNLI_v1.0/chaosNLI_snli.jsonl",
        split="snli",
    ),
]

LOCAL_DIR_ENV = "OVA_ARR_CHAOSNLI_DIR"
DEFAULT_LOCAL_DIR = Path("/Users/tanmoy/research/data/chaosNLI_v1.0")
MODAL_LOCAL_DIR = Path("/root/local_data/chaosNLI_v1.0")
LOCAL_FILES = [
    ("mnli_m", "chaosNLI_mnli_m.jsonl"),
    ("snli", "chaosNLI_snli.jsonl"),
]


def _nested_get(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _premise(row: dict[str, Any]) -> str:
    return (
        first_text(row.get("premise"))
        or first_text(row.get("sentence1"))
        or first_text(_nested_get(row, "example", "premise"))
        or first_text(_nested_get(row, "example", "sentence1"))
    )


def _hypothesis(row: dict[str, Any]) -> str:
    return (
        first_text(row.get("hypothesis"))
        or first_text(row.get("sentence2"))
        or first_text(_nested_get(row, "example", "hypothesis"))
        or first_text(_nested_get(row, "example", "sentence2"))
    )


def _label_counts(row: dict[str, Any]) -> dict[str, int]:
    raw = row.get("label_counter") or row.get("label_counts") or {}
    out = {"e": 0, "n": 0, "c": 0}
    if isinstance(raw, dict):
        for key, value in raw.items():
            label = str(key).lower()[0]
            if label in out:
                out[label] = int(value)
    return out


def _majority_label(row: dict[str, Any], counts: dict[str, int]) -> str:
    raw = row.get("majority_label") or row.get("label")
    if raw in LABEL_TEXT:
        text = LABEL_TEXT[raw]
        return {"entailment": "e", "neutral": "n", "contradiction": "c"}[text]
    return max(counts, key=counts.get)


def _wrong_label(majority: str, counts: dict[str, int]) -> str:
    candidates = [k for k in ("e", "n", "c") if k != majority]
    return max(candidates, key=lambda k: counts.get(k, 0))


def _record_from_row(row: dict[str, Any], source_name: str) -> Record | None:
    premise = _premise(row)
    hypothesis = _hypothesis(row)
    if not premise or not hypothesis:
        return None

    counts = _label_counts(row)
    total = sum(counts.values())
    if total <= 0:
        return None

    maj = _majority_label(row, counts)
    wrong = _wrong_label(maj, counts)
    agreement = counts[maj] / total
    uid = first_text(row.get("uid")) or first_text(row.get("id")) or f"{premise}::{hypothesis}"

    question = (
        "Premise: "
        f"{premise}\n"
        "Hypothesis: "
        f"{hypothesis}\n"
        "What is the NLI relation? Answer with entailment, neutral, or contradiction."
    )
    return Record(
        example_id=stable_id("chaosnli", source_name, uid),
        question=question,
        correct_answer=LABEL_TEXT[maj],
        wrong_answer=LABEL_TEXT[wrong],
        category=f"ChaosNLI-{source_name}",
        expert_reliable=agreement >= AGREEMENT_THRESHOLD,
        meta={
            "uid": uid,
            "source": source_name,
            "label_counter": counts,
            "majority_label": maj,
            "agreement": agreement,
        },
    )


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    out: list[Record] = []
    local_dir = Path(os.environ.get(LOCAL_DIR_ENV, ""))
    if not local_dir:
        local_dir = MODAL_LOCAL_DIR if MODAL_LOCAL_DIR.exists() else DEFAULT_LOCAL_DIR
    if local_dir.exists():
        for source_name, fname in LOCAL_FILES:
            path = local_dir / fname
            if not path.exists():
                continue
            for row in load_local_jsonl(path):
                record = _record_from_row(row, source_name)
                if record is not None:
                    out.append(record)
                if len(out) >= max_rows:
                    return out
        if out:
            return out

    for source in SOURCES:
        ds = load_jsonl_source(source)
        for row in ds:
            record = _record_from_row(dict(row), source.split)
            if record is not None:
                out.append(record)
            if len(out) >= max_rows:
                return out
    return out
