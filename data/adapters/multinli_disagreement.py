"""MultiNLI-Disagreement adapter for held-out regime validation.

Primary source is Hugging Face `metaeval/nli-disagreement-tasks`, config
`multinli`. The adapter is intentionally schema-flexible because mirrors of
this dataset use slightly different field names for the annotator labels.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from data.common_dataloader import first_text
from data.schema import Record, stable_id

HF_PATH = "metaeval/nli-disagreement-tasks"
HF_NAME = "multinli"
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_ROWS = 3000

LABEL_TEXT = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction",
}
TEXT_TO_SHORT = {"entailment": "e", "neutral": "n", "contradiction": "c"}


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if value in LABEL_TEXT:
        return LABEL_TEXT[value]
    text = str(value).strip().lower()
    return LABEL_TEXT.get(text)


def _premise(row: dict[str, Any]) -> str:
    return (
        first_text(row.get("premise"))
        or first_text(row.get("sentence1"))
        or first_text(row.get("context"))
    )


def _hypothesis(row: dict[str, Any]) -> str:
    return (
        first_text(row.get("hypothesis"))
        or first_text(row.get("sentence2"))
        or first_text(row.get("target"))
    )


def _annotation_values(row: dict[str, Any]) -> list[str]:
    for key in (
        "annotations",
        "annotator_labels",
        "labels",
        "label_list",
        "votes",
        "gold_label_list",
    ):
        raw = row.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict):
            for subkey in ("label", "labels", "annotations"):
                if subkey in raw:
                    raw = raw[subkey]
                    break
        if not isinstance(raw, (list, tuple, np.ndarray)):
            raw = [raw]
        labels = [_normalize_label(x) for x in raw]
        labels = [x for x in labels if x is not None]
        if labels:
            return labels

    label = _normalize_label(row.get("label") or row.get("gold_label") or row.get("majority_label"))
    return [label] if label else []


def _entropy_from_counts(counts: Counter[str]) -> float:
    values = np.asarray(list(counts.values()), dtype=float)
    probs = values / values.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum() / np.log(3.0))


def _record_from_row(row: dict[str, Any]) -> Record | None:
    premise = _premise(row)
    hypothesis = _hypothesis(row)
    if not premise or not hypothesis:
        return None

    annotations = _annotation_values(row)
    if not annotations:
        return None

    counts = Counter(annotations)
    majority = counts.most_common(1)[0][0]
    majority_short = TEXT_TO_SHORT[majority]
    wrong_candidates = [x for x in ("entailment", "neutral", "contradiction") if x != majority]
    wrong = min(wrong_candidates, key=lambda x: counts.get(x, 0))
    entropy = _entropy_from_counts(counts)
    uid = first_text(row.get("uid")) or first_text(row.get("id")) or f"{premise}::{hypothesis}"

    question = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "What is the NLI relation? Answer with entailment, neutral, or contradiction."
    )
    return Record(
        example_id=stable_id("multinli_disagreement", uid),
        question=question,
        correct_answer=majority,
        wrong_answer=wrong,
        category="MultiNLI-Disagreement",
        expert_reliable=entropy < 0.5,
        meta={
            "uid": uid,
            "annotations": annotations,
            "label_counter": {TEXT_TO_SHORT[k]: int(v) for k, v in counts.items()},
            "majority_label": majority_short,
            "annotator_entropy_norm": entropy,
        },
    )


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset(HF_PATH, HF_NAME, split=DEFAULT_SPLIT)
    out: list[Record] = []
    for row in ds:
        record = _record_from_row(dict(row))
        if record is not None:
            out.append(record)
        if len(out) >= max_rows:
            break
    return out
