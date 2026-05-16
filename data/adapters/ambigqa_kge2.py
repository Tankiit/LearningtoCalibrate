"""AmbigQA k>=2 adapter.

Uses the official Hugging Face dataset ``sewon/ambig_qa``. We keep examples
with at least two plausible answer clusters/questions. The expert signal is
``expert_reliable=False`` for these rows because the original question is
ambiguous by construction: a single default answer is not reliably sufficient.

Mapping:
  question        <- original ambiguous question
  correct_answer  <- first answer from the first QA pair / answer cluster
  wrong_answer    <- first answer from a different QA pair / answer cluster
  category        <- AmbigQA-k{number of plausible answers}
"""
from __future__ import annotations

from typing import Any

from data.common_dataloader import HFSource, first_text, load_hf_source
from data.schema import Record, stable_id

DEFAULT_MAX_ROWS = 8000


def _annotation_items(annotations: Any) -> list[dict[str, Any]]:
    """Normalize HF sequence-of-structs and struct-of-sequences variants."""
    if isinstance(annotations, list):
        return [a for a in annotations if isinstance(a, dict)]
    if not isinstance(annotations, dict):
        return []

    types = annotations.get("type") or []
    answers = annotations.get("answer") or []
    qa_pairs = annotations.get("qaPairs") or []
    n = max(len(types), len(answers), len(qa_pairs))
    items = []
    for i in range(n):
        items.append({
            "type": types[i] if i < len(types) else "",
            "answer": answers[i] if i < len(answers) else [],
            "qaPairs": qa_pairs[i] if i < len(qa_pairs) else [],
        })
    return items


def _answers_from_row(row: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for ann in _annotation_items(row.get("annotations")):
        ann_type = ann.get("type")
        if ann_type == "singleAnswer":
            text = first_text(ann.get("answer"))
            if text:
                answers.append(text)
        else:
            for pair in ann.get("qaPairs") or []:
                text = first_text(pair.get("answer") if isinstance(pair, dict) else pair)
                if text:
                    answers.append(text)

    deduped = []
    seen = set()
    for answer in answers:
        key = answer.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(answer)
    return deduped


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    ds = load_hf_source(HFSource("sewon/ambig_qa", name="light", split="validation"))

    out: list[Record] = []
    for row in ds:
        q = first_text(row.get("question"))
        answers = _answers_from_row(dict(row))
        if not q or len(answers) < 2:
            continue

        out.append(Record(
            example_id=stable_id("ambigqa_kge2", str(row.get("id") or q)),
            question=q,
            correct_answer=answers[0],
            wrong_answer=answers[1],
            category=f"AmbigQA-k{len(answers)}",
            expert_reliable=False,
            meta={
                "ambigqa_id": row.get("id"),
                "n_answers": len(answers),
                "answers": answers,
            },
        ))
        if len(out) >= max_rows:
            break
    return out

