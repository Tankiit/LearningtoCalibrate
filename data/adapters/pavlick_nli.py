"""Pavlick/Kwiatkowski NLI variation adapter.

Uses the local NLI-variation-data repository by default:
``/Users/tanmoy/research/data/NLI-variation-data``.

The sentence-pair analysis file contains redundant scalar human judgments in
[-50, 50], where positive values indicate entailment-like judgments and negative
values indicate non-entailment/contradiction-like judgments. We convert the sign
of the mean judgment into a two-answer NLI prompt and use annotator agreement as
the expert-reliability signal.
"""
from __future__ import annotations

import os
from pathlib import Path

from data.common_dataloader import load_local_jsonl
from data.schema import Record, stable_id

DEFAULT_ROOT = Path("/Users/tanmoy/research/data/NLI-variation-data")
MODAL_ROOT = Path("/root/local_data/NLI-variation-data")
ROOT_ENV = "OVA_ARR_PAVLICK_NLI_DIR"
DEFAULT_MAX_ROWS = 8000
AGREEMENT_THRESHOLD = 0.80


def _source_file() -> Path:
    root = Path(os.environ.get(ROOT_ENV, ""))
    if not root:
        root = MODAL_ROOT if MODAL_ROOT.exists() else DEFAULT_ROOT
    return root / "sentence-pair-analysis" / "preprocessed-data.jsonl"


def _agreement(labels: list[float]) -> float:
    if not labels:
        return 0.0
    n_pos = sum(1 for x in labels if float(x) > 0)
    n_neg = sum(1 for x in labels if float(x) < 0)
    return max(n_pos, n_neg) / len(labels)


def _answers(labels: list[float]) -> tuple[str, str]:
    mean_label = sum(float(x) for x in labels) / len(labels)
    if mean_label >= 0:
        return "entailment", "contradiction"
    return "contradiction", "entailment"


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    path = _source_file()
    if not path.exists():
        raise FileNotFoundError(
            f"Pavlick NLI data not found at {path}. Set {ROOT_ENV} to the "
            "NLI-variation-data checkout."
        )

    out: list[Record] = []
    for row in load_local_jsonl(path):
        premise = (row.get("premise") or "").strip()
        hypothesis = (row.get("hypothesis") or "").strip()
        labels = [float(x) for x in row.get("labels") or []]
        if not premise or not hypothesis or not labels:
            continue

        correct, wrong = _answers(labels)
        agreement = _agreement(labels)
        uid = str(row.get("id") or f"{premise}::{hypothesis}")
        question = (
            "Premise: "
            f"{premise}\n"
            "Hypothesis: "
            f"{hypothesis}\n"
            "Does the premise entail the hypothesis? Answer with entailment "
            "or contradiction."
        )

        out.append(Record(
            example_id=stable_id("pavlick_nli", uid),
            question=question,
            correct_answer=correct,
            wrong_answer=wrong,
            category=f"PavlickNLI-{row.get('task', 'unknown')}",
            expert_reliable=agreement >= AGREEMENT_THRESHOLD,
            meta={
                "uid": uid,
                "task": row.get("task"),
                "original_dataset_label": row.get("original-dataset-label"),
                "mean_label": sum(labels) / len(labels),
                "agreement": agreement,
                "num_na": row.get("num-NA"),
                "n_labels": len(labels),
            },
        ))
        if len(out) >= max_rows:
            break
    return out
