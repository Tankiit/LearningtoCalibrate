"""TriviaQA adapter.

TriviaQA only gives correct answers (no natural distractors), so we
construct a^- by shifting the correct-answer column by one position.
This is a deliberately weak distractor — the gap signal still works
because Δ measures DIFFERENCE between two reps, not their absolute
quality.

`expert_reliable=True` for every record: TriviaQA is encyclopedic
fact recall where a domain expert is reliable by construction.
"""
from __future__ import annotations

from data.schema import Record, stable_id

DEFAULT_MAX_ROWS = 8000


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc", split="validation")
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    raw: list[tuple[str, str]] = []
    for row in ds:
        q       = (row.get("question") or "").strip()
        aliases = row.get("answer", {}).get("aliases", [])
        if not q or not aliases:
            continue
        raw.append((q, aliases[0]))
    if not raw:
        return []

    correct = [c for _, c in raw]
    shifted = correct[1:] + [correct[0]]
    return [
        Record(
            example_id      = stable_id("triviaqa", q),
            question        = q,
            correct_answer  = c,
            wrong_answer    = w,
            category        = "TriviaQA",
            expert_reliable = True,
        )
        for (q, c), w in zip(raw, shifted)
    ]
