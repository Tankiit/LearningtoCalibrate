"""PopQA adapter.

PopQA pairs each (subject, predicate, object) triple with subject and
object Wikipedia page-view counts. We use page views as a proxy for
"how reliable is a domain expert here" — popular entities are well-known
and an expert is reliable; rare entities (long-tail) are likely outside
expert knowledge.

  expert_reliable = max(s_pop, o_pop) >= VIEW_THRESHOLD

Wrong answers are a shift-by-1 over the correct-answer column (same trick
as TriviaQA — PopQA doesn't provide natural distractors either).
"""
from __future__ import annotations
import json

from data.schema import Record, stable_id

DEFAULT_MAX_ROWS       = 8000
DEFAULT_VIEW_THRESHOLD = 1000


def load(
    max_rows: int = DEFAULT_MAX_ROWS,
    view_threshold: int = DEFAULT_VIEW_THRESHOLD,
) -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    raw: list[dict] = []
    seen_qs: set[str] = set()
    for row in ds:
        q = (row.get("question") or "").strip()
        if q in seen_qs:
            continue          # dataset has duplicate rows; keep first only
        seen_qs.add(q)
        answers = row.get("possible_answers", [])
        # `possible_answers` ships either as a list or a JSON-encoded string.
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except Exception:
                answers = [answers]
        if not q or not answers or not answers[0]:
            continue
        pop = max(int(row.get("s_pop") or 0), int(row.get("o_pop") or 0))
        raw.append({
            "q":       q,
            "correct": answers[0],
            "answers": [str(a).strip() for a in answers if str(a).strip()],
            "pop":     pop,
            "prop":    row.get("prop", ""),
            "subj":    row.get("subj", ""),
        })
    if not raw:
        return []

    correct = [r["correct"] for r in raw]
    shifted = correct[1:] + [correct[0]]
    return [
        Record(
            example_id      = stable_id("popqa", r["q"]),
            question        = r["q"],
            correct_answer  = r["correct"],
            wrong_answer    = w,
            category        = f"PopQA-{r['prop']}",
            expert_reliable = r["pop"] >= view_threshold,
            meta            = {
                "page_views": r["pop"],
                "prop":       r["prop"],
                "subj":       r["subj"],
                "possible_answers": r["answers"],
            },
        )
        for r, w in zip(raw, shifted)
    ]
