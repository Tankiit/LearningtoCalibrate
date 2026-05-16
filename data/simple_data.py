"""
simple_data.py — one file, all five datasets, drop-in usable.

Usage:
    from data.simple_data import load
    records = load("truthfulqa")
    print(records[0])

Yields the same Record schema the rest of the repo uses, so output is
compatible with `data/registry.py:load_dataset`. The difference is that
this file is self-contained — read it top-to-bottom, no jumping between
adapter files.

When to use this over data/registry.py:
  - Quick exploration in a notebook
  - Debugging an adapter's output without running the registry
  - Porting your existing datasets_loader.py code (this is the lift)

When to use data/registry.py instead:
  - Production sweeps (registry validates schema, this doesn't)
  - Adding a new dataset (registry's plug-in pattern is cleaner)
"""
from __future__ import annotations
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── Record (mirrors data/schema.py so output is interchangeable) ───────────
@dataclass
class Record:
    example_id: str
    question: str
    correct_answer: str
    wrong_answer: str
    category: str = ""
    expert_reliable: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def _sid(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]


# ─── TruthfulQA: per-category expert reliability ────────────────────────────
_TRUTHFULQA_HUMAN_ACC = {
    "Misconceptions": 0.65, "Conspiracies": 0.52, "Myths and Fairytales": 0.67,
    "Paranormal": 0.60, "Superstitions": 0.63, "Fiction": 0.75,
    "Advertising": 0.72, "Psychology": 0.78, "Sociology": 0.76,
    "Economics": 0.79, "History": 0.88, "Politics": 0.81, "Law": 0.84,
    "Health": 0.83, "Science": 0.92, "Nutrition": 0.80, "Statistics": 0.85,
    "Weather": 0.90, "Geography": 0.91, "Religion": 0.77, "Language": 0.82,
    "Logical Falsehoods": 0.88, "Distraction": 0.85,
}
_EXPERT_THRESHOLD = 0.80


def _load_truthfulqa() -> list[Record]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for row in ds:
        choices, labels = row["mc1_targets"]["choices"], row["mc1_targets"]["labels"]
        correct = [c for c, l in zip(choices, labels) if l == 1]
        wrong   = [c for c, l in zip(choices, labels) if l == 0]
        if not correct or not wrong:
            continue
        cat = row.get("category", "")
        h = _TRUTHFULQA_HUMAN_ACC.get(cat, _EXPERT_THRESHOLD)
        out.append(Record(
            example_id      = _sid("truthfulqa", row["question"]),
            question        = row["question"],
            correct_answer  = correct[0],
            wrong_answer    = wrong[0],
            category        = cat,
            expert_reliable = h >= _EXPERT_THRESHOLD,
            meta            = {"human_acc": h},
        ))
    return out


# ─── HaluEval QA: GPT-generated hallucinations ──────────────────────────────
def _load_halueval_qa(max_rows: int = 8000) -> list[Record]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))
    out = []
    for row in ds:
        q     = (row.get("question") or "").strip()
        right = (row.get("right_answer") or "").strip()
        wrong = (row.get("hallucinated_answer") or "").strip()
        if not (q and right and wrong):
            continue
        out.append(Record(
            example_id      = _sid("halueval_qa", q),
            question        = q,
            correct_answer  = right,
            wrong_answer    = wrong,
            category        = "HaluEval-QA",
            expert_reliable = True,
        ))
    return out


# ─── TriviaQA: shift-by-1 distractor ────────────────────────────────────────
def _load_triviaqa(max_rows: int = 8000) -> list[Record]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc", split="validation")
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))
    raw = []
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
            example_id      = _sid("triviaqa", q),
            question        = q,
            correct_answer  = c,
            wrong_answer    = w,
            category        = "TriviaQA",
            expert_reliable = True,
        )
        for (q, c), w in zip(raw, shifted)
    ]


# ─── PopQA: popularity-driven expert reliability ────────────────────────────
def _load_popqa(max_rows: int = 8000, view_threshold: int = 1000) -> list[Record]:
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))
    raw = []
    for row in ds:
        q = (row.get("question") or "").strip()
        answers = row.get("possible_answers", [])
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except Exception:
                answers = [answers]
        if not q or not answers or not answers[0]:
            continue
        pop = max(int(row.get("s_pop") or 0), int(row.get("o_pop") or 0))
        raw.append({
            "q": q, "correct": answers[0], "pop": pop,
            "prop": row.get("prop", ""), "subj": row.get("subj", ""),
        })
    if not raw:
        return []
    correct = [r["correct"] for r in raw]
    shifted = correct[1:] + [correct[0]]
    return [
        Record(
            example_id      = _sid("popqa", r["q"]),
            question        = r["q"],
            correct_answer  = r["correct"],
            wrong_answer    = w,
            category        = f"PopQA-{r['prop']}",
            expert_reliable = r["pop"] >= view_threshold,
            meta            = {"page_views": r["pop"], "prop": r["prop"], "subj": r["subj"]},
        )
        for r, w in zip(raw, shifted)
    ]


# ─── BioASQ: yesno (opposite) + factoid (shift-by-1), three-tier load ───────
def _load_bioasq(max_rows: int = 2500) -> list[Record]:
    items = _bioasq_raw()
    yesno, factoid_q, factoid_a = [], [], []
    for it in items:
        q_type = str(it.get("type", "")).lower()
        q = str(it.get("question") or "").strip()
        if not q:
            continue
        exact = it.get("exact_answer") or []
        if isinstance(exact, list) and exact:
            ans = exact[0]
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            correct = str(ans).strip()
        else:
            correct = ""
        if not correct:
            continue
        if q_type == "yesno" and correct.lower() in ("yes", "no"):
            wrong = "no" if correct.lower() == "yes" else "yes"
            yesno.append(Record(
                example_id      = _sid("bioasq", q),
                question        = q,
                correct_answer  = correct,
                wrong_answer    = wrong,
                category        = "BioASQ-YesNo",
                expert_reliable = True,
            ))
        elif q_type == "factoid":
            factoid_q.append(q)
            factoid_a.append(correct)

    factoid_records = []
    if factoid_a:
        shifted = factoid_a[1:] + [factoid_a[0]]
        factoid_records = [
            Record(
                example_id      = _sid("bioasq", q),
                question        = q,
                correct_answer  = c,
                wrong_answer    = w,
                category        = "BioASQ-Factoid",
                expert_reliable = True,
            )
            for q, c, w in zip(factoid_q, factoid_a, shifted)
        ]

    combined = yesno + factoid_records
    random.seed(42)
    random.shuffle(combined)
    return combined[:max_rows]


def _bioasq_raw() -> list[dict]:
    """Three-tier fallback: parquet HF → direct URL → local JSON."""
    from datasets import load_dataset
    for hf_id, split in [
        ("kroshan/bioasq_task_b", "train"),
        ("enoriega-info/bioasq11b", "train"),
        ("enoriega-info/bioasq11b", "test"),
    ]:
        try:
            return list(load_dataset(hf_id, split=split))
        except Exception:
            continue
    try:
        import pandas as pd
        df = pd.read_parquet(
            "https://huggingface.co/datasets/kroshan/bioasq_task_b/"
            "resolve/main/data/train-00000-of-00001.parquet"
        )
        return df.to_dict("records")
    except Exception:
        pass
    for p in [Path("/data/bioasq/training11b.json"), Path("./data/bioasq.json")]:
        if p.exists():
            with p.open() as f:
                return (json.load(f).get("questions") or [])
    raise RuntimeError("BioASQ: all loading strategies failed")


# ─── Dispatch ───────────────────────────────────────────────────────────────
_LOADERS = {
    "truthfulqa":  _load_truthfulqa,
    "halueval_qa": _load_halueval_qa,
    "triviaqa":    _load_triviaqa,
    "popqa":       _load_popqa,
    "bioasq":      _load_bioasq,
}


def load(dataset: str) -> list[Record]:
    """Load one dataset by name. Returns a list of Record objects."""
    if dataset not in _LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {list(_LOADERS)}")
    return _LOADERS[dataset]()


def list_datasets() -> list[str]:
    return list(_LOADERS.keys())


# ─── Smoke test when run as a script ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "truthfulqa"
    records = load(name)
    print(f"Loaded {len(records)} records from {name}")
    if records:
        r = records[0]
        print(f"  example_id:      {r.example_id}")
        print(f"  question:        {r.question[:80]}")
        print(f"  correct_answer:  {r.correct_answer[:80]}")
        print(f"  wrong_answer:    {r.wrong_answer[:80]}")
        print(f"  category:        {r.category}")
        print(f"  expert_reliable: {r.expert_reliable}")
        print(f"  meta:            {r.meta}")
