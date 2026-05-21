"""BioASQ adapter.

Two question types are usable for the contrastive setup:
  yesno    →  natural distractor (yes ↔ no)
  factoid  →  shift-by-1 distractor over the correct-answer column

Other types (list, summary) are dropped because they don't admit a clean
binary contrast.

Loading is three-tier because BioASQ doesn't have a single canonical HF
release: parquet HF datasets → direct parquet URL → local JSON dump.
`load_raw()` is exposed for debugging which tier won.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from data.schema import Record, stable_id

DEFAULT_MAX_ROWS = 2500
SHUFFLE_SEED     = 42


def load(max_rows: int = DEFAULT_MAX_ROWS) -> list[Record]:
    items = load_raw()

    yesno: list[Record]   = []
    factoid_q: list[str]  = []
    factoid_a: list[str]  = []

    for it in items:
        q_type = str(it.get("type", "")).lower()
        q      = str(it.get("question") or "").strip()
        if not q:
            continue

        # `exact_answer` may be [["yes"]] or [["Apoptosis"]] depending on
        # task variant — flatten one level if needed.
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
                example_id      = stable_id("bioasq", q),
                question        = q,
                correct_answer  = correct,
                wrong_answer    = wrong,
                category        = "BioASQ-YesNo",
                expert_reliable = True,
            ))
        elif q_type == "factoid":
            factoid_q.append(q)
            factoid_a.append(correct)

    factoid_records: list[Record] = []
    if factoid_a:
        shifted = factoid_a[1:] + [factoid_a[0]]
        factoid_records = [
            Record(
                example_id      = stable_id("bioasq", q),
                question        = q,
                correct_answer  = c,
                wrong_answer    = w,
                category        = "BioASQ-Factoid",
                expert_reliable = True,
            )
            for q, c, w in zip(factoid_q, factoid_a, shifted)
        ]

    combined = yesno + factoid_records
    random.Random(SHUFFLE_SEED).shuffle(combined)
    return combined[:max_rows]


def load_raw() -> list[dict]:
    """Three-tier fallback: parquet HF → direct URL → local JSON."""
    from datasets import load_dataset

    for hf_id, split in [
        ("kroshan/bioasq_task_b",  "train"),
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
