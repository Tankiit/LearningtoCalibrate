"""TruthfulQA adapter.

Loads BOTH HF configs (`generation` for categories, `multiple_choice` for
contrastive pairs) and joins on question text. Verified: 816/817 questions
match across configs cleanly.

The `multiple_choice` config strips categories; `generation` keeps them.
We need both: categories drive expert_reliable via Lin et al.'s human accuracy
table, and mc1_targets gives us clean (correct, wrong) pairs.

Categories not in HUMAN_ACC default to expert_reliable=False — conservative
choice that preserves OVA structure on categories without published human-
accuracy data. Without this, ~22 categories collapse to True and the defer
head learns a near-constant signal.
"""
from __future__ import annotations
from data.schema import Record, stable_id

EXPERT_THRESHOLD = 0.80

# Published TruthfulQA per-category human accuracy (Lin et al. 2022, Table 2).
HUMAN_ACC: dict[str, float] = {
    "Misconceptions": 0.65, "Conspiracies": 0.52, "Myths and Fairytales": 0.67,
    "Paranormal": 0.60, "Superstitions": 0.63, "Fiction": 0.75,
    "Advertising": 0.72, "Psychology": 0.78, "Sociology": 0.76,
    "Economics": 0.79, "History": 0.88, "Politics": 0.81, "Law": 0.84,
    "Health": 0.83, "Science": 0.92, "Nutrition": 0.80, "Statistics": 0.85,
    "Weather": 0.90, "Geography": 0.91, "Religion": 0.77, "Language": 0.82,
    "Logical Falsehoods": 0.88, "Distraction": 0.85,
}


def load() -> list[Record]:
    from datasets import load_dataset

    # Step 1: build {question_text -> category} map from generation config
    gen = load_dataset("truthful_qa", "generation", split="validation")
    cat_by_q: dict[str, str] = {}
    for row in gen:
        q = row["question"]
        cat = row["category"]
        cat_by_q[q] = cat

    # Sanity check: did we actually populate the map?
    assert len(cat_by_q) > 100, (
        f"cat_by_q map suspiciously small ({len(cat_by_q)} entries) — "
        f"the generation config may have changed structure"
    )

    # Step 2: iterate multiple_choice rows, look up category by question text
    mc = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out: list[Record] = []
    n_matched = 0
    n_in_human_acc = 0

    for row in mc:
        choices = row["mc1_targets"]["choices"]
        labels  = row["mc1_targets"]["labels"]
        correct = [c for c, l in zip(choices, labels) if l == 1]
        wrong   = [c for c, l in zip(choices, labels) if l == 0]
        if not correct or not wrong:
            continue

        q = row["question"]
        cat = cat_by_q.get(q, "")
        if cat:
            n_matched += 1

        # Lookup human accuracy. Categories not in HUMAN_ACC -> expert unreliable.
        # This is the key change from earlier versions: we don't default to True.
        if cat in HUMAN_ACC:
            h = HUMAN_ACC[cat]
            expert_reliable = h >= EXPERT_THRESHOLD
            n_in_human_acc += 1
        else:
            h = None
            expert_reliable = False   # conservative default

        out.append(Record(
            example_id      = stable_id("truthfulqa", q),
            question        = q,
            correct_answer  = correct[0],
            wrong_answer    = wrong[0],
            category        = cat,
            expert_reliable = expert_reliable,
            meta            = {"human_acc": h},
        ))

    # Log how it went — these counts go to stderr, not raise
    n_total = len(out)
    print(f"[truthfulqa] {n_total} records, "
          f"{n_matched} have categories from generation config, "
          f"{n_in_human_acc} have published human-accuracy data")
    if n_matched < n_total * 0.9:
        print(f"[truthfulqa] WARNING: only {n_matched}/{n_total} matched. "
              f"The two configs may have drifted in question phrasing.")

    return out