"""rsuq.qa_data — QA instance loaders for E-C/E-D.

Produces a uniform instance schema regardless of dataset:
    {"question": str, "response": str, "label": int,  # 1 = hallucinated
     "answer_char_span": (start, end)}                # span within q+" "+resp

Two datasets, OPPOSITE label polarities (the classic bug, handled here):

  HaluEval  each item has BOTH right_answer and hallucinated_answer ->
            two balanced instances per item (label 0 and 1). Span = the
            appended answer tail. Cleanest substrate for the screen because
            it is balanced and the labels are gold, not generation-derived.

  TriviaQA  gold answer + aliases; label by exact-match of a response against
            the alias set. For the width experiment we append the GOLD answer
            (teacher-forced scoring) and pair it with a DISTRACTOR answer
            sampled from another question, so each item again yields a
            balanced 0/1 pair without needing a generation pass.
"""

from __future__ import annotations
import random


def _appended_span(question: str, response: str) -> tuple[int, int]:
    start = len(question) + 1                       # +1 for the joining space
    return (start, start + len(response))


def load_halueval(n: int, task: str = "qa", seed: int = 0):
    """HaluEval QA split. Requires `datasets`."""
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", task, split="data")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        item = ds[i]
        q = item["question"].strip()
        for key, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            resp = item[key].strip()
            out.append({"question": q, "response": resp, "label": lab,
                        "answer_char_span": _appended_span(q, resp)})
    return out


def load_triviaqa(n: int, seed: int = 0):
    """TriviaQA (rc.nocontext). Balanced via distractor pairing."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pool = idx[:max(n, 200)]
    out = []
    for k, i in enumerate(pool):
        if len(out) >= n:
            break
        item = ds[i]
        q = item["question"].strip()
        gold = item["answer"]["value"].strip()
        # distractor: gold answer of a different question
        j = pool[(k + 1) % len(pool)]
        distractor = ds[j]["answer"]["value"].strip()
        aliases = set(a.lower() for a in item["answer"].get("aliases", [gold]))
        if distractor.lower() in aliases:           # avoid accidental match
            continue
        for resp, lab in ((gold, 0), (distractor, 1)):
            out.append({"question": q, "response": resp, "label": lab,
                        "answer_char_span": _appended_span(q, resp)})
    return out


LOADERS = {"halueval": load_halueval, "triviaqa": load_triviaqa}


def load_qa_instances(dataset: str, n: int, seed: int = 0):
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; have {list(LOADERS)}")
    return LOADERS[dataset](n, seed=seed)
