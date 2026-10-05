"""Tinker TriviaQA reporter predictions -> aligned per-item arrays (E1/E2).

One row per item (row_id). Fields per (arm, stage, legend): p_correct. Shared:
question_id, split, correct. Arms: q = question only (stage 0), qa = question and
answer (stage 1). Raises if any (arm, stage, legend) misses an item or labels disagree.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

DEFAULT = ("/Users/tanmoy/research/OOD_detection/OVA_ARR/verbalized_conformal_confidence/"
           "LearningtoCalibrate/tinker_uq/runs/full_llama31_8b_qwen3_8b/predictions.jsonl")


def load_tinker(path=DEFAULT, cache=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache is not None and Path(cache).is_file():
        z = np.load(cache, allow_pickle=False)
        if str(z["source_sha256"]) == digest:
            return {k: z[k] for k in z.files}
    rows = {}
    for line in path.open():
        r = json.loads(line)
        item = rows.setdefault(r["row_id"], {"question_id": r["question_id"], "split": r["split"],
                                             "correct": r["correct"]})
        if (item["correct"], item["split"]) != (r["correct"], r["split"]):
            raise ValueError(f"Inconsistent label/split for {r['row_id']}")
        key = f"{r['arm']}_{r['stage']}_{r['legend']}"
        if key in item:
            raise ValueError(f"Duplicate {key} for {r['row_id']}")
        item[key] = r["p_correct"]
    ids = sorted(rows)
    keys = sorted(k for k in rows[ids[0]] if k.count("_") == 2)
    out = {"row_id": np.array(ids), "question_id": np.array([rows[i]["question_id"] for i in ids]),
           "split": np.array([rows[i]["split"] for i in ids]),
           "correct": np.array([rows[i]["correct"] for i in ids], dtype=np.int8),
           "source_sha256": np.array(digest)}
    for k in keys:
        if any(k not in rows[i] for i in ids):
            raise ValueError(f"Missing {k} for some items")
        out["p_" + k] = np.array([rows[i][k] for i in ids], float)
    if len(set(out["question_id"])) != len(ids):
        raise ValueError("Items are not one per question")
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out
