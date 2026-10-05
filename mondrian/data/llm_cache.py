"""Strict optional cache schema; no implicit pooling/basis conversion."""
from pathlib import Path
import json
import numpy as np


def load_cache(path,model,dataset,layer_pool):
    root=Path(path)
    meta=json.loads((root/"meta.json").read_text())
    expected={"schema_version":1,"model":model,"dataset":dataset,"layer_pool":layer_pool,
              "representation":"residual_stream"}
    for key,value in expected.items():
        if meta.get(key)!=value:
            raise ValueError(f"Cache metadata mismatch for {key}")
    if not meta.get("basis_id"):
        raise ValueError("A single explicit residual-stream basis_id is required")
    with np.load(root/"cache.npz",allow_pickle=False) as z:
        X,y,qid=z["X"].copy(),z["correct"].copy(),z["question_id"].astype(str)
    if X.ndim!=2 or len(X)!=len(y) or y.shape!=qid.shape or y.ndim!=1:
        raise ValueError("Expected aligned X, correct, question_id")
    if not np.isfinite(X).all() or not np.isin(y,[0,1]).all() or np.any(qid==""):
        raise ValueError("Invalid states, correctness labels, or question ids")
    return X,y,qid


def one_row_per_question(X,y,qid,seed=0):
    """Choose a row without labels, preserving question-level iid unit.

    Splitting by question alone does not make repeated rows iid calibration
    observations. This adapter uses one randomly selected variant per question.
    """
    rng=np.random.default_rng(seed)
    idx=np.array([rng.choice(np.flatnonzero(qid==q)) for q in np.unique(qid)])
    return X[idx],y[idx],qid[idx],idx
