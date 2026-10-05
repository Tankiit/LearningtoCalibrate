"""Batch-invariance checks, strict JSON, provenance, and fail-closed gates."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np


def array_hash(X):
    a = np.ascontiguousarray(X)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def write_free_check(rule_fn, X_probe, X_other):
    """Necessary batch-invariance check, not a proof of training independence."""
    X, other = np.asarray(X_probe), np.asarray(X_other)
    if len(X) == 0:
        raise ValueError("Need a nonempty probe")
    before_x, before_other = array_hash(X), array_hash(other)
    base = np.array(rule_fn(X), copy=True)
    tests = [rule_fn(np.concatenate([X, other]))[:len(X)],
             rule_fn(np.concatenate([other, X]))[len(other):],
             rule_fn(X[::-1])[::-1],
             np.concatenate([rule_fn(x[None]) for x in X]), rule_fn(X)]
    if any(not np.array_equal(base, v) for v in tests):
        raise AssertionError("Rule depends on unrelated queries or query order")
    if before_x != array_hash(X) or before_other != array_hash(other):
        raise AssertionError("Rule mutated query data")
    return True


def source_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for folder in ("mondrian", "models", "data", "experiments", "tests"):
        for path in sorted((root / folder).glob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    for path in sorted((root / "configs").glob("*.json")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def manifest(**kw):
    required = {"split_hashes", "M", "seed", "floor", "rule", "n_below_floor"}
    if required - kw.keys():
        raise ValueError(f"Missing manifest fields: {sorted(required - kw.keys())}")
    import scipy, sklearn
    return {"schema_version": 1, "paper_status": "HOLD", "source_sha256": source_hash(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "floor_policy": "infinite_set_keep_all_rows", **kw}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")


def require_gate(path, expected_stage):
    gate = json.loads(Path(path).read_text())
    if gate.get("stage") != expected_stage or gate.get("status") != "PASS":
        raise RuntimeError(f"{expected_stage} gate has not cleared: {path}")
    if gate.get("source_sha256") != source_hash():
        raise RuntimeError("Gate is stale for current source tree; rerun validation")
    return gate
