"""Three frozen folds; clustered observations are split as whole groups."""
import hashlib
import json
from types import MappingProxyType
import numpy as np


def split_hash(idx):
    """Order-sensitive, platform-independent SHA256 of row indices."""
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def make_splits(n, fracs=(0.3, 0.4, 0.3), seed=0, groups=None):
    f = np.asarray(fracs, dtype=float)
    if n < 3 or f.shape != (3,) or np.any(f <= 0) or not np.isclose(f.sum(), 1):
        raise ValueError("Need n >= 3 and three positive fractions summing to one")
    if groups is None:
        inverse, units = np.arange(n), n
    else:
        groups = np.asarray(groups)
        if groups.shape != (n,):
            raise ValueError("One group id is required per row")
        _, inverse = np.unique(groups.astype(str), return_inverse=True)
        units = int(inverse.max()) + 1
    sizes = np.floor(f * units).astype(int)
    sizes[np.argmax(f)] += units - sizes.sum()
    if np.any(sizes == 0):
        raise ValueError("Too few independent units for nonempty folds")
    perm = np.random.default_rng(seed).permutation(units)
    chunks = np.split(perm, np.cumsum(sizes)[:-1])
    result = {}
    for name, ids in zip(("fit", "cal", "test"), chunks):
        idx = np.flatnonzero(np.isin(inverse, ids))
        idx.setflags(write=False)
        result[name] = idx
    assert_disjoint(result, n=n, groups=groups)
    return MappingProxyType(result)


def assert_disjoint(splits, n=None, groups=None):
    if set(splits) != {"fit", "cal", "test"}:
        raise ValueError("Expected exactly fit, cal, test")
    values = [np.asarray(splits[k]) for k in ("fit", "cal", "test")]
    if any(v.ndim != 1 or v.dtype.kind not in "iu" for v in values):
        raise ValueError("Indices must be integer vectors")
    all_idx = np.concatenate(values)
    if len(np.unique(all_idx)) != len(all_idx) or np.any(all_idx < 0):
        raise ValueError("Folds overlap or contain duplicate/negative indices")
    if n is not None and not np.array_equal(np.sort(all_idx), np.arange(n)):
        raise ValueError("Folds must cover each row exactly once")
    if groups is not None:
        g = np.asarray(groups).astype(str)
        sets = [set(g[v]) for v in values]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("A group crosses folds")


def freeze_splits(path, splits, dataset_hash):
    """Persist actual indices. Refuse to overwrite a different split or dataset."""
    from pathlib import Path
    record = {"dataset_hash": dataset_hash, "indices": {k: v.tolist() for k, v in splits.items()},
              "hashes": {k: split_hash(v) for k, v in splits.items()}}
    p = Path(path)
    if p.exists():
        if json.loads(p.read_text()) != record:
            raise ValueError("Frozen split already exists with different contents")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x") as handle:
            json.dump(record, handle, indent=2)
    return record
