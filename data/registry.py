"""
Dataset registry. Single function `load_dataset(name)` returns a list of
`Record` objects. All adapter complexity (HF loading, BioASQ multi-strategy,
expert-reliability mapping) is hidden behind this interface.

To add a new dataset:
  1. Implement an adapter in `data/adapters/{name}.py` exposing a `load()` function
  2. Register it in `_REGISTRY` below
  3. Run `pytest tests/test_data.py` to verify it conforms to the schema
"""
from __future__ import annotations
from typing import Callable

from data.schema import Record


# Adapter registry — populated lazily to avoid importing every adapter on import
_REGISTRY: dict[str, str] = {
    "ambigqa_kge2": "data.adapters.ambigqa_kge2",
    "chaosnli":    "data.adapters.chaosnli",
    "pavlick_nli": "data.adapters.pavlick_nli",
    "truthfulqa":  "data.adapters.truthfulqa",
    "halueval_qa": "data.adapters.halueval_qa",
    "triviaqa":    "data.adapters.triviaqa",
    "popqa":       "data.adapters.popqa",
    "bioasq":      "data.adapters.bioasq",
    "pubmedqa":    "data.adapters.pubmedqa",
    "medqa":       "data.adapters.medqa",
    "multinli_disagreement": "data.adapters.multinli_disagreement",
}


class _DatasetLoader:
    """Compatibility wrapper exposing `.load()` for one registered dataset."""

    def __init__(self, name: str):
        self.name = name

    def load(self) -> list[Record]:
        return load_dataset(self.name)


DATASET_REGISTRY = {
    name: _DatasetLoader(name)
    for name in _REGISTRY
}


def list_datasets() -> list[str]:
    return list(_REGISTRY.keys())


def load_dataset(name: str) -> list[Record]:
    """
    Load a dataset by name. Returns a list of Record objects.

    The adapter is imported on demand, so adapters with heavy dependencies
    (BioASQ's parquet+HF+JSON fallback) don't slow down imports of other adapters.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    import importlib
    adapter = importlib.import_module(_REGISTRY[name])
    if not hasattr(adapter, "load"):
        raise ImportError(
            f"Adapter {_REGISTRY[name]} must define a `load()` function"
        )
    records = adapter.load()

    # Schema check — fail fast if adapter returns malformed data
    if not records:
        raise RuntimeError(f"{name}: adapter returned empty list")
    _validate_records(records, name)
    return records


def _validate_records(records: list[Record], name: str) -> None:
    """Spot-check the first 5 records for schema conformance."""
    for i, r in enumerate(records[:5]):
        if not isinstance(r, Record):
            raise TypeError(
                f"{name} record {i} is {type(r).__name__}, expected Record. "
                f"Adapter must return list[Record]."
            )
        if not r.example_id or not r.question or not r.correct_answer:
            raise ValueError(
                f"{name} record {i} has empty required field. id={r.example_id!r} "
                f"q={r.question[:40]!r} ans={r.correct_answer[:40]!r}"
            )
    # Stable-id uniqueness check
    ids = [r.example_id for r in records]
    if len(set(ids)) != len(ids):
        from collections import Counter
        dup = [k for k, v in Counter(ids).items() if v > 1][:3]
        raise ValueError(f"{name}: duplicate example_ids — first 3: {dup}")
