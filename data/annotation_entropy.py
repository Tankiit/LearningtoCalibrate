"""Annotation-disagreement utilities.

Adapters in this repo return `Record` objects. Dataset-specific annotation
metadata, when available, lives in `record.meta`. This module extracts
per-item annotation entropy without requiring each adapter to expose a top-level
`annotation_entropy` attribute.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.schema import Record


def load_entropy_for_dataset(dataset: str, items: Sequence[Record]) -> np.ndarray:
    """Return annotation entropies in bits for a dataset.

    Missing entries are returned as NaN. For ChaosNLI, entropy is computed from
    `record.meta["label_counter"]`, which the current adapter already stores.
    """
    if dataset == "chaosnli":
        entropies = np.asarray([_entropy_from_chaosnli_record(item) for item in items], dtype=float)
    else:
        entropies = np.asarray([_generic_entropy(item) for item in items], dtype=float)

    valid = np.isfinite(entropies)
    if not valid.any():
        print(f"[WARN] no valid annotation entropies found for {dataset!r}.")
        if items:
            first = items[0]
            print("Available Record attributes:", sorted(vars(first).keys()))
            print("Available meta keys:", sorted((first.meta or {}).keys()))
            add_entropy_to_adapter(dataset)
    else:
        print(
            f"{dataset}: loaded annotation entropy for "
            f"{int(valid.sum())}/{len(entropies)} items "
            f"(mean={float(np.nanmean(entropies)):.4f} bits)"
        )
    return entropies


def add_entropy_to_adapter(dataset: str) -> None:
    """Print adapter-specific instructions for adding entropy metadata."""
    if dataset == "chaosnli":
        print(
            "ChaosNLI entropy can be computed from Record.meta['label_counter']. "
            "The adapter should preserve the raw label counts as a dict over "
            "entailment/neutral/contradiction labels. The current adapter uses "
            "the key 'label_counter'."
        )
    else:
        print(
            f"No dataset-specific instructions for {dataset!r}. Add raw "
            "annotation counts or probabilities to Record.meta, then extend "
            "data.annotation_entropy.load_entropy_for_dataset."
        )


def entropy_from_counts(counts: list[int] | dict[Any, int | float] | np.ndarray) -> float:
    """Compute Shannon entropy in bits from raw annotation counts."""
    if isinstance(counts, dict):
        values = np.asarray([float(v) for v in counts.values()], dtype=float)
    else:
        values = np.asarray(counts, dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else float("nan")
    probs = values / values.sum()
    return float(-(probs * np.log2(probs)).sum())


def entropy_from_distribution(p: list[float] | np.ndarray) -> float:
    """Compute Shannon entropy in bits from a probability distribution."""
    probs = np.asarray(p, dtype=float).reshape(-1)
    probs = probs[np.isfinite(probs) & (probs > 0.0)]
    if len(probs) <= 1:
        return 0.0 if len(probs) == 1 else float("nan")
    probs = probs / probs.sum()
    return float(-(probs * np.log2(probs)).sum())


def normalized_entropy_from_counts(counts: dict[Any, int | float]) -> float:
    """Compute entropy(counts) / log2(K), retained for old diagnostics."""
    if not counts:
        return float("nan")
    return entropy_from_counts(counts) / np.log2(len(counts))


def _entropy_from_chaosnli_record(item: Record) -> float:
    meta = item.meta or {}
    counts = (
        meta.get("label_counter")
        or meta.get("label_counts")
        or meta.get("label_distribution")
    )
    if isinstance(counts, dict):
        return entropy_from_counts(counts)
    if isinstance(counts, (list, tuple, np.ndarray)):
        return entropy_from_counts(counts)
    return _generic_entropy(item)


def _generic_entropy(item: Record) -> float:
    for attr in ("annotation_entropy", "entropy", "annotator_entropy_norm"):
        if hasattr(item, attr):
            return _finite_or_nan(getattr(item, attr))
    meta = item.meta or {}
    for key in ("annotation_entropy", "entropy", "annotator_entropy_norm"):
        if key in meta:
            return _finite_or_nan(meta[key])
    for key in ("label_counter", "label_counts", "label_distribution"):
        counts = meta.get(key)
        if isinstance(counts, dict):
            return entropy_from_counts(counts)
        if isinstance(counts, (list, tuple, np.ndarray)):
            return entropy_from_counts(counts)
    return float("nan")


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


if __name__ == "__main__":
    h0 = entropy_from_counts([100, 0, 0])
    h1 = entropy_from_counts([50, 50, 0])
    h2 = entropy_from_counts([33, 33, 34])
    assert abs(h0) < 1e-6
    assert abs(h1 - 1.0) < 1e-6
    assert abs(h2 - np.log2(3)) < 0.01
    print("annotation_entropy sanity checks passed")
    print(f"  [100, 0, 0]  -> H = {h0:.3f} bits")
    print(f"  [50, 50, 0]  -> H = {h1:.3f} bits")
    print(f"  [33, 33, 34] -> H = {h2:.3f} bits")
