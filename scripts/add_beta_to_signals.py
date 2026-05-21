"""Attach beta sidecar scores to existing signals.pt artifacts.

This script does not compute beta. It ingests real beta scores produced by the
base-vs-instruct extraction pipeline and writes them into the corresponding
`signals.pt` files under the key `beta`.

Supported sidecar layouts:
  <beta-dir>/<model>/<dataset>/seed<seed>/beta.{pt,csv,parquet,jsonl}
  <beta-dir>/<model>/<dataset>/beta.{pt,csv,parquet,jsonl}
  <beta-dir>/<model>_<dataset>_seed<seed>.{pt,csv,parquet,jsonl}
  <beta-dir>/<model>_<dataset>.{pt,csv,parquet,jsonl}

Supported schemas:
  - row-aligned array/key: beta
  - beta_pos and beta_neg arrays plus example_ids
  - table with columns example_id, optional is_pos, and beta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.calibration_analysis import DATASETS, MODELS, SEEDS, resolve_signals_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add beta scores to signals.pt files.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--beta-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = 0
    skipped = 0

    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                signals_path = resolve_signals_path(args.outputs_dir, model, dataset, seed)
                beta_path = resolve_beta_path(args.beta_dir, model, dataset, seed)
                if beta_path is None:
                    print(f"SKIP {model:14s} {dataset:14s} seed={seed}: no beta sidecar")
                    skipped += 1
                    continue
                if not signals_path.exists():
                    print(f"SKIP {model:14s} {dataset:14s} seed={seed}: missing {signals_path}")
                    skipped += 1
                    continue

                raw = torch.load(signals_path, map_location="cpu", weights_only=False)
                beta, beta_valid = load_beta(beta_path, raw)
                if args.dry_run:
                    print(
                        f"DRY  {model:14s} {dataset:14s} seed={seed}: "
                        f"would add beta from {beta_path}"
                    )
                else:
                    raw["beta"] = beta.astype(np.float32)
                    raw["beta_valid"] = beta_valid.astype(bool)
                    torch.save(raw, signals_path)
                    print(
                        f"OK   {model:14s} {dataset:14s} seed={seed}: "
                        f"added beta ({int(beta_valid.sum())}/{len(beta_valid)} valid rows)"
                    )
                written += 1

    print(f"Done. Added beta to {written} cells; skipped {skipped}.")


def resolve_beta_path(beta_dir: Path, model: str, dataset: str, seed: int) -> Optional[Path]:
    candidates: list[Path] = []
    for suffix in ("pt", "csv", "parquet", "jsonl"):
        candidates.extend(
            [
                beta_dir / model / dataset / f"seed{seed}" / f"beta.{suffix}",
                beta_dir / model / dataset / f"beta.{suffix}",
                beta_dir / f"{model}_{dataset}_seed{seed}.{suffix}",
                beta_dir / f"{model}_{dataset}.{suffix}",
            ]
        )
    return next((path for path in candidates if path.exists()), None)


def load_beta(path: Path, signals_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            return beta_from_mapping(blob, signals_raw)
        beta = validate_beta(to_numpy(blob, dtype=float), signals_raw)
        return beta, np.ones(len(beta), dtype=bool)
    if suffix == ".csv":
        return beta_from_frame(pd.read_csv(path), signals_raw)
    if suffix == ".parquet":
        return beta_from_frame(pd.read_parquet(path), signals_raw)
    if suffix == ".jsonl":
        return beta_from_frame(pd.read_json(path, lines=True), signals_raw)
    raise ValueError(f"unsupported beta sidecar extension: {suffix}")


def beta_from_mapping(blob: dict, signals_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    if "beta" in blob and "example_ids" in blob:
        return align_item_beta(
            to_numpy(blob["example_ids"], dtype=str),
            to_numpy(blob["beta"], dtype=float),
            signals_raw,
        )
    if "beta" in blob:
        beta = validate_beta(to_numpy(blob["beta"], dtype=float), signals_raw)
        return beta, np.ones(len(beta), dtype=bool)
    if "beta_pos" in blob and "beta_neg" in blob and "example_ids" in blob:
        return align_pos_neg(
            to_numpy(blob["example_ids"], dtype=str),
            to_numpy(blob["beta_pos"], dtype=float),
            to_numpy(blob["beta_neg"], dtype=float),
            signals_raw,
        )
    if "rows" in blob:
        return beta_from_frame(pd.DataFrame(blob["rows"]), signals_raw)
    raise ValueError("beta sidecar dict must contain beta or beta_pos/beta_neg/example_ids")


def beta_from_frame(df: pd.DataFrame, signals_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    if "beta" not in df.columns:
        raise ValueError("table beta sidecar must contain a beta column")
    if len(df) == len(signals_raw["logit_pred"]) and "example_id" not in df.columns:
        beta = validate_beta(df["beta"].to_numpy(dtype=float), signals_raw)
        return beta, np.ones(len(beta), dtype=bool)
    if "example_id" not in df.columns:
        raise ValueError("table beta sidecar must include example_id unless row-aligned")

    signal_ids = np.asarray(signals_raw["example_ids"], dtype=str)
    signal_is_pos = np.asarray(signals_raw.get("is_pos", np.ones(len(signal_ids))), dtype=bool)
    out = np.zeros(len(signal_ids), dtype=float)
    valid = np.zeros(len(signal_ids), dtype=bool)

    if "is_pos" in df.columns:
        lookup = {
            (str(row.example_id), bool(row.is_pos)): float(row.beta)
            for row in df.itertuples(index=False)
        }
        for i, key in enumerate(zip(signal_ids, signal_is_pos)):
            lookup_key = (str(key[0]), bool(key[1]))
            if lookup_key in lookup:
                out[i] = lookup[lookup_key]
                valid[i] = True
    else:
        lookup = {str(row.example_id): float(row.beta) for row in df.itertuples(index=False)}
        for i, example_id in enumerate(signal_ids):
            if str(example_id) in lookup:
                out[i] = lookup[str(example_id)]
                valid[i] = True
    return validate_beta(out, signals_raw), valid


def align_item_beta(
    example_ids: np.ndarray,
    beta_values: np.ndarray,
    signals_raw: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Align item-level beta values to row-level contrastive signals."""
    if len(example_ids) != len(beta_values):
        raise ValueError("example_ids and beta lengths must match")
    lookup = dict(zip(example_ids.astype(str), beta_values.astype(float)))
    signal_ids = np.asarray(signals_raw["example_ids"], dtype=str)
    out = np.zeros(len(signal_ids), dtype=float)
    valid = np.zeros(len(signal_ids), dtype=bool)
    for i, example_id in enumerate(signal_ids):
        value = lookup.get(str(example_id))
        if value is not None:
            out[i] = value
            valid[i] = True
    return validate_beta(out, signals_raw), valid


def align_pos_neg(
    example_ids: np.ndarray,
    beta_pos: np.ndarray,
    beta_neg: np.ndarray,
    signals_raw: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if len(example_ids) != len(beta_pos) or len(example_ids) != len(beta_neg):
        raise ValueError("example_ids, beta_pos, and beta_neg lengths must match")
    pos_lookup = dict(zip(example_ids.astype(str), beta_pos.astype(float)))
    neg_lookup = dict(zip(example_ids.astype(str), beta_neg.astype(float)))
    signal_ids = np.asarray(signals_raw["example_ids"], dtype=str)
    signal_is_pos = np.asarray(signals_raw.get("is_pos", np.ones(len(signal_ids))), dtype=bool)
    out = np.empty(len(signal_ids), dtype=float)
    valid = np.ones(len(signal_ids), dtype=bool)
    for i, (example_id, is_pos) in enumerate(zip(signal_ids, signal_is_pos)):
        out[i] = pos_lookup[str(example_id)] if is_pos else neg_lookup[str(example_id)]
    return validate_beta(out, signals_raw), valid


def validate_beta(beta: np.ndarray, signals_raw: dict) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    expected = len(signals_raw["logit_pred"])
    if len(beta) != expected:
        raise ValueError(f"length mismatch: beta has {len(beta)}, expected {expected}")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains non-finite values")
    return beta


def to_numpy(value, dtype=float) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).reshape(-1)


if __name__ == "__main__":
    main()
