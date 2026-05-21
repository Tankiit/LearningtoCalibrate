"""Merge Modal beta shard sidecars into per-cell beta.pt files.

Input layout, typically downloaded from the Modal volume:
  <shards-dir>/<model>/<dataset>/shard_0000_of_0016.pt
  <shards-dir>/<model>/<dataset>/checkpoint_shard_0000_of_0016.pt

Output layout:
  <out-dir>/<model>/<dataset>/beta.pt

The output schema is accepted by scripts/add_beta_to_signals.py and keeps the
item-level beta values keyed by example_id.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SHARD_RE = re.compile(r"^shard_(\d{4})_of_(\d{4})\.pt$")
CHECKPOINT_RE = re.compile(r"^checkpoint_shard_(\d{4})_of_(\d{4})\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge beta shard sidecars.")
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--allow-checkpoints",
        action="store_true",
        help="Use latest checkpoint files for missing final shards.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write merged files even when some shards are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells = discover_cells(args.shards_dir, args.models, args.datasets)
    if not cells:
        raise SystemExit(f"No beta shard cells found under {args.shards_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for model, dataset, cell_dir in cells:
        try:
            payload = merge_cell(
                cell_dir,
                allow_checkpoints=args.allow_checkpoints,
                allow_incomplete=args.allow_incomplete,
            )
        except Exception as exc:
            print(f"SKIP {model:14s} {dataset:14s}: {exc}")
            continue

        out_dir = args.out_dir / model / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "beta.pt"
        payload["model"] = model
        payload["dataset"] = dataset
        torch.save(payload, out_path)
        valid = np.asarray(payload["beta_valid"], dtype=bool)
        print(
            f"OK   {model:14s} {dataset:14s}: "
            f"{int(valid.sum())}/{len(valid)} valid items -> {out_path}"
        )
        written += 1

    print(f"Done. Wrote {written}/{len(cells)} merged beta files.")


def discover_cells(
    shards_dir: Path,
    models: list[str] | None,
    datasets: list[str] | None,
) -> list[tuple[str, str, Path]]:
    cells: list[tuple[str, str, Path]] = []
    model_dirs = [shards_dir / model for model in models] if models else sorted(p for p in shards_dir.iterdir() if p.is_dir())
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        dataset_dirs = [model_dir / dataset for dataset in datasets] if datasets else sorted(p for p in model_dir.iterdir() if p.is_dir())
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                continue
            if any(SHARD_RE.match(p.name) or CHECKPOINT_RE.match(p.name) for p in dataset_dir.glob("*.pt")):
                cells.append((model_dir.name, dataset_dir.name, dataset_dir))
    return cells


def merge_cell(
    cell_dir: Path,
    allow_checkpoints: bool,
    allow_incomplete: bool,
) -> dict:
    final_files = parse_shard_files(cell_dir, SHARD_RE)
    checkpoint_files = parse_shard_files(cell_dir, CHECKPOINT_RE)
    if not final_files and not checkpoint_files:
        raise ValueError("no shard files")

    expected_total = infer_expected_total(final_files, checkpoint_files)
    shard_files: dict[int, Path] = dict(final_files)
    if allow_checkpoints:
        for shard_index, path in checkpoint_files.items():
            shard_files.setdefault(shard_index, path)

    missing = sorted(set(range(expected_total)) - set(shard_files))
    if missing and not allow_incomplete:
        raise ValueError(f"missing {len(missing)}/{expected_total} shards; first missing={missing[:5]}")

    rows: dict[str, tuple[float, bool]] = {}
    source_shards: dict[str, list[int] | int | bool] = {
        "expected_total": expected_total,
        "complete": len(missing) == 0,
        "missing": missing,
        "final_shards": sorted(final_files),
        "checkpoint_shards_used": sorted(set(shard_files) - set(final_files)),
    }

    result_summaries = {}
    for shard_index in sorted(shard_files):
        blob = torch.load(shard_files[shard_index], map_location="cpu", weights_only=False)
        ids = to_numpy(blob["example_ids"], dtype=str)
        beta = to_numpy(blob["beta"], dtype=float)
        valid = to_numpy(blob.get("beta_valid", np.isfinite(beta)), dtype=bool)
        if not (len(ids) == len(beta) == len(valid)):
            raise ValueError(f"{shard_files[shard_index]} has inconsistent lengths")
        result_summaries[str(shard_index)] = blob.get("result", {})
        for example_id, beta_value, is_valid in zip(ids, beta, valid):
            key = str(example_id)
            if key in rows and rows[key] != (float(beta_value), bool(is_valid)):
                raise ValueError(f"duplicate conflicting beta for example_id={key}")
            rows[key] = (float(beta_value), bool(is_valid))

    example_ids = np.asarray(sorted(rows), dtype=str)
    beta = np.asarray([rows[example_id][0] for example_id in example_ids], dtype=np.float32)
    beta_valid = np.asarray([rows[example_id][1] for example_id in example_ids], dtype=bool)
    return {
        "example_ids": example_ids,
        "beta": beta,
        "beta_valid": beta_valid,
        "source_shards": source_shards,
        "result_by_shard": result_summaries,
    }


def parse_shard_files(cell_dir: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    total_by_shard = {}
    for path in cell_dir.glob("*.pt"):
        match = pattern.match(path.name)
        if not match:
            continue
        shard_index = int(match.group(1))
        total = int(match.group(2))
        if shard_index in out:
            raise ValueError(f"duplicate shard {shard_index}: {out[shard_index]} and {path}")
        out[shard_index] = path
        total_by_shard[shard_index] = total
    totals = set(total_by_shard.values())
    if len(totals) > 1:
        raise ValueError(f"inconsistent shard totals: {sorted(totals)}")
    return out


def infer_expected_total(
    final_files: dict[int, Path],
    checkpoint_files: dict[int, Path],
) -> int:
    candidate = next(iter(final_files.values()), None) or next(iter(checkpoint_files.values()))
    match = SHARD_RE.match(candidate.name) or CHECKPOINT_RE.match(candidate.name)
    if not match:
        raise ValueError(f"cannot infer shard total from {candidate}")
    return int(match.group(2))


def to_numpy(value, dtype) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).reshape(-1)


if __name__ == "__main__":
    main()
