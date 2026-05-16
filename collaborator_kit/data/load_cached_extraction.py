"""Validate and load appendix collaborator cache cells.

This module intentionally stays small and dependency-light. It checks the file
contract documented in collaborator_kit/INTERFACE.md and gives collaborators a
single entry point before they start appendix tasks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODELS = (
    "llama3.1_8b_inst",
    "mistral_7b_inst_v0.3",
    "qwen2.5_7b_inst",
)

DEFAULT_DATASETS = (
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "truthfulqa",
    "halueval",
)

REQUIRED_FILES = (
    "contrastive_h.pt",
    "contrastive_meta.pt",
    "gen_greedy_h.pt",
    "gen_greedy_meta.pt",
    "judge_greedy.pt",
)

NSAMPLES_DATASETS = {"chaosnli", "truthfulqa"}


@dataclass(frozen=True)
class CacheCellStatus:
    model: str
    dataset: str
    path: Path
    ok: bool
    missing: tuple[str, ...]
    error: str | None = None


def expected_files(dataset: str) -> tuple[str, ...]:
    files = list(REQUIRED_FILES)
    if dataset in NSAMPLES_DATASETS:
        files.append("nsamples10.pt")
    return tuple(files)


def load_pt(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"{path} must contain a dict, got {type(blob).__name__}")
    return blob


def validate_cell(root: Path, model: str, dataset: str, deep: bool = True) -> CacheCellStatus:
    cell = root / model / dataset
    missing = tuple(name for name in expected_files(dataset) if not (cell / name).exists())
    if missing:
        return CacheCellStatus(model, dataset, cell, False, missing)

    if not deep:
        return CacheCellStatus(model, dataset, cell, True, ())

    try:
        contrastive_h = load_pt(cell / "contrastive_h.pt")
        contrastive_meta = load_pt(cell / "contrastive_meta.pt")
        gen_h = load_pt(cell / "gen_greedy_h.pt")
        gen_meta = load_pt(cell / "gen_greedy_meta.pt")
        judge = load_pt(cell / "judge_greedy.pt")

        h_pos = contrastive_h["h_pos"]
        h_neg = contrastive_h["h_neg"]
        records = contrastive_meta["records"]
        if len(h_pos) != len(h_neg) or len(h_pos) != len(records):
            raise ValueError(
                "contrastive alignment mismatch: "
                f"h_pos={len(h_pos)} h_neg={len(h_neg)} records={len(records)}"
            )

        greedy_h = gen_h["h"]
        greedy_records = gen_meta["records"]
        y_correct = judge["y_correct"]
        if len(greedy_h) != len(greedy_records) or len(greedy_h) != len(y_correct):
            raise ValueError(
                "greedy alignment mismatch: "
                f"h={len(greedy_h)} records={len(greedy_records)} "
                f"y_correct={len(y_correct)}"
            )

        if dataset in NSAMPLES_DATASETS:
            nsamples = load_pt(cell / "nsamples10.pt")
            samples = nsamples["samples"]
            sample_records = nsamples["records"]
            if len(samples) != len(sample_records):
                raise ValueError(
                    "sample alignment mismatch: "
                    f"samples={len(samples)} records={len(sample_records)}"
                )
            if samples and any(len(row) != 10 for row in samples):
                raise ValueError("nsamples10.pt must contain exactly 10 samples per row")
    except Exception as exc:  # noqa: BLE001 - CLI should report validation failures plainly.
        return CacheCellStatus(model, dataset, cell, False, (), str(exc))

    return CacheCellStatus(model, dataset, cell, True, ())


def iter_statuses(
    root: Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    deep: bool = True,
) -> list[CacheCellStatus]:
    return [
        validate_cell(root, model, dataset, deep=deep)
        for model in models
        for dataset in datasets
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate collaborator cache files.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "cached_extractions")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--check", action="store_true", help="Validate cache and print a summary.")
    parser.add_argument("--shallow", action="store_true", help="Only check file presence.")
    args = parser.parse_args()

    if not args.check:
        parser.error("Only --check mode is currently supported.")

    statuses = iter_statuses(
        args.root,
        models=tuple(args.models),
        datasets=tuple(args.datasets),
        deep=not args.shallow,
    )
    ok = [s for s in statuses if s.ok]
    print(
        f"{len(ok)}/{len(statuses)} expected cache cells valid "
        f"under {args.root}"
    )
    for status in statuses:
        if status.ok:
            print(f"OK      {status.model}/{status.dataset}")
        elif status.missing:
            print(
                f"MISSING {status.model}/{status.dataset}: "
                + ", ".join(status.missing)
            )
        else:
            print(f"INVALID {status.model}/{status.dataset}: {status.error}")

    if len(ok) != len(statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

