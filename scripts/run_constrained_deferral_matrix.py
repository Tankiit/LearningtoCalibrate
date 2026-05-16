"""
Run constrained-deferral LP diagnostics across extracted feature caches.

This script is intentionally cache-first:
  - skips missing outputs/step1_extract/{model}/{dataset}/hidden_states.pt
  - caches expensive bootstrap interval arrays under outputs/constrained_deferral_cache/
  - writes one summary parquet and one per-item parquet per cell/K
  - writes an aggregate status parquet

Examples:
  python scripts/run_constrained_deferral_matrix.py \
    --models llama3_8b \
    --datasets truthfulqa \
    --k-list 5 20

  python scripts/run_constrained_deferral_matrix.py \
    --models llama3_8b mistral_7b qwen2_5_7b \
    --datasets chaosnli pavlick_nli ambigqa_kge2 truthfulqa halueval_qa \
    --k-list 20
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.constrained_deferral import run_bootstrap_features  # noqa: E402
from utils.paths import hidden_states_path  # noqa: E402


DEFAULT_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DEFAULT_DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "truthfulqa",
    "halueval_qa",
    "triviaqa",
    "popqa",
    "bioasq",
]


@dataclass
class MatrixStatus:
    model: str
    dataset: str
    seed: int
    k_boot: int
    status: str
    seconds: float
    summary_path: str
    item_path: str
    message: str


def _paths(out_dir: Path, model: str, dataset: str, seed: int, k_boot: int) -> tuple[Path, Path]:
    stem = f"{model}__{dataset}__seed{seed}__k{k_boot}"
    return out_dir / f"{stem}.parquet", out_dir / f"{stem}_items.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained deferral over available extracted features.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k-list", nargs="+", type=int, default=[20])
    parser.add_argument("--out-dir", type=Path, default=Path("results/constrained_deferral_matrix"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/constrained_deferral_cache"))
    parser.add_argument("--status-out", type=Path, default=Path("results/constrained_deferral_matrix_status.parquet"))
    parser.add_argument("--force", action="store_true", help="Overwrite summary/item outputs.")
    parser.add_argument("--force-cache", action="store_true", help="Recompute bootstrap interval cache.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    statuses: list[MatrixStatus] = []

    for model in args.models:
        for dataset in args.datasets:
            h_path = hidden_states_path(model, dataset)
            if not h_path.exists():
                for seed in args.seeds:
                    for k_boot in args.k_list:
                        statuses.append(MatrixStatus(
                            model=model,
                            dataset=dataset,
                            seed=seed,
                            k_boot=k_boot,
                            status="skipped_missing_features",
                            seconds=0.0,
                            summary_path="",
                            item_path="",
                            message=f"missing {h_path}",
                        ))
                print(f"[skip] {model}/{dataset}: missing {h_path}")
                continue

            for seed in args.seeds:
                for k_boot in args.k_list:
                    summary_path, item_path = _paths(args.out_dir, model, dataset, seed, k_boot)
                    if summary_path.exists() and item_path.exists() and not args.force:
                        statuses.append(MatrixStatus(
                            model=model,
                            dataset=dataset,
                            seed=seed,
                            k_boot=k_boot,
                            status="skipped_existing",
                            seconds=0.0,
                            summary_path=str(summary_path),
                            item_path=str(item_path),
                            message="outputs already exist",
                        ))
                        print(f"[skip-existing] {summary_path}")
                        continue

                    start = time.time()
                    try:
                        df = run_bootstrap_features(
                            model=model,
                            dataset=dataset,
                            seed=seed,
                            k_boot=k_boot,
                            item_out=item_path,
                            cache_dir=args.cache_dir,
                            force_cache=args.force_cache,
                            verbose=True,
                        )
                        df.to_parquet(summary_path, index=False)
                        statuses.append(MatrixStatus(
                            model=model,
                            dataset=dataset,
                            seed=seed,
                            k_boot=k_boot,
                            status="ok",
                            seconds=time.time() - start,
                            summary_path=str(summary_path),
                            item_path=str(item_path),
                            message="",
                        ))
                        print(f"[ok] wrote {summary_path}")
                    except Exception as exc:  # noqa: BLE001 - matrix runner should keep going.
                        statuses.append(MatrixStatus(
                            model=model,
                            dataset=dataset,
                            seed=seed,
                            k_boot=k_boot,
                            status="failed",
                            seconds=time.time() - start,
                            summary_path=str(summary_path),
                            item_path=str(item_path),
                            message=f"{type(exc).__name__}: {exc}",
                        ))
                        print(f"[fail] {model}/{dataset}/seed{seed}/k{k_boot}: {exc}")

    status_df = pd.DataFrame([asdict(s) for s in statuses])
    status_df.to_parquet(args.status_out, index=False)
    print(f"\nWrote {args.status_out}")
    if not status_df.empty:
        print(status_df.groupby(["status"]).size().to_string())

    raise SystemExit(1 if status_df["status"].eq("failed").any() else 0)


if __name__ == "__main__":
    main()

