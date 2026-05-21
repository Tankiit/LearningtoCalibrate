"""Run Path 2 Verma diagnostic across cached model/dataset cells."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.path2_verma_ova_alternative import Config, DEFAULT_DATASETS, DEFAULT_MODELS, run  # noqa: E402
from utils.paths import hidden_states_path  # noqa: E402


@dataclass
class CellStatus:
    model: str
    dataset: str
    status: str
    seconds: float
    message: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/path2_matrix"))
    parser.add_argument("--K", type=int, default=20, dest="k_bootstrap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[CellStatus] = []

    for model in args.models:
        for dataset in args.datasets:
            cache = hidden_states_path(model, dataset)
            if not cache.exists():
                msg = f"missing cache: {cache}"
                print(f"[skip] {model}/{dataset}: {msg}")
                statuses.append(CellStatus(model, dataset, "skipped_missing", 0.0, msg))
                continue

            summary = args.output_dir / f"path2_diagnostic_summary_{model}_{dataset}.json"
            if summary.exists() and not args.force:
                msg = f"existing summary: {summary}"
                print(f"[skip] {model}/{dataset}: {msg}")
                statuses.append(CellStatus(model, dataset, "skipped_existing", 0.0, msg))
                continue

            cfg = Config(
                model=model,
                dataset=dataset,
                output_dir=args.output_dir,
                k_bootstrap=args.k_bootstrap,
                seed=args.seed,
                max_items=args.max_items,
            )
            print(f"[run] {model}/{dataset} K={args.k_bootstrap}")
            start = time.time()
            try:
                run(cfg)
            except Exception as exc:
                seconds = time.time() - start
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[fail] {model}/{dataset}: {msg}")
                statuses.append(CellStatus(model, dataset, "failed", seconds, msg))
            else:
                seconds = time.time() - start
                statuses.append(CellStatus(model, dataset, "ok", seconds, ""))

            status_path = args.output_dir / "path2_matrix_status.parquet"
            pd.DataFrame([asdict(x) for x in statuses]).to_parquet(status_path, index=False)
            (args.output_dir / "path2_matrix_status.json").write_text(
                json.dumps([asdict(x) for x in statuses], indent=2)
            )

    status_path = args.output_dir / "path2_matrix_status.parquet"
    pd.DataFrame([asdict(x) for x in statuses]).to_parquet(status_path, index=False)
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
