"""
Aggregate per-cell results.parquet files into one table for paper figures/tables.

Run AFTER the sweep completes:
  python scripts/aggregate_results.py
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from utils.logging import get_logger
from utils.paths import OUTPUTS_ROOT, aggregated_results_path

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=aggregated_results_path())
    args = p.parse_args()

    files = sorted(OUTPUTS_ROOT.glob("step4_eval/*/*/seed*/results.parquet"))
    if not files:
        log.error(f"No results.parquet files found under {OUTPUTS_ROOT/'step4_eval'}")
        return
    log.info(f"aggregating {len(files)} result files")

    dfs = [pd.read_parquet(f) for f in files]
    agg = pd.concat(dfs, ignore_index=True)
    agg.to_parquet(args.out)
    log.info(f"wrote → {args.out}  ({len(agg)} rows)")

    # Print headline pivot for quick inspection
    pivot = agg.pivot_table(
        index=["model_key", "dataset"],
        columns="method", values="aurc",
        aggfunc=["mean", "std"],
    )
    log.info("\n" + pivot.to_string())


if __name__ == "__main__":
    main()
