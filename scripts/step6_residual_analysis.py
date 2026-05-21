"""Run residual calibration analyses from saved calibration results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.calibration_analysis import DATASETS, MODELS, SEEDS  # noqa: E402
from eval.residual_analysis import (  # noqa: E402
    format_paper_table,
    per_dataset_ece,
    run_residual_correlation_matrix,
    summarize_residual_correlations,
    verbalized_vs_probe_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run residual calibration analyses over probe artifacts."
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("results/calibration/calibration_results.parquet"),
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--verbalized-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/residual_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.calibration_results.exists():
        raise FileNotFoundError(
            f"Missing {args.calibration_results}. Run scripts/step5_calibrate_probe.py first."
        )

    results_df = pd.read_parquet(args.calibration_results)

    per_dataset = per_dataset_ece(results_df, method="platt")
    per_dataset_path = args.out_dir / "per_dataset_ece_platt.csv"
    per_dataset.to_csv(per_dataset_path, index=False)
    print(f"Saved: {per_dataset_path}")
    print("\nPer-dataset Platt ECE:")
    print(per_dataset.to_string(index=False))

    residual_df = run_residual_correlation_matrix(
        outputs_dir=args.outputs_dir,
        models=args.models,
        datasets=args.datasets,
        seeds=args.seeds,
        split_seed=args.split_seed,
        verbalized_dir=args.verbalized_dir,
        verbose=False,
    )
    residual_path = args.out_dir / "residual_vs_defer.csv"
    residual_df.to_csv(residual_path, index=False)
    print(f"\nSaved: {residual_path}")

    residual_summary = summarize_residual_correlations(residual_df)
    residual_summary_path = args.out_dir / "residual_vs_defer_summary.csv"
    residual_summary.to_csv(residual_summary_path, index=False)
    print(f"Saved: {residual_summary_path}")
    print("\nResidual vs sigma_defer summary:")
    print(residual_summary.to_string(index=False))

    comparison = verbalized_vs_probe_comparison(results_df)
    if comparison is not None:
        comparison_path = args.out_dir / "verbalized_vs_probe.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\nSaved: {comparison_path}")
        print("\nVerbalized vs probe:")
        print(comparison.to_string(index=False))

    paper_table = format_paper_table(results_df)
    paper_table_path = args.out_dir / "paper_table_formatted.csv"
    paper_table.to_csv(paper_table_path, index=False)
    print(f"\nSaved: {paper_table_path}")
    print("\nFormatted table:")
    print(paper_table.to_string(index=False))


if __name__ == "__main__":
    main()
