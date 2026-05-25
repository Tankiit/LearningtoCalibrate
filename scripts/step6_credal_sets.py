"""Run conformal credal-set coverage/width evaluation.

This script expects existing OVA probe artifacts. Beta-adaptive rows are emitted
only for cells whose signals contain a `beta` key.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.calibration_analysis import DATASETS, MODELS, SEEDS  # noqa: E402
from eval.coverage import (  # noqa: E402
    ALPHAS,
    adaptivity_analysis,
    run_credal_matrix,
    summarize_coverage_width,
    width_reduction_vs_fixed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate conformal KL credal intervals on OVA probe signals."
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--beta-transform",
        choices=["abs", "neg", "pos", "raw"],
        default="abs",
        help="Uncertainty transform applied to beta before isotonic fitting.",
    )
    parser.add_argument(
        "--no-proxies",
        action="store_true",
        help="Disable adaptive proxy methods derived from existing signals, e.g. neg_gap.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/credal_sets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = run_credal_matrix(
        outputs_dir=args.outputs_dir,
        models=args.models,
        datasets=args.datasets,
        seeds=args.seeds,
        alphas=args.alphas,
        split_seed=args.split_seed,
        beta_transform=args.beta_transform,
        include_proxies=not args.no_proxies,
    )
    results_path = args.out_dir / "credal_results.parquet"
    df.to_parquet(results_path, index=False)
    print(f"Saved: {results_path} ({len(df)} rows)")

    summary = summarize_coverage_width(df)
    summary_path = args.out_dir / "coverage_width_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    if not summary.empty:
        print("\nCoverage-width summary:")
        print(summary.to_string(index=False))

    reductions = width_reduction_vs_fixed(df)
    reductions_path = args.out_dir / "width_reduction_vs_fixed.csv"
    reductions.to_csv(reductions_path, index=False)
    print(f"\nSaved: {reductions_path}")
    if not reductions.empty:
        print("\nWidth reduction vs fixed:")
        print(reductions.to_string(index=False))
    else:
        print("\n[INFO] No adaptive rows found. Add beta or semantic entropy scores to signals.pt.")

    adaptivity = adaptivity_analysis(df)
    adaptivity_path = args.out_dir / "adaptivity_by_dataset.csv"
    adaptivity.to_csv(adaptivity_path, index=False)
    print(f"Saved: {adaptivity_path}")


if __name__ == "__main__":
    main()
