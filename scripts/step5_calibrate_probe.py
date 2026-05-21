"""Run full calibration analysis on existing probe artifacts.

Default input layout:
  outputs/step2_probes/<model>/<dataset>/seed<seed>/signals.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.calibration_analysis import (  # noqa: E402
    DATASETS,
    MODELS,
    SEEDS,
    build_ece_reduction_table,
    build_reliability_diagram_data,
    build_summary_table,
    results_to_dataframe,
    run_full_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibration analysis on OVA f_pred probe artifacts."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root output directory, usually repo-local outputs/.",
    )
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to include.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to include.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Probe seeds to include.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for the calibration/test split within each artifact.",
    )
    parser.add_argument("--n-bins", type=int, default=15, help="Number of ECE bins.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/calibration"),
        help="Output directory for calibration result tables.",
    )
    parser.add_argument(
        "--verbalized-dir",
        type=Path,
        default=None,
        help=(
            "Optional root containing verbalized.{pt,csv,parquet,jsonl} sidecars. "
            "Checked as <root>/<model>/<dataset>/seed<seed>/verbalized.* and "
            "<root>/<model>_<dataset>_seed<seed>.*."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("OVA Probe Calibration Analysis")
    print("=" * 72)
    print(f"Outputs dir: {args.outputs_dir}")
    print(f"Models:      {args.models}")
    print(f"Datasets:    {args.datasets}")
    print(f"Seeds:       {args.seeds}")
    print(f"Split seed:  {args.split_seed}")
    print(f"Results dir: {args.out_dir}")
    print(f"Verb. dir:   {args.verbalized_dir}")
    print("=" * 72)
    print()

    all_results = run_full_matrix(
        outputs_dir=args.outputs_dir,
        models=args.models,
        datasets=args.datasets,
        seeds=args.seeds,
        split_seed=args.split_seed,
        n_bins=args.n_bins,
        verbalized_dir=args.verbalized_dir,
    )
    if not all_results:
        print("\nNo results produced. Check that signals.pt files exist.")
        return

    df = results_to_dataframe(all_results)
    parquet_path = args.out_dir / "calibration_results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved: {parquet_path} ({len(df)} rows)")

    summary = build_summary_table(df)
    summary_path = args.out_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print("\nSummary preview:")
    print(summary.to_string(index=False))

    reduction = build_ece_reduction_table(df)
    reduction_path = args.out_dir / "ece_reduction_table.csv"
    reduction.to_csv(reduction_path, index=False)
    print(f"\nSaved: {reduction_path}")
    print("\nECE reduction by ambiguity type:")
    print(reduction.to_string(index=False))

    for method in ["raw", "platt", "temperature", "verbalized", "sigma_pred", "logprob"]:
        diagram_data = build_reliability_diagram_data(all_results, method=method)
        if diagram_data:
            pkl_path = args.out_dir / f"reliability_{method}.pkl"
            with open(pkl_path, "wb") as handle:
                pickle.dump(diagram_data, handle)
            print(f"Saved: {pkl_path} ({len(diagram_data)} cells)")

    print("\n" + "-" * 72)
    print("Sanity check: Platt ECE across ambiguity types")
    platt_by_type = (
        df[df["method"] == "platt"]
        .groupby("ambiguity_type")["ece_15"]
        .agg(["mean", "std", "count"])
    )
    print(platt_by_type.to_string())

    if "verbalized" in df["method"].values:
        print("\nVerbalized confidence ECE across ambiguity types:")
        verbalized_by_type = (
            df[df["method"] == "verbalized"]
            .groupby("ambiguity_type")["ece_15"]
            .agg(["mean", "std", "count"])
        )
        print(verbalized_by_type.to_string())
    else:
        print("\n[INFO] No verbalized confidence baseline found in artifacts.")

    print("\n" + "=" * 72)
    print("Done. Results written to:", args.out_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
