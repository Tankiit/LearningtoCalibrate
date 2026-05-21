"""Aggregate Path 3 held-out preregistration checks across models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/path3_aggregate_summary.parquet"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("path3_prereg_check_*.json")):
        blob = json.loads(path.read_text())
        model = blob["model"]
        metrics = blob["metrics"]
        width = blob["width_stats"]
        corr = blob["defer_width_correctness_corr"]
        for check_id, check in blob["verdict"]["checks"].items():
            rows.append({
                "model": model,
                "check_id": check_id,
                "pass": bool(check["pass"]),
                "observed": json.dumps(check["observed"]),
                "aurc_gap": metrics["aurc_gap"],
                "aurc_p_pred": metrics["aurc_p_pred"],
                "aurc_improvement": metrics["aurc_p_pred"] - metrics["aurc_gap"],
                "defer_width_median": width["defer_width_median"],
                "defer_width_p90": width["defer_width_p90"],
                "width_correctness_rho": corr["rho"],
                "width_correctness_p": corr["p"],
                "source": str(path),
            })

    if not rows:
        raise FileNotFoundError(f"No path3_prereg_check_*.json files in {args.results_dir}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    summary = df.groupby("check_id")["pass"].agg(["sum", "count"]).reset_index()
    print(summary.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
