"""
Path 3: held-out MultiNLI-Disagreement validation.

Run extraction first:

    modal run extraction/extract_full.py --only-models llama3_8b --only-datasets multinli_disagreement --skip-gen --skip-judge --skip-samples

Then compact/import as with the other datasets, and run this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.path2_verma_ova_alternative import Config as Path2Config  # noqa: E402
from scripts.path2_verma_ova_alternative import (  # noqa: E402
    evaluate_construction,
    load_contrastive_cache,
    split_by_item,
)


@dataclass
class HeldoutConfig:
    model: str
    output_dir: Path = Path("results")
    prereg_path: Path = Path("PRE_REGISTRATION.json")
    k_bootstrap: int = 20
    seed: int = 0
    max_items: int | None = None


def _heldout_prereg(prereg_path: Path) -> dict[str, Any]:
    if not prereg_path.exists():
        raise FileNotFoundError(f"Missing preregistration file: {prereg_path}")
    prereg = json.loads(prereg_path.read_text())
    heldout = prereg.get("heldout_validation")
    if not heldout:
        raise KeyError(
            f"{prereg_path} does not contain heldout_validation. "
            "Add the Path 3 pre-registration entry before running held-out evaluation."
        )
    return heldout


def check_predictions(result: dict[str, Any], heldout: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    width = result["width_stats"]
    corr = result["defer_width_correctness_corr"]
    improvement = metrics["aurc_p_pred"] - metrics["aurc_gap"]

    checks = {
        "p1_credal_width_classification": {
            "claim": "median amb/defer bootstrap width > 0.10",
            "observed": width["defer_width_median"],
            "pass": bool(width["defer_width_median"] > 0.10),
        },
        "p2_aurc_improvement_direction": {
            "claim": "AURC(gap) < AURC(p_pred)",
            "observed": {
                "aurc_gap": metrics["aurc_gap"],
                "aurc_p_pred": metrics["aurc_p_pred"],
            },
            "pass": bool(metrics["aurc_gap"] < metrics["aurc_p_pred"]),
        },
        "p3_aurc_improvement_magnitude": {
            "claim": "AURC(p_pred) - AURC(gap) >= 0.015",
            "observed": improvement,
            "pass": bool(improvement >= 0.015),
        },
        "p4_credal_width_correlates_with_correctness": {
            "claim": "rho(width, y_correct) <= -0.10 and p < 0.05",
            "observed": corr,
            "pass": bool(corr["rho"] <= -0.10 and corr["p"] < 0.05),
        },
    }
    return {
        "registered_prediction": heldout.get("regime_prediction", "showcase"),
        "checks": checks,
        "n_confirmed": int(sum(x["pass"] for x in checks.values())),
        "n_predictions": len(checks),
    }


def run(cfg: HeldoutConfig) -> dict[str, Any]:
    heldout = _heldout_prereg(cfg.prereg_path)
    path2_cfg = Path2Config(
        model=cfg.model,
        dataset="multinli_disagreement",
        output_dir=cfg.output_dir,
        k_bootstrap=cfg.k_bootstrap,
        seed=cfg.seed,
        max_items=cfg.max_items,
    )

    data = load_contrastive_cache(path2_cfg)
    masks = split_by_item(data, path2_cfg)
    result = evaluate_construction(data, masks, "y_expert", path2_cfg)
    verdict = check_predictions(result, heldout)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / f"path3_prereg_check_{cfg.model}.json"
    out.write_text(json.dumps({
        "model": cfg.model,
        "dataset": "multinli_disagreement",
        "prereg_path": str(cfg.prereg_path),
        "metrics": result["metrics"],
        "width_stats": result["width_stats"],
        "defer_width_correctness_corr": result["defer_width_correctness_corr"],
        "verdict": verdict,
    }, indent=2))

    print(f"\n=== Path 3 held-out validation: {cfg.model} ===")
    for key, check in verdict["checks"].items():
        mark = "PASS" if check["pass"] else "FAIL"
        print(f"  {mark:4s} {key}: {check['observed']}")
    print(f"  confirmed {verdict['n_confirmed']}/{verdict['n_predictions']}")
    print(f"Wrote {out}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["llama3_8b", "mistral_7b", "qwen2_5_7b"])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--prereg-path", type=Path, default=Path("PRE_REGISTRATION.json"))
    parser.add_argument("--K", type=int, default=20, dest="k_bootstrap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()
    run(HeldoutConfig(**vars(args)))


if __name__ == "__main__":
    main()
