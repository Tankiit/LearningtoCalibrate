"""
Run the full sweep: 3 models × 8 datasets × 3 seeds = 72 experiments.

Each cell runs steps 1-4 in sequence. Cells are independent — failures
in one cell don't block others. The script logs a STATUS.md summary at
the end so you can see what succeeded.

Usage:
  python scripts/run_full_sweep.py
  python scripts/run_full_sweep.py --models llama3_8b mistral_7b
  python scripts/run_full_sweep.py --datasets truthfulqa halueval_qa
  python scripts/run_full_sweep.py --skip-extract   # if extraction is already done
"""
import argparse
import subprocess
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logging import get_logger
from utils.paths import results_path, OUTPUTS_ROOT

log = get_logger(__name__)

DEFAULT_MODELS   = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
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
DEFAULT_SEEDS    = [0, 1, 2]


def _experiment_config_path(model: str, dataset: str) -> Path:
    """Compose a config path. If the per-pair config doesn't exist, create
    it on the fly by composing model+dataset configs."""
    config_dir = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    p = config_dir / f"{model}_{dataset}.yaml"
    if p.exists():
        return p
    # Auto-generate a minimal config if not present
    p.write_text(
        f"extends:\n"
        f"  - models/{model}\n"
        f"  - datasets/{dataset}\n"
        f"seed: 0\n"
    )
    return p


def run_cell(model: str, dataset: str, seed: int, args) -> tuple[str, bool]:
    """Run all 4 steps for one (model, dataset, seed). Returns (label, ok)."""
    label = f"{model}/{dataset}/seed{seed}"
    # If results already exist, skip (idempotency)
    if results_path(model, dataset, seed).exists():
        log.info(f"[skip-done] {label}")
        return label, True

    # Override seed via env var path: the config file's seed: 0 is overridden
    # at the per-cell level. We do this by writing a tiny seed-override config.
    # Simpler approach: pass --seed on the command line. But our scripts read
    # from the config. So we emit a per-seed shim config here.
    cfg_path = _experiment_config_path(model, dataset)
    cfg_dir = cfg_path.parent
    seed_cfg = cfg_dir / f"_run_{model}_{dataset}_seed{seed}.yaml"
    seed_cfg.write_text(
        f"extends: [{cfg_path.relative_to(cfg_dir.parent).as_posix().replace('.yaml','')}]\n"
        f"seed: {seed}\n"
    )

    steps = [
        ["python", "scripts/step1_extract.py",       "--config", str(seed_cfg)],
        ["python", "scripts/step2_train_probes.py",  "--config", str(seed_cfg)],
        ["python", "scripts/step3_calibrate.py",     "--config", str(seed_cfg)],
        ["python", "scripts/step4_evaluate.py",      "--config", str(seed_cfg)],
    ]
    if args.skip_extract:
        steps = steps[1:]   # skip step 1
    if args.up_to_step is not None:
        steps = steps[: args.up_to_step]

    for step in steps:
        log.info(f"[run] {label}: {step[1]}")
        try:
            subprocess.run(step, check=True)
        except subprocess.CalledProcessError:
            log.error(f"[fail] {label} at {step[1]}")
            traceback.print_exc()
            return label, False
    return label, True


def write_status(results: list[tuple[str, bool]]) -> None:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUTS_ROOT / "STATUS.md"
    n_ok = sum(1 for _, ok in results if ok)
    lines = ["# Sweep status\n",
             f"\n{n_ok}/{len(results)} cells succeeded.\n\n"]
    for label, ok in results:
        mark = "✓" if ok else "✗"
        lines.append(f"- {mark} {label}\n")
    status_path.write_text("".join(lines))
    log.info(f"wrote {status_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",       nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets",     nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--seeds",        nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--up-to-step",   type=int, default=None,
                   help="Run only the first N steps (1=extract, 4=eval)")
    args = p.parse_args()

    results = []
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                results.append(run_cell(model, dataset, seed, args))

    write_status(results)
    n_ok = sum(1 for _, ok in results if ok)
    log.info(f"sweep complete: {n_ok}/{len(results)} succeeded")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
