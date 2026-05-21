"""
Run lead-author main tasks across the model/dataset matrix.

This is the orchestration layer for the non-appendix work:

  - p0: headline + aggregate + scod + audit
  - p1: selector + ood (plus future conformal/pool/prereg hooks)
  - headline: run steps 1-4 via scripts/run_full_sweep.py
  - aggregate: collect headline results.parquet files
  - scod: run SCOD diagnostics across available hidden-state caches
  - audit: run y_expert semantic audit across available hidden-state caches
  - selector: run SCOD-optimal selector comparison across available caches
  - ood: run ID/OOD analogy diagnostics, if you want the stretch framing

The script is intentionally conservative:
  - --dry-run prints commands without running them
  - cache-dependent tasks skip cells without hidden_states.pt unless --no-skip-missing
  - status is written to results/main_matrix_status.parquet

Examples:
  python scripts/run_main_matrix.py --tasks p0 --dry-run
  python scripts/run_main_matrix.py --tasks scod audit --models llama3_8b mistral_7b
  python scripts/run_main_matrix.py --tasks headline --seeds 0 1 2 --skip-extract
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.paths import hidden_states_path  # noqa: E402


DEFAULT_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DEFAULT_DATASETS = [
    "chaosnli",
    "pavlick_nli",
    "ambigqa_kge2",
    "pubmedqa",
    "truthfulqa",
    "medqa",
]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TASKS = ["p0"]
TASK_ALIASES = {
    "p0": ["headline", "aggregate", "scod", "audit"],
    "p1": ["selector", "ood"],
    "all": ["headline", "aggregate", "scod", "audit", "selector", "ood"],
}


@dataclass
class TaskResult:
    task: str
    model: str
    dataset: str
    seed: str
    status: str
    seconds: float
    command: str
    message: str


def experiment_config_path(model: str, dataset: str) -> Path:
    """Return an existing or generated config path for one model/dataset cell."""
    config_dir = _PROJECT_ROOT / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{model}_{dataset}.yaml"
    if path.exists():
        return path
    path.write_text(
        "extends:\n"
        f"  - models/{model}\n"
        f"  - datasets/{dataset}\n"
        "seed: 0\n\n"
        "ova_train:\n"
        "  lr: 1.0e-3\n"
        "  weight_decay: 1.0e-4\n"
        "  epochs: 30\n"
        "  batch_size: 256\n"
        "  balance_classes: true\n\n"
        "conformal:\n"
        "  alpha: 0.10\n"
        "  cal_frac: 0.5\n"
    )
    return path


def run_command(
    task: str,
    cmd: list[str],
    *,
    model: str = "",
    dataset: str = "",
    seed: str = "",
    dry_run: bool = False,
) -> TaskResult:
    command_str = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {command_str}")
        return TaskResult(task, model, dataset, seed, "dry_run", 0.0, command_str, "")

    print(f"[run] {command_str}")
    start = time.time()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    seconds = time.time() - start
    status = "ok" if proc.returncode == 0 else "failed"
    return TaskResult(
        task=task,
        model=model,
        dataset=dataset,
        seed=seed,
        status=status,
        seconds=seconds,
        command=command_str,
        message=f"returncode={proc.returncode}",
    )


def cache_missing_result(task: str, model: str, dataset: str, seed: str = "") -> TaskResult:
    path = hidden_states_path(model, dataset)
    msg = f"missing hidden states at {path}"
    print(f"[skip] {task} {model}/{dataset}: {msg}")
    return TaskResult(task, model, dataset, seed, "skipped_missing_cache", 0.0, "", msg)


def has_hidden_states(model: str, dataset: str) -> bool:
    return hidden_states_path(model, dataset).exists()


def headline_commands(args: argparse.Namespace) -> list[list[str]]:
    cmd = [
        sys.executable,
        "scripts/run_full_sweep.py",
        "--models",
        *args.models,
        "--datasets",
        *args.datasets,
        "--seeds",
        *[str(s) for s in args.seeds],
    ]
    if args.skip_extract:
        cmd.append("--skip-extract")
    return [cmd]


def aggregate_commands() -> list[list[str]]:
    return [[sys.executable, "scripts/aggregate_results.py"]]


def scod_command(model: str, datasets: list[str], seed: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/scod_diagnostics.py",
        "--model",
        model,
        "--datasets",
        *datasets,
        "--seed",
        str(seed),
        "--out",
        str(out_dir / f"scod_diagnostics__{model}__seed{seed}.parquet"),
    ]


def audit_command(model: str, dataset: str, seeds: list[int], out_dir: Path) -> list[str]:
    cfg = experiment_config_path(model, dataset)
    return [
        sys.executable,
        "scripts/y_expert_semantic_audit.py",
        "--config",
        str(cfg),
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--out-dir",
        str(out_dir),
    ]


def selector_command(model: str, dataset: str, seed: int, out_dir: Path) -> list[str]:
    cfg = experiment_config_path(model, dataset)
    return [
        sys.executable,
        "scripts/scod_optimal_selector.py",
        "--config",
        str(cfg),
        "--seed",
        str(seed),
        "--out",
        str(out_dir / f"scod_selector__{model}__{dataset}__seed{seed}.parquet"),
    ]


def ood_command(model: str, train_dataset: str, eval_datasets: list[str], seed: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/id_ood_analogy.py",
        "--model",
        model,
        "--train-dataset",
        train_dataset,
        "--eval-datasets",
        *eval_datasets,
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]


def expand_tasks(tasks: list[str]) -> list[str]:
    out: list[str] = []
    for task in tasks:
        expanded = TASK_ALIASES.get(task, [task])
        for item in expanded:
            if item not in out:
                out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main tasks across model/dataset matrix.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        choices=[
                            "p0", "p1", "all",
                            "headline", "aggregate", "scod", "audit", "selector", "ood",
                        ])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=Path("results/main_matrix"))
    parser.add_argument("--status-out", type=Path, default=Path("results/main_matrix_status.parquet"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Forwarded to headline/run_full_sweep.py.")
    parser.add_argument("--no-skip-missing", action="store_true",
                        help="Run cache-dependent commands even when hidden_states.pt is missing.")
    parser.add_argument("--audit-datasets", nargs="+", default=None,
                        help="Restrict y_expert audit datasets. Defaults to --datasets.")
    parser.add_argument("--ood-train-datasets", nargs="+", default=None,
                        help="Restrict OOD train datasets. Defaults to --datasets.")
    args = parser.parse_args()
    tasks = expand_tasks(args.tasks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    seed0 = args.seeds[0]
    results: list[TaskResult] = []

    if "headline" in tasks:
        for cmd in headline_commands(args):
            results.append(run_command("headline", cmd, seed=",".join(map(str, args.seeds)), dry_run=args.dry_run))

    if "aggregate" in tasks:
        for cmd in aggregate_commands():
            results.append(run_command("aggregate", cmd, dry_run=args.dry_run))

    if "scod" in tasks:
        scod_out = args.out_dir / "scod"
        scod_out.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            present = [d for d in args.datasets if has_hidden_states(model, d)]
            missing = [d for d in args.datasets if d not in present]
            for dataset in missing:
                if not args.no_skip_missing:
                    results.append(cache_missing_result("scod", model, dataset, str(seed0)))
            run_datasets = args.datasets if args.no_skip_missing else present
            if run_datasets:
                cmd = scod_command(model, run_datasets, seed0, scod_out)
                results.append(run_command("scod", cmd, model=model, dataset=",".join(run_datasets),
                                          seed=str(seed0), dry_run=args.dry_run))

    if "audit" in tasks:
        audit_out = args.out_dir / "y_expert_audit"
        audit_datasets = args.audit_datasets or args.datasets
        for model in args.models:
            for dataset in audit_datasets:
                if not args.no_skip_missing and not has_hidden_states(model, dataset):
                    results.append(cache_missing_result("audit", model, dataset, ",".join(map(str, args.seeds))))
                    continue
                cmd = audit_command(model, dataset, args.seeds, audit_out)
                results.append(run_command("audit", cmd, model=model, dataset=dataset,
                                          seed=",".join(map(str, args.seeds)), dry_run=args.dry_run))

    if "selector" in tasks:
        selector_out = args.out_dir / "scod_selector"
        selector_out.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            for dataset in args.datasets:
                if not args.no_skip_missing and not has_hidden_states(model, dataset):
                    results.append(cache_missing_result("selector", model, dataset, str(seed0)))
                    continue
                cmd = selector_command(model, dataset, seed0, selector_out)
                results.append(run_command("selector", cmd, model=model, dataset=dataset,
                                          seed=str(seed0), dry_run=args.dry_run))

    if "ood" in tasks:
        ood_out = args.out_dir / "id_ood_analogy"
        train_datasets = args.ood_train_datasets or args.datasets
        for model in args.models:
            present_eval = [d for d in args.datasets if has_hidden_states(model, d)]
            for train_dataset in train_datasets:
                if not args.no_skip_missing and not has_hidden_states(model, train_dataset):
                    results.append(cache_missing_result("ood", model, train_dataset, str(seed0)))
                    continue
                eval_datasets = args.datasets if args.no_skip_missing else present_eval
                cmd = ood_command(model, train_dataset, eval_datasets, seed0, ood_out)
                results.append(run_command("ood", cmd, model=model, dataset=train_dataset,
                                          seed=str(seed0), dry_run=args.dry_run))

    status = pd.DataFrame([asdict(r) for r in results])
    status.to_parquet(args.status_out, index=False)
    print(f"\nWrote {args.status_out} ({len(status)} rows)")

    if not status.empty:
        print(status.groupby(["task", "status"]).size().to_string())

    failed = status["status"].eq("failed").any() if not status.empty else False
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
