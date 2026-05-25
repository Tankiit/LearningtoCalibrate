"""Run calibration analyses over saved probe signal artifacts."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from eval.calibration import CalibrationResult, evaluate, run_calibration_suite


MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]

AMBIGUITY_TYPE: dict[str, str] = {
    "chaosnli": "ambiguous",
    "pavlick_nli": "ambiguous",
    "ambigqa_kge2": "ambiguous",
    "pubmedqa": "medical",
    "medqa": "medical",
    "truthfulqa": "unambiguous",
    "halueval_qa": "unambiguous",
    "triviaqa": "unambiguous",
    "popqa": "unambiguous",
}
DATASETS = list(AMBIGUITY_TYPE.keys())
SEEDS = [0, 1, 2]
CAL_FRACTION = 0.20


VERBALIZED_KEYS = ("verbalized", "verbalized_probs", "verbalized_confidence", "confidence")


def load_signals(signals_path: Path) -> Optional[dict[str, np.ndarray]]:
    """Load one signals.pt file and return calibration-ready arrays."""
    if not signals_path.exists():
        return None

    raw = torch.load(signals_path, map_location="cpu", weights_only=False)
    if "logit_pred" not in raw:
        print(f"  [WARN] {signals_path}: missing key 'logit_pred'")
        return None

    label_key = "y_correct" if "y_correct" in raw else "is_pos" if "is_pos" in raw else None
    if label_key is None:
        print(f"  [WARN] {signals_path}: missing label key 'y_correct' or 'is_pos'")
        return None

    signals = {
        "logit_pred": _to_numpy(raw["logit_pred"], dtype=float),
        "y_correct": _to_numpy(raw[label_key], dtype=float),
    }
    if "example_ids" in raw:
        signals["example_ids"] = _to_numpy(raw["example_ids"], dtype=str)
    for key in (
        "sigma_pred",
        "logit_defer",
        "sigma_defer",
        "gap",
        "logprob",
        "beta",
        "beta_valid",
        "semantic_entropy",
        "sem_entropy",
        "entropy",
        *VERBALIZED_KEYS,
    ):
        if key in raw:
            signals["verbalized" if key in VERBALIZED_KEYS else key] = _to_numpy(
                raw[key], dtype=float
            )

    if "verbalized" not in signals:
        sidecar = load_verbalized_sidecar(signals_path, raw)
        if sidecar is not None:
            signals["verbalized"] = sidecar

    n = len(signals["logit_pred"])
    bad_lengths = {key: len(value) for key, value in signals.items() if len(value) != n}
    if bad_lengths:
        print(f"  [WARN] {signals_path}: length mismatch {bad_lengths}, expected {n}")
        return None

    return signals


def load_verbalized_sidecar(
    signals_path: Path,
    signals_raw: dict,
    verbalized_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load optional verbalized confidence sidecars aligned to signals.pt rows.

    Supported locations, checked in order:
      - explicit verbalized_path if provided
      - beside signals.pt: verbalized.{pt,csv,parquet,jsonl}
      - dataset-level dir: ../verbalized.{pt,csv,parquet,jsonl}

    Supported schemas:
      - one array under verbalized/verbalized_probs/verbalized_confidence/confidence
      - *_pos and *_neg arrays plus example_ids
      - table rows with example_id, optional is_pos, and a verbalized/confidence column
    """
    candidates = _verbalized_candidates(signals_path, verbalized_path)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            values = _read_verbalized_file(candidate, signals_raw)
        except Exception as exc:
            print(f"  [WARN] {candidate}: could not load verbalized baseline ({exc})")
            continue
        if values is not None:
            return values
    return None


def make_cal_test_split(
    signals: dict[str, np.ndarray],
    cal_fraction: float = CAL_FRACTION,
    split_seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split all signal arrays into calibration and test portions."""
    n = len(signals["logit_pred"])
    if n < 2:
        raise ValueError("need at least two examples for a calibration/test split")

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(n)
    n_cal = int(round(n * cal_fraction))
    n_cal = min(max(n_cal, min(20, n - 1)), n - 1)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]

    cal = {key: value[cal_idx] for key, value in signals.items()}
    test = {key: value[test_idx] for key, value in signals.items()}
    return cal, test


def resolve_signals_path(
    outputs_dir: Path,
    model: str,
    dataset: str,
    seed: int,
) -> Path:
    """Resolve seeded probe artifacts, with fallbacks for older layouts."""
    candidates = [
        outputs_dir / "step2_probes" / model / dataset / f"seed{seed}" / "signals.pt",
        outputs_dir / model / dataset / f"seed{seed}" / "signals.pt",
        outputs_dir / model / dataset / "signals.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def evaluate_cell(
    signals_path: Path,
    model: str,
    dataset: str,
    ambiguity_type: str,
    probe_seed: int,
    split_seed: int = 42,
    n_bins: int = 15,
    verbalized_path: Optional[Path] = None,
) -> Optional[dict[str, CalibrationResult]]:
    """Evaluate one model/dataset/seed cell."""
    signals = load_signals(signals_path)
    if signals is None:
        return None
    if verbalized_path is not None and "verbalized" not in signals:
        raw = torch.load(signals_path, map_location="cpu", weights_only=False)
        sidecar = load_verbalized_sidecar(signals_path, raw, verbalized_path)
        if sidecar is not None:
            signals["verbalized"] = sidecar

    cal, test = make_cal_test_split(signals, split_seed=split_seed)
    results = run_calibration_suite(
        logits_cal=cal["logit_pred"],
        labels_cal=cal["y_correct"],
        logits_test=test["logit_pred"],
        labels_test=test["y_correct"],
        model=model,
        dataset=dataset,
        ambiguity_type=ambiguity_type,
        verbalized_probs_test=test.get("verbalized"),
        n_bins=n_bins,
        seed=probe_seed,
    )

    if "sigma_pred" in test:
        results["sigma_pred"] = evaluate(
            probs=np.clip(test["sigma_pred"], 0.0, 1.0),
            labels=test["y_correct"],
            model=model,
            dataset=dataset,
            method="sigma_pred",
            ambiguity_type=ambiguity_type,
            n_bins=n_bins,
            seed=probe_seed,
        )

    if "logprob" in test:
        results["logprob"] = evaluate(
            probs=expit(test["logprob"]),
            labels=test["y_correct"],
            model=model,
            dataset=dataset,
            method="logprob",
            ambiguity_type=ambiguity_type,
            n_bins=n_bins,
            seed=probe_seed,
        )

    return results


def run_full_matrix(
    outputs_dir: Path,
    models: list[str] = MODELS,
    datasets: list[str] = DATASETS,
    seeds: list[int] = SEEDS,
    split_seed: int = 42,
    n_bins: int = 15,
    verbalized_dir: Optional[Path] = None,
) -> list[CalibrationResult]:
    """Sweep the model/dataset/seed matrix and collect all results."""
    all_results: list[CalibrationResult] = []
    n_cells = len(models) * len(datasets) * len(seeds)
    done = 0
    skipped = 0

    print(
        f"Running calibration matrix: {len(models)} models x "
        f"{len(datasets)} datasets x {len(seeds)} seeds"
    )
    print("-" * 72)

    for model in models:
        for dataset in datasets:
            ambiguity_type = AMBIGUITY_TYPE.get(dataset, "unambiguous")
            for seed in seeds:
                path = resolve_signals_path(outputs_dir, model, dataset, seed)
                verbalized_path = resolve_verbalized_path(verbalized_dir, model, dataset, seed)
                cell_results = evaluate_cell(
                    path,
                    model,
                    dataset,
                    ambiguity_type,
                    probe_seed=seed,
                    split_seed=split_seed,
                    n_bins=n_bins,
                    verbalized_path=verbalized_path,
                )

                if cell_results is None:
                    print(f"  SKIP  {model:14s} x {dataset:14s} seed={seed:<2d}  (no artifact)")
                    skipped += 1
                    continue

                all_results.extend(cell_results.values())
                ece_raw = cell_results["raw"].ece_15
                ece_platt = cell_results["platt"].ece_15
                print(
                    f"  OK    {model:14s} x {dataset:14s} seed={seed:<2d}  "
                    f"ECE raw={ece_raw:.3f} -> platt={ece_platt:.3f} "
                    f"[{ambiguity_type}]"
                )
                done += 1

    print("-" * 72)
    print(f"Completed {done}/{n_cells} cells ({skipped} skipped).")
    return all_results


def resolve_verbalized_path(
    verbalized_dir: Optional[Path],
    model: str,
    dataset: str,
    seed: int,
) -> Optional[Path]:
    """Resolve an explicit verbalized sidecar root, if provided."""
    if verbalized_dir is None:
        return None
    candidates = []
    for suffix in ("pt", "csv", "parquet", "jsonl"):
        candidates.extend(
            [
                verbalized_dir / model / dataset / f"seed{seed}" / f"verbalized.{suffix}",
                verbalized_dir / model / dataset / f"verbalized.{suffix}",
                verbalized_dir / f"{model}_{dataset}_seed{seed}.{suffix}",
                verbalized_dir / f"{model}_{dataset}.{suffix}",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def results_to_dataframe(results: list[CalibrationResult]) -> pd.DataFrame:
    """Convert CalibrationResult objects to a flat DataFrame."""
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("bins_equal_width")
        row.pop("bins_adaptive")
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the stratified summary table across ambiguity types."""
    return (
        df.groupby(["method", "ambiguity_type"])
        .agg(
            ece_15_mean=("ece_15", "mean"),
            ece_15_std=("ece_15", "std"),
            ece_adaptive_mean=("ece_adaptive", "mean"),
            ece_adaptive_std=("ece_adaptive", "std"),
            brier_mean=("brier_score", "mean"),
            brier_std=("brier_score", "std"),
            n_cells=("ece_15", "count"),
        )
        .reset_index()
        .sort_values(["ambiguity_type", "method"])
    )


def build_ece_reduction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Platt ECE reduction vs raw by ambiguity type."""
    raw = df[df["method"] == "raw"][
        ["model", "dataset", "seed", "ambiguity_type", "ece_15"]
    ].rename(columns={"ece_15": "ece_raw"})
    platt = df[df["method"] == "platt"][
        ["model", "dataset", "seed", "ece_15"]
    ].rename(columns={"ece_15": "ece_platt"})

    merged = raw.merge(platt, on=["model", "dataset", "seed"])
    merged["ece_reduction_abs"] = merged["ece_raw"] - merged["ece_platt"]
    merged["ece_reduction_rel"] = merged["ece_reduction_abs"] / merged["ece_raw"].clip(1e-6)

    return (
        merged.groupby("ambiguity_type")
        .agg(
            reduction_abs_mean=("ece_reduction_abs", "mean"),
            reduction_abs_std=("ece_reduction_abs", "std"),
            reduction_rel_mean=("ece_reduction_rel", "mean"),
            reduction_rel_std=("ece_reduction_rel", "std"),
            n_cells=("ece_reduction_abs", "count"),
        )
        .reset_index()
    )


def build_reliability_diagram_data(
    results: list[CalibrationResult],
    method: str = "platt",
    use_adaptive: bool = False,
) -> dict[tuple[str, str, int, str], dict[str, list[float] | float | int]]:
    """Extract reliability diagram data for one method across all cells."""
    diagram_data = {}
    for result in results:
        if result.method != method:
            continue

        bins = result.bins_adaptive if use_adaptive else result.bins_equal_width
        non_empty = [bin_stat for bin_stat in bins if bin_stat.count > 0]
        seed = -1 if result.seed is None else result.seed
        diagram_data[(result.model, result.dataset, seed, result.ambiguity_type)] = {
            "confidence": [b.mean_confidence for b in non_empty],
            "accuracy": [b.fraction_positive for b in non_empty],
            "weight": [b.weight for b in non_empty],
            "count": [b.count for b in non_empty],
            "ece_15": result.ece_15,
            "n_test": result.n_test,
        }
    return diagram_data


def _to_numpy(value, dtype=float) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).reshape(-1)


def _verbalized_candidates(
    signals_path: Path,
    verbalized_path: Optional[Path],
) -> list[Path]:
    if verbalized_path is not None:
        return [verbalized_path]
    candidates = []
    for base in (signals_path.parent, signals_path.parent.parent):
        for suffix in ("pt", "csv", "parquet", "jsonl"):
            candidates.append(base / f"verbalized.{suffix}")
    return candidates


def _read_verbalized_file(path: Path, signals_raw: dict) -> Optional[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            return _verbalized_from_mapping(blob, signals_raw)
        return _validate_verbalized_array(_to_numpy(blob, dtype=float), signals_raw)
    if suffix == ".csv":
        return _verbalized_from_frame(pd.read_csv(path), signals_raw)
    if suffix == ".parquet":
        return _verbalized_from_frame(pd.read_parquet(path), signals_raw)
    if suffix == ".jsonl":
        return _verbalized_from_frame(pd.read_json(path, lines=True), signals_raw)
    raise ValueError(f"unsupported extension {suffix}")


def _verbalized_from_mapping(blob: dict, signals_raw: dict) -> Optional[np.ndarray]:
    for key in VERBALIZED_KEYS:
        if key in blob:
            return _validate_verbalized_array(_to_numpy(blob[key], dtype=float), signals_raw)

    pos_key = next((key for key in ("verbalized_pos", "confidence_pos", "p_pos") if key in blob), None)
    neg_key = next((key for key in ("verbalized_neg", "confidence_neg", "p_neg") if key in blob), None)
    if pos_key is not None and neg_key is not None and "example_ids" in blob:
        return _align_pos_neg_arrays(
            _to_numpy(blob["example_ids"], dtype=str),
            _to_numpy(blob[pos_key], dtype=float),
            _to_numpy(blob[neg_key], dtype=float),
            signals_raw,
        )

    if "rows" in blob:
        return _verbalized_from_frame(pd.DataFrame(blob["rows"]), signals_raw)

    return None


def _verbalized_from_frame(df: pd.DataFrame, signals_raw: dict) -> Optional[np.ndarray]:
    value_col = next((key for key in VERBALIZED_KEYS if key in df.columns), None)
    if value_col is None:
        for key in ("probability", "prob", "p"):
            if key in df.columns:
                value_col = key
                break
    if value_col is None:
        raise ValueError(f"missing confidence column; tried {VERBALIZED_KEYS}")

    if len(df) == len(signals_raw["logit_pred"]) and "example_id" not in df.columns:
        return _validate_verbalized_array(df[value_col].to_numpy(dtype=float), signals_raw)

    if "example_id" not in df.columns:
        raise ValueError("table sidecar must include example_id unless row-aligned")

    signal_ids = np.asarray(signals_raw["example_ids"], dtype=str)
    signal_is_pos = np.asarray(signals_raw.get("is_pos", np.ones(len(signal_ids))), dtype=bool)
    values = np.empty(len(signal_ids), dtype=float)

    has_is_pos = "is_pos" in df.columns
    if has_is_pos:
        lookup = {
            (str(row.example_id), bool(row.is_pos)): float(getattr(row, value_col))
            for row in df.itertuples(index=False)
        }
        for i, (example_id, is_pos) in enumerate(zip(signal_ids, signal_is_pos)):
            values[i] = lookup[(str(example_id), bool(is_pos))]
    else:
        lookup = {
            str(row.example_id): float(getattr(row, value_col))
            for row in df.itertuples(index=False)
        }
        for i, example_id in enumerate(signal_ids):
            values[i] = lookup[str(example_id)]

    return _validate_verbalized_array(values, signals_raw)


def _align_pos_neg_arrays(
    example_ids: np.ndarray,
    pos_values: np.ndarray,
    neg_values: np.ndarray,
    signals_raw: dict,
) -> np.ndarray:
    if len(example_ids) != len(pos_values) or len(example_ids) != len(neg_values):
        raise ValueError("example_ids, *_pos, and *_neg lengths must match")
    pos_lookup = dict(zip(example_ids.astype(str), pos_values.astype(float)))
    neg_lookup = dict(zip(example_ids.astype(str), neg_values.astype(float)))
    signal_ids = np.asarray(signals_raw["example_ids"], dtype=str)
    signal_is_pos = np.asarray(signals_raw.get("is_pos", np.ones(len(signal_ids))), dtype=bool)
    values = np.empty(len(signal_ids), dtype=float)
    for i, (example_id, is_pos) in enumerate(zip(signal_ids, signal_is_pos)):
        values[i] = pos_lookup[str(example_id)] if is_pos else neg_lookup[str(example_id)]
    return _validate_verbalized_array(values, signals_raw)


def _validate_verbalized_array(values: np.ndarray, signals_raw: dict) -> np.ndarray:
    expected = len(signals_raw["logit_pred"])
    if len(values) != expected:
        raise ValueError(f"length mismatch: verbalized has {len(values)}, expected {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError("verbalized values contain non-finite entries")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("verbalized values must be probabilities in [0, 1]")
    return values
