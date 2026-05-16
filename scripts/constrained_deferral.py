"""
Binary commit/defer decisions via constrained linear programming.

Primary variant:
  - Γ-Maximin robust LP using lower credal bounds on p_pred and p_amb.

Appendix variant:
  - Expected-cost LP using point estimates.

The module is intentionally standalone: it can be smoke-tested on synthetic
data, then called from the evaluation pipeline once probe outputs are available.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class DeferralCosts:
    """Per-action cost structure."""

    c_wrong: float = 1.0
    c_defer: float = 0.3

    def __post_init__(self) -> None:
        if self.c_wrong <= 0 or self.c_defer <= 0:
            raise ValueError("c_wrong and c_defer must be positive")
        if self.c_defer >= self.c_wrong:
            print(
                f"WARNING: c_defer ({self.c_defer}) >= c_wrong ({self.c_wrong}). "
                "Deferral may dominate under this cost structure."
            )


@dataclass
class CoverageBudgets:
    """System-level LP constraints."""

    kappa_min: float = 0.30
    kappa_max: float = 0.95
    alpha: float = 0.10

    def __post_init__(self) -> None:
        if not 0 <= self.kappa_min <= self.kappa_max <= 1:
            raise ValueError("Require 0 <= kappa_min <= kappa_max <= 1")
        if not 0 < self.alpha < 1:
            raise ValueError("Require 0 < alpha < 1")


@dataclass
class SolverConfig:
    """LP solver settings."""

    primary: str = "ECOS"
    fallbacks: tuple[str, ...] = ("CLARABEL", "SCS")
    verbose: bool = False
    eps: float = 1e-8


@dataclass
class LPSolution:
    """Output of solve_robust_lp / solve_expected_lp."""

    policy: np.ndarray
    actions: np.ndarray
    objective_value: float
    dual_thresholds: dict[str, float]
    constraint_slack: dict[str, float]
    solver_status: str
    solver_used: str
    n_fractional: int
    coverage_realized: float
    committed_acc_realized: float

    def summary(self) -> dict[str, float | int | str]:
        return {
            "objective_value": self.objective_value,
            "solver_status": self.solver_status,
            "solver_used": self.solver_used,
            "n_fractional": self.n_fractional,
            "coverage_realized": self.coverage_realized,
            "committed_acc_realized": self.committed_acc_realized,
            **{f"dual_{k}": v for k, v in self.dual_thresholds.items()},
            **{f"slack_{k}": v for k, v in self.constraint_slack.items()},
        }


def _as_prob_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    if ((arr < -1e-8) | (arr > 1 + 1e-8)).any():
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def _solve_lp(
    p_pred_eff: np.ndarray,
    p_amb_eff: np.ndarray,
    costs: DeferralCosts,
    budgets: CoverageBudgets,
    cfg: SolverConfig,
) -> LPSolution:
    """Generic LP solver used by robust and expected variants."""
    p_pred_eff = _as_prob_vector(p_pred_eff, "p_pred_eff")
    p_amb_eff = _as_prob_vector(p_amb_eff, "p_amb_eff")
    if p_pred_eff.shape != p_amb_eff.shape:
        raise ValueError(f"shape mismatch: {p_pred_eff.shape} vs {p_amb_eff.shape}")

    n = len(p_pred_eff)
    if n == 0:
        raise ValueError("LP needs at least one item")

    pi = cp.Variable(n)
    cost_commit = costs.c_wrong * (1.0 - p_pred_eff)
    cost_defer = costs.c_defer * (1.0 - p_amb_eff)

    objective = cp.Minimize(
        cp.sum(cp.multiply(pi, cost_commit) + cp.multiply(1.0 - pi, cost_defer))
    )

    coverage_lower = cp.sum(pi) >= budgets.kappa_min * n
    coverage_upper = cp.sum(pi) <= budgets.kappa_max * n
    conformal = cp.sum(cp.multiply(pi, p_pred_eff - (1.0 - budgets.alpha))) >= 0
    constraints = [pi >= 0, pi <= 1, coverage_lower, coverage_upper, conformal]

    problem = cp.Problem(objective, constraints)
    solvers_to_try = (cfg.primary, *cfg.fallbacks)
    last_err = None
    solver_used = None
    for solver_name in solvers_to_try:
        try:
            problem.solve(solver=solver_name, verbose=cfg.verbose)
            if problem.status in ("optimal", "optimal_inaccurate") and pi.value is not None:
                solver_used = solver_name
                break
            last_err = f"{solver_name} status={problem.status}"
        except Exception as exc:  # noqa: BLE001 - fallback chain should catch solver-specific failures.
            last_err = f"{solver_name} raised {type(exc).__name__}: {exc}"

    if solver_used is None:
        raise RuntimeError(
            f"LP did not solve with {solvers_to_try}. Last error: {last_err}"
        )

    policy = np.clip(np.asarray(pi.value, dtype=np.float64), 0.0, 1.0)
    actions = (policy > 0.5).astype(np.int64)
    committed_mass = float(policy.sum())
    committed_acc = (
        float((policy * p_pred_eff).sum() / committed_mass)
        if committed_mass > cfg.eps
        else float("nan")
    )

    return LPSolution(
        policy=policy,
        actions=actions,
        objective_value=float(problem.value),
        dual_thresholds={
            "coverage_lower": float(coverage_lower.dual_value or 0.0),
            "coverage_upper": float(coverage_upper.dual_value or 0.0),
            "conformal": float(conformal.dual_value or 0.0),
        },
        constraint_slack={
            "coverage_lower": float(policy.sum() - budgets.kappa_min * n),
            "coverage_upper": float(budgets.kappa_max * n - policy.sum()),
            "conformal": float((policy * (p_pred_eff - (1.0 - budgets.alpha))).sum()),
        },
        solver_status=str(problem.status),
        solver_used=str(solver_used),
        n_fractional=int(((policy > 0.01) & (policy < 0.99)).sum()),
        coverage_realized=float(policy.mean()),
        committed_acc_realized=committed_acc,
    )


def solve_robust_lp(
    p_pred_lower: np.ndarray,
    p_amb_lower: np.ndarray,
    costs: DeferralCosts = DeferralCosts(),
    budgets: CoverageBudgets = CoverageBudgets(),
    cfg: SolverConfig = SolverConfig(),
) -> LPSolution:
    """Γ-Maximin robust LP using lower credal bounds."""
    return _solve_lp(p_pred_lower, p_amb_lower, costs, budgets, cfg)


def solve_expected_lp(
    p_pred: np.ndarray,
    p_amb: np.ndarray,
    costs: DeferralCosts = DeferralCosts(),
    budgets: CoverageBudgets = CoverageBudgets(),
    cfg: SolverConfig = SolverConfig(),
) -> LPSolution:
    """Expected-cost LP using point estimates."""
    return _solve_lp(p_pred, p_amb, costs, budgets, cfg)


def bootstrap_credal_intervals(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_eval: np.ndarray,
    k: int = 20,
    probe_c: float = 1.0,
    class_weight: str | dict | None = "balanced",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-item bootstrap [min, mean, max] probe probabilities."""
    h_train = np.asarray(h_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    h_eval = np.asarray(h_eval, dtype=np.float64)
    if h_train.ndim != 2 or h_eval.ndim != 2:
        raise ValueError("h_train and h_eval must be 2D")
    if len(h_train) != len(y_train):
        raise ValueError("h_train/y_train length mismatch")
    if len(np.unique(y_train)) < 2:
        raise ValueError("bootstrap probes require both classes")

    rng = np.random.default_rng(seed)
    all_probs = np.zeros((k, len(h_eval)), dtype=np.float64)
    for j in range(k):
        idx = rng.choice(len(h_train), size=len(h_train), replace=True)
        probe = LogisticRegression(
            C=probe_c,
            class_weight=class_weight,
            max_iter=1000,
            random_state=int(seed + j),
        )
        probe.fit(h_train[idx], y_train[idx])
        all_probs[j] = probe.predict_proba(h_eval)[:, 1]
    return all_probs.min(axis=0), all_probs.mean(axis=0), all_probs.max(axis=0)


def bootstrap_or_constant_intervals(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_eval: np.ndarray,
    k: int = 20,
    probe_c: float = 1.0,
    class_weight: str | dict | None = "balanced",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap intervals, with a deterministic fallback for one-class labels."""
    y_train = np.asarray(y_train, dtype=np.int64)
    uniq = np.unique(y_train)
    if len(uniq) == 1:
        p = np.full(len(h_eval), float(uniq[0]), dtype=np.float64)
        return p, p, p
    return bootstrap_credal_intervals(
        h_train,
        y_train,
        h_eval,
        k=k,
        probe_c=probe_c,
        class_weight=class_weight,
        seed=seed,
    )


def apply_policy_to_test(
    p_pred_test_eff: np.ndarray,
    p_amb_test_eff: np.ndarray,
    duals: dict[str, float],
    costs: DeferralCosts,
    budgets: CoverageBudgets,
) -> dict[str, np.ndarray]:
    """Apply calibration LP dual variables to test items via reduced costs."""
    p_pred_test_eff = _as_prob_vector(p_pred_test_eff, "p_pred_test_eff")
    p_amb_test_eff = _as_prob_vector(p_amb_test_eff, "p_amb_test_eff")
    lam_lo = duals.get("coverage_lower", 0.0)
    lam_hi = duals.get("coverage_upper", 0.0)
    lam_conf = duals.get("conformal", 0.0)

    rc_commit = (
        costs.c_wrong * (1.0 - p_pred_test_eff)
        - lam_lo
        + lam_hi
        - lam_conf * (p_pred_test_eff - (1.0 - budgets.alpha))
    )
    rc_defer = costs.c_defer * (1.0 - p_amb_test_eff)
    actions = (rc_commit < rc_defer).astype(np.int64)
    score_diff = np.clip(rc_defer - rc_commit, -50, 50)
    policy_soft = 1.0 / (1.0 + np.exp(-score_diff))
    return {
        "actions": actions,
        "commit_score": -rc_commit,
        "defer_score": -rc_defer,
        "policy": policy_soft,
    }


def sanity_checks(
    sol: LPSolution,
    p_pred_eff: np.ndarray,
    budgets: CoverageBudgets,
    verbose: bool = True,
) -> dict[str, dict[str, float | bool | str]]:
    """Run post-hoc checks on an LP solution."""
    p_pred_eff = _as_prob_vector(p_pred_eff, "p_pred_eff")
    n = len(sol.policy)
    conf_value = float((sol.policy * (p_pred_eff - (1.0 - budgets.alpha))).sum())
    frac_ratio = sol.n_fractional / n if n else 0.0
    target_acc = 1.0 - budgets.alpha

    results = {
        "coverage_within_bounds": {
            "pass": bool(budgets.kappa_min - 1e-3 <= sol.coverage_realized <= budgets.kappa_max + 1e-3),
            "value": sol.coverage_realized,
            "msg": f"coverage={sol.coverage_realized:.4f}, target=[{budgets.kappa_min}, {budgets.kappa_max}]",
        },
        "conformal_satisfied": {
            "pass": bool(conf_value >= -1e-3),
            "value": conf_value,
            "msg": f"sum(pi * (p - (1-alpha)))={conf_value:.4f}",
        },
        "committed_accuracy": {
            "pass": bool(sol.committed_acc_realized >= target_acc - 1e-2),
            "value": sol.committed_acc_realized,
            "msg": f"committed_acc={sol.committed_acc_realized:.4f}, target>={target_acc:.4f}",
        },
        "few_fractional": {
            "pass": bool(frac_ratio < 0.10),
            "value": frac_ratio,
            "msg": f"fractional={sol.n_fractional}/{n} ({frac_ratio:.4f})",
        },
        "dual_sign_conformal": {
            "pass": bool(sol.dual_thresholds["conformal"] >= -1e-6),
            "value": sol.dual_thresholds["conformal"],
            "msg": f"conformal_dual={sol.dual_thresholds['conformal']:.6f}",
        },
    }
    if verbose:
        print(f"\n=== Sanity checks ({sol.solver_used}, {sol.solver_status}) ===")
        for name, result in results.items():
            mark = "PASS" if result["pass"] else "FAIL"
            print(f"  {mark:4s} {name:24s} {result['msg']}")
    return results


def run_constrained_deferral(
    p_pred_cal: np.ndarray,
    p_amb_cal: np.ndarray,
    p_pred_test: np.ndarray,
    p_amb_test: np.ndarray,
    p_pred_cal_lower: Optional[np.ndarray] = None,
    p_amb_cal_lower: Optional[np.ndarray] = None,
    p_pred_test_lower: Optional[np.ndarray] = None,
    p_amb_test_lower: Optional[np.ndarray] = None,
    costs: DeferralCosts = DeferralCosts(),
    budgets: CoverageBudgets = CoverageBudgets(),
    verbose: bool = True,
) -> dict[str, dict[str, object]]:
    """Solve expected and, if bounds are provided, robust LPs."""
    out: dict[str, dict[str, object]] = {}
    if verbose:
        print("\n========== EXPECTED-COST LP ==========")
    sol_exp = solve_expected_lp(p_pred_cal, p_amb_cal, costs, budgets)
    sanity_checks(sol_exp, p_pred_cal, budgets, verbose=verbose)
    out["expected"] = {
        "calibration": sol_exp,
        "test": apply_policy_to_test(
            p_pred_test, p_amb_test, sol_exp.dual_thresholds, costs, budgets
        ),
    }

    have_bounds = all(
        x is not None
        for x in (p_pred_cal_lower, p_amb_cal_lower, p_pred_test_lower, p_amb_test_lower)
    )
    if have_bounds:
        if verbose:
            print("\n========== ROBUST LP ==========")
        try:
            sol_rob = solve_robust_lp(p_pred_cal_lower, p_amb_cal_lower, costs, budgets)
        except RuntimeError as exc:
            out["robust_error"] = {
                "error": str(exc),
                "max_robust_coverage_at_target": max_feasible_coverage(
                    p_pred_cal_lower,
                    target_accuracy=1.0 - budgets.alpha,
                ),
            }
            if verbose:
                print(f"Robust LP infeasible: {exc}")
                print(
                    "  max robust coverage at target accuracy "
                    f"{1.0 - budgets.alpha:.3f}: "
                    f"{out['robust_error']['max_robust_coverage_at_target']:.4f}"
                )
            return out
        sanity_checks(sol_rob, p_pred_cal_lower, budgets, verbose=verbose)
        out["robust"] = {
            "calibration": sol_rob,
            "test": apply_policy_to_test(
                p_pred_test_lower,
                p_amb_test_lower,
                sol_rob.dual_thresholds,
                costs,
                budgets,
            ),
        }
        if verbose:
            agree = (
                out["robust"]["test"]["actions"] == out["expected"]["test"]["actions"]
            ).mean()
            print("\n========== COMPARISON ==========")
            print(f"  action agreement={agree:.4f}")
    elif verbose:
        print("\nSkipping robust LP: credal lower bounds were not supplied.")
    return out


def max_feasible_coverage(p_pred_eff: np.ndarray, target_accuracy: float) -> float:
    """Maximum coverage achievable while average p_pred_eff >= target_accuracy."""
    p = np.sort(_as_prob_vector(p_pred_eff, "p_pred_eff"))[::-1]
    if len(p) == 0:
        return 0.0
    cum_mean = np.cumsum(p) / np.arange(1, len(p) + 1)
    ok = np.where(cum_mean >= target_accuracy)[0]
    if len(ok) == 0:
        return 0.0
    return float((ok[-1] + 1) / len(p))


def _summarize_results(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for variant, blob in results.items():
        if variant.endswith("_error"):
            rows.append({"variant": variant, **blob})
            continue
        sol = blob["calibration"]
        if isinstance(sol, LPSolution):
            rows.append({"variant": variant, **sol.summary()})
    return pd.DataFrame(rows)


def _smoke_test() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_cal, n_test = 5000, 1000
    p_pred_cal = rng.beta(14, 1.5, size=n_cal)
    p_amb_cal = rng.beta(5, 3, size=n_cal)
    p_pred_test = rng.beta(14, 1.5, size=n_test)
    p_amb_test = rng.beta(5, 3, size=n_test)
    width = 0.05
    p_pred_cal_lower = np.clip(p_pred_cal - width * rng.uniform(size=n_cal), 0, 1)
    p_amb_cal_lower = np.clip(p_amb_cal - width * rng.uniform(size=n_cal), 0, 1)
    p_pred_test_lower = np.clip(p_pred_test - width * rng.uniform(size=n_test), 0, 1)
    p_amb_test_lower = np.clip(p_amb_test - width * rng.uniform(size=n_test), 0, 1)
    results = run_constrained_deferral(
        p_pred_cal,
        p_amb_cal,
        p_pred_test,
        p_amb_test,
        p_pred_cal_lower,
        p_amb_cal_lower,
        p_pred_test_lower,
        p_amb_test_lower,
        budgets=CoverageBudgets(kappa_min=0.30, kappa_max=0.95, alpha=0.10),
        verbose=True,
    )
    df = _summarize_results(results)
    print("\n========== SUMMARY ==========")
    print(df.to_string(index=False))
    return df


def _demo_from_signals(
    model: str,
    dataset: str,
    seed: int,
    synthetic_width: float,
) -> pd.DataFrame:
    from gap.signal import load_signals
    from utils.paths import signals_path

    sig = load_signals(signals_path(model, dataset, seed))
    p_pred = sig.sigma_pred.astype(float)
    p_amb = sig.sigma_defer.astype(float)
    n = len(p_pred)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = n // 2
    cal_idx, test_idx = idx[:split], idx[split:]
    pred_lower = np.clip(p_pred - synthetic_width, 0, 1)
    amb_lower = np.clip(p_amb - synthetic_width, 0, 1)
    results = run_constrained_deferral(
        p_pred[cal_idx],
        p_amb[cal_idx],
        p_pred[test_idx],
        p_amb[test_idx],
        pred_lower[cal_idx],
        amb_lower[cal_idx],
        pred_lower[test_idx],
        amb_lower[test_idx],
        verbose=True,
    )
    df = _summarize_results(results)
    df.insert(0, "dataset", dataset)
    df.insert(0, "model", model)
    print("\n========== SIGNALS DEMO SUMMARY ==========")
    print(df.to_string(index=False))
    return df


def run_bootstrap_features(
    model: str,
    dataset: str,
    seed: int,
    k_boot: int,
    item_out: Path | None = None,
    cache_dir: Path | None = None,
    force_cache: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    import torch

    from data.registry import load_dataset
    from utils.paths import hidden_states_path

    h_blob = torch.load(hidden_states_path(model, dataset), map_location="cpu", weights_only=False)
    h_pos = np.asarray(h_blob["h_pos"], dtype=np.float64)
    h_neg = np.asarray(h_blob["h_neg"], dtype=np.float64)
    example_ids = np.asarray(h_blob["example_ids"])
    n = len(h_pos)

    records = load_dataset(dataset)
    er_by_id = {r.example_id: r.expert_reliable for r in records}
    expert_reliable = np.asarray([er_by_id[eid] for eid in example_ids], dtype=np.int64)

    h_all = np.concatenate([h_pos, h_neg], axis=0)
    y_pred = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    y_amb = np.concatenate([expert_reliable, expert_reliable]).astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(h_all))
    split = len(h_all) // 2
    cal_idx, test_idx = idx[:split], idx[split:]

    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / model / dataset / f"seed{seed}" / f"bootstrap_k{k_boot}.npz"

    if cache_path is not None and cache_path.exists() and not force_cache:
        blob = np.load(cache_path, allow_pickle=True)
        p_pred_lower_all = blob["p_pred_lower"]
        p_pred_mean_all = blob["p_pred_mean"]
        p_pred_upper_all = blob["p_pred_upper"]
        p_amb_lower_all = blob["p_amb_lower"]
        p_amb_mean_all = blob["p_amb_mean"]
        p_amb_upper_all = blob["p_amb_upper"]
        cal_idx = blob["cal_idx"]
        test_idx = blob["test_idx"]
        if verbose:
            print(f"Loaded cached bootstrap intervals from {cache_path}")
    else:
        if verbose:
            print(
                f"Training bootstrap intervals: K={k_boot}, train={len(h_all)}, "
                f"cal={len(cal_idx)}, test={len(test_idx)}"
            )
        p_pred_lower_all, p_pred_mean_all, p_pred_upper_all = bootstrap_or_constant_intervals(
            h_all, y_pred, h_all, k=k_boot, seed=seed,
        )
        p_amb_lower_all, p_amb_mean_all, p_amb_upper_all = bootstrap_or_constant_intervals(
            h_all, y_amb, h_all, k=k_boot, seed=seed + 10_000,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                p_pred_lower=p_pred_lower_all,
                p_pred_mean=p_pred_mean_all,
                p_pred_upper=p_pred_upper_all,
                p_amb_lower=p_amb_lower_all,
                p_amb_mean=p_amb_mean_all,
                p_amb_upper=p_amb_upper_all,
                cal_idx=cal_idx,
                test_idx=test_idx,
                y_pred=y_pred,
                y_amb=y_amb,
                example_ids=example_ids,
            )
            if verbose:
                print(f"Wrote bootstrap interval cache to {cache_path}")

    p_pred_cal_lower, p_pred_cal, p_pred_cal_upper = (
        p_pred_lower_all[cal_idx],
        p_pred_mean_all[cal_idx],
        p_pred_upper_all[cal_idx],
    )
    p_pred_test_lower, p_pred_test, p_pred_test_upper = (
        p_pred_lower_all[test_idx],
        p_pred_mean_all[test_idx],
        p_pred_upper_all[test_idx],
    )
    p_amb_cal_lower, p_amb_cal, p_amb_cal_upper = (
        p_amb_lower_all[cal_idx],
        p_amb_mean_all[cal_idx],
        p_amb_upper_all[cal_idx],
    )
    p_amb_test_lower, p_amb_test, p_amb_test_upper = (
        p_amb_lower_all[test_idx],
        p_amb_mean_all[test_idx],
        p_amb_upper_all[test_idx],
    )

    if verbose:
        print(
            "Interval diagnostics: "
            f"pred_width_mean={np.mean(p_pred_cal - p_pred_cal_lower):.4f}, "
            f"amb_width_mean={np.mean(p_amb_cal - p_amb_cal_lower):.4f}"
        )
    results = run_constrained_deferral(
        p_pred_cal,
        p_amb_cal,
        p_pred_test,
        p_amb_test,
        p_pred_cal_lower,
        p_amb_cal_lower,
        p_pred_test_lower,
        p_amb_test_lower,
        verbose=verbose,
    )
    df = _summarize_results(results)
    df.insert(0, "k_boot", k_boot)
    df.insert(0, "dataset", dataset)
    df.insert(0, "model", model)

    if item_out is not None:
        expected = results["expected"]["test"]
        robust = results.get("robust", {}).get("test")
        item_df = pd.DataFrame({
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "k_boot": k_boot,
            "row_idx": test_idx,
            "example_id": np.asarray(h_blob["example_ids"])[test_idx % n],
            "is_pos": test_idx < n,
            "y_correct": y_pred[test_idx].astype(int),
            "p_pred_lower": p_pred_test_lower,
            "p_pred_mean": p_pred_test,
            "p_pred_upper": p_pred_test_upper,
            "p_amb_lower": p_amb_test_lower,
            "p_amb_mean": p_amb_test,
            "p_amb_upper": p_amb_test_upper,
            "policy_expected": expected["policy"],
            "action_expected": expected["actions"],
            "commit_score_expected": expected["commit_score"],
        })
        if robust is not None:
            item_df["policy_robust"] = robust["policy"]
            item_df["action_robust"] = robust["actions"]
            item_df["commit_score_robust"] = robust["commit_score"]
        else:
            item_df["policy_robust"] = np.nan
            item_df["action_robust"] = np.nan
            item_df["commit_score_robust"] = np.nan
        item_out.parent.mkdir(parents=True, exist_ok=True)
        item_df.to_parquet(item_out, index=False)
        print(f"Wrote item diagnostics to {item_out}")

    if verbose:
        print("\n========== BOOTSTRAP FEATURE SUMMARY ==========")
        print(df.to_string(index=False))
    return df


def _demo_bootstrap_features(
    model: str,
    dataset: str,
    seed: int,
    k_boot: int,
    item_out: Path | None = None,
    cache_dir: Path | None = None,
    force_cache: bool = False,
) -> pd.DataFrame:
    return run_bootstrap_features(
        model,
        dataset,
        seed,
        k_boot,
        item_out=item_out,
        cache_dir=cache_dir,
        force_cache=force_cache,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained deferral LP smoke/demo runner.")
    parser.add_argument("--demo-signals", action="store_true")
    parser.add_argument("--demo-bootstrap-features", action="store_true")
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--dataset", default="truthfulqa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-boot", type=int, default=20)
    parser.add_argument("--synthetic-width", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--item-out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/constrained_deferral_cache"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    if args.demo_bootstrap_features:
        item_out = args.item_out
        if item_out is None and args.out is not None:
            item_out = args.out.with_name(f"{args.out.stem}_items{args.out.suffix}")
        df = _demo_bootstrap_features(
            args.model,
            args.dataset,
            args.seed,
            args.k_boot,
            item_out=item_out,
            cache_dir=args.cache_dir,
            force_cache=args.force_cache,
        )
    elif args.demo_signals:
        df = _demo_from_signals(args.model, args.dataset, args.seed, args.synthetic_width)
    else:
        df = _smoke_test()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
