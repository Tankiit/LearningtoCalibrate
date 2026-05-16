"""
Write the ARR showcase pre-registration artifact.

This script should be run before inspecting any showcase probe/audit results.
It records the locked method, canonical representation choice, dataset regimes,
predicted outcomes, and a content hash over the pre-registration payload.

Usage:
  python scripts/preregister_predictions.py
  python scripts/preregister_predictions.py --out PRE_REGISTRATION.json --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SHOWCASE_DATASETS = ["chaosnli", "pavlick_nli", "ambigqa_kge2"]
DIAGNOSTIC_DATASETS = ["truthfulqa", "halueval_qa"]
LOCKED_MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
LOCKED_SEEDS = [0, 1, 2, 3, 4]


def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_payload() -> dict[str, Any]:
    git_commit = _git(["rev-parse", "HEAD"])
    git_status = _git(["status", "--short"])

    payload: dict[str, Any] = {
        "artifact": "ARR May 2026 pre-registration",
        "locked": True,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "project": {
            "name": "OVA_ARR",
            "title": "LLMs Know What They Won't Say: Turning Internal-External Disagreement into Calibrated Deferral",
            "repo_path": str(PROJECT_ROOT),
            "git_commit": git_commit,
            "git_status_short_at_registration": git_status,
        },
        "scope": {
            "models": LOCKED_MODELS,
            "showcase_datasets": SHOWCASE_DATASETS,
            "diagnostic_datasets": DIAGNOSTIC_DATASETS,
            "seeds": LOCKED_SEEDS,
            "primary_metric": "AURC",
            "secondary_metrics": ["coverage_at_acc_80", "coverage_at_acc_90"],
            "lower_aurc_is_better": True,
        },
        "locked_method": {
            "headline_name": "gap",
            "definition": "raw logit-space difference: logit(f_pred) - logit(f_amb)",
            "probe_pred_baseline": "probe_pred_only = sigmoid(f_pred)",
            "probe_amb_ablation": "probe_amb_only = sigmoid(f_amb)",
            "calibrated_gap_status": "P1 comparison, not headline unless it materially outperforms raw gap before method lock",
        },
        "representation_lock": {
            "pool": "response_mean",
            "relative_layer": -4,
            "fallback_for_current_single-layer_cache": "use the cached layer/pool if only one representation is available; mark the run as legacy-single-representation",
        },
        "headline_predictions": [
            {
                "id": "H1_showcase_gap_beats_probe",
                "datasets": SHOWCASE_DATASETS,
                "prediction": "gap improves over probe_pred_only by at least 0.015 AURC on the mean over seeds",
                "success_rule": "mean(probe_pred_only_aurc - gap_aurc) >= 0.015",
                "expected_direction": "positive_delta",
            },
            {
                "id": "H2_diagnostic_collapse",
                "datasets": DIAGNOSTIC_DATASETS,
                "prediction": "gap is statistically indistinguishable from probe_pred_only or improves by less than 0.01 AURC",
                "success_rule": "abs(mean(probe_pred_only_aurc - gap_aurc)) < 0.01 or paired permutation p >= 0.01",
                "expected_direction": "near_zero_delta",
            },
            {
                "id": "H3_scod_effective_dimension",
                "datasets": SHOWCASE_DATASETS + DIAGNOSTIC_DATASETS,
                "prediction": "showcase datasets have higher effective dimensionality in (p_pred, p_amb) than diagnostic datasets",
                "success_rule": "showcase effective_dim generally > 1.6 and diagnostic effective_dim generally <= 1.15; threshold treated as diagnostic, not exclusionary",
                "expected_direction": "showcase_higher",
            },
            {
                "id": "H4_y_expert_native_load_bearing",
                "datasets": ["chaosnli", "pavlick_nli"],
                "prediction": "native y_expert gap beats shuffled-within-correctness and constant-half variants",
                "success_rule": "native gap AURC < shuffled gap AURC and native gap AURC < constant-half gap AURC on mean over seeds",
                "expected_direction": "native_lower_aurc",
            },
        ],
        "decision_rules": [
            {
                "case": "Showcase gap does not beat probe-only on ChaosNLI",
                "action": "run inverse-oracle semantic audit; if oracle helps but native does not, pivot framing or defer submission",
            },
            {
                "case": "SCOD-additive or learned-2D beats gap by >= 0.05 AURC on showcase datasets",
                "action": "promote calibrated SCOD-additive or learned-2D to headline method before final writing",
            },
            {
                "case": "Diagnostic datasets do not collapse",
                "action": "keep headline if showcase wins; revise diagnostic-regime claim to dataset-specific characterization",
            },
        ],
        "analysis_plan": {
            "significance_test": "paired permutation test over examples, p < 0.01, comparing gap to next-best method per row",
            "aggregation": "mean +/- std over seeds 0-4",
            "main_table_methods": [
                "logprob",
                "seq_entropy",
                "probe_pred_only",
                "semantic_entropy",
                "gap",
            ],
            "main_audit_variants": [
                "native",
                "shuffled_within_y_correct",
                "constant_half",
                "inverse_y_correct_oracle",
            ],
        },
        "disclosure": {
            "known_existing_result_before_registration": "llama3_8b x truthfulqa contrastive/diagnostic cache and preliminary collapse-style outputs existed before this artifact",
            "not_yet_observed_at_registration": [
                "chaosnli showcase probe/audit results",
                "pavlick_nli showcase probe/audit results",
                "ambigqa_kge2 showcase probe/audit results",
            ],
        },
    }
    payload["content_sha256"] = _sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write locked ARR pre-registration JSON.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "PRE_REGISTRATION.json")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing pre-registration.")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        try:
            existing = json.loads(args.out.read_text())
        except Exception:
            existing = {}
        if existing.get("locked") is True:
            raise SystemExit(
                f"{args.out} already exists and is locked. Pass --force only if "
                "you intentionally need to replace the artifact before tagging."
            )

    payload = build_payload()
    args.out.write_text(_canonical_json(payload))
    print(f"Wrote {args.out}")
    print(f"content_sha256={payload['content_sha256']}")
    if payload["project"]["git_commit"] is None:
        print("warning: not inside a Git repository; git commit/tag must be handled elsewhere")
    elif payload["project"]["git_status_short_at_registration"]:
        print("warning: working tree had uncommitted changes at registration")


if __name__ == "__main__":
    main()

