"""Metric shim for collaborator appendix tasks.

The canonical implementation lives in eval/risk_coverage.py. This module keeps
the collaborator-facing import stable while avoiding a second implementation.
"""
from eval.risk_coverage import (
    aurc,
    aurc_table,
    coverage_at_acc,
    coverage_at_accuracy,
    risk_coverage_curve,
)

__all__ = [
    "aurc",
    "aurc_table",
    "coverage_at_acc",
    "coverage_at_accuracy",
    "risk_coverage_curve",
]
