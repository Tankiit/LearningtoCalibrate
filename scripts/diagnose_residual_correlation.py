"""Compatibility entrypoint for residual-vs-defer diagnostics.

This wraps scripts/step6_residual_analysis.py so the shorter diagnostic command
keeps working while all implementation lives in eval.residual_analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.step6_residual_analysis import main  # noqa: E402


if __name__ == "__main__":
    main()
