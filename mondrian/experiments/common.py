"""Experiment validation and summaries; all evidence starts on HOLD."""
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from mondrian.audit import source_hash, write_json, require_gate


def validate_first(out):
    """Run write-free tests before every experiment entry point, then all tests."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    logs = []
    for target in ("tests/test_write_free.py", "tests"):
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", target],
                                text=True, capture_output=True)
        logs.append(result.stdout + result.stderr)
        if result.returncode:
            (out / "tests.log").write_text("\n".join(logs))
            write_json(out / "tests_gate.json", {"stage": "tests", "status": "FAIL",
                       "source_sha256": source_hash(), "returncode": result.returncode})
            raise RuntimeError("Tests failed; experiment not started")
    (out / "tests.log").write_text("\n".join(logs))
    write_json(out / "tests_gate.json", {"stage": "tests", "status": "PASS",
               "source_sha256": source_hash(), "wall_seconds": time.perf_counter()-start})


def require_tests(out):
    return require_gate(Path(out) / "tests_gate.json", "tests")


def summarize(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group, values in sorted(groups.items()):
        record = dict(zip(keys, group))
        for metric in metrics:
            a = np.asarray([v[metric] for v in values], float)
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite metric {metric}")
            record[metric] = {"mean": float(a.mean()),
                              "std": float(a.std(ddof=1)) if len(a)>1 else None,
                              "n_seeds": len(a)}
        result.append(record)
    return result
