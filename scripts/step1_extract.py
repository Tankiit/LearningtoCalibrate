"""
Step 1: extract hidden states + generation logprobs.

Two-phase: invoke the Modal app, then download artifacts to local outputs/.
The Modal phase runs on remote GPU; the download is local. Both are idempotent.

Usage:
  python scripts/step1_extract.py --config configs/experiments/llama3_8b_truthfulqa.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config, assert_required
from utils.logging import get_logger
from utils.paths import hidden_states_path, logprobs_path
from extraction.download import download_extraction

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip-extract", action="store_true",
                   help="Only download; assume Modal extraction is already done")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    assert_required(cfg, ["model_key", "dataset"], context="step1_extract")

    h_path = hidden_states_path(cfg.model_key, cfg.dataset)
    l_path = logprobs_path(cfg.model_key, cfg.dataset)
    if h_path.exists() and l_path.exists() and not args.force:
        log.info(f"already extracted: {h_path.parent}")
        return

    if not args.skip_extract:
        log.info(f"running Modal extraction: {cfg.model_key} × {cfg.dataset}")
        # `main` is the local_entrypoint in extraction/modal_app.py;
        # `--datasets` accepts a comma-separated list (one element here).
        subprocess.run([
            "modal", "run",
            "extraction/modal_app.py",
            "--model",    cfg.model_key,
            "--datasets", cfg.dataset,
        ], check=True)

    log.info("downloading extracted artifacts...")
    download_extraction(model_key=cfg.model_key, dataset=cfg.dataset, force=args.force)
    log.info(f"step 1 complete: {h_path.parent}")


if __name__ == "__main__":
    main()
