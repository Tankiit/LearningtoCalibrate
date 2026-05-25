"""Mirror Modal-volume extraction artifacts into local outputs/.

After `modal_app.run_extraction` finishes, the volume holds:
    /{model_key}/{dataset}/hidden_states.pt
    /{model_key}/{dataset}/logprobs.pt
(see modal_app.py docstring). This downloader pulls them to the canonical
local paths so steps 2-4 can read them without Modal involvement.

Idempotent: already-present local files are skipped unless `force=True`.
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logging import get_logger
from utils.paths import (
    hidden_states_path, logprobs_path, ensure_parent,
)

log = get_logger(__name__)

VOLUME_NAME = "ova-arr-extract"


def _download_one(volume, remote: str, local: Path, force: bool) -> bool:
    """Return True if a download happened, False if skipped."""
    if local.exists() and not force:
        log.info(f"already present, skipping: {local}")
        return False
    ensure_parent(local)
    log.info(f"downloading {VOLUME_NAME}:{remote} → {local}")
    with open(local, "wb") as f:
        for chunk in volume.read_file(remote):
            f.write(chunk)
    return True


def download_extraction(
    model_key: str,
    dataset: str,
    force: bool = False,
) -> tuple[Path, Path]:
    """Pull hidden_states.pt and logprobs.pt for one (model, dataset) cell.

    Returns (hidden_states_local_path, logprobs_local_path).
    Raises a clear error if Modal isn't configured.
    """
    try:
        import modal
    except ImportError as e:
        raise RuntimeError(
            "Modal not installed. Install via `pip install modal` and run "
            "`modal token new` before calling download_extraction."
        ) from e

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    h_local = hidden_states_path(model_key, dataset)
    l_local = logprobs_path(model_key, dataset)

    _download_one(volume, f"/{model_key}/{dataset}/hidden_states.pt", h_local, force)
    _download_one(volume, f"/{model_key}/{dataset}/logprobs.pt",       l_local, force)

    return h_local, l_local
