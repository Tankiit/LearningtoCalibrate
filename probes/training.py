"""
Training loop for OVA heads.

Plain PyTorch — no Lightning, no accelerate, no Trainer. Probe training is
short enough (5-10 minutes per config on a laptop) that abstraction overhead
isn't worth it.

Inputs from extraction step: frozen hidden states h^+ and h^-.
Inputs from data step: per-record y_model and y_expert labels.

The training set is the UNION of (h^+, y_model=1) and (h^-, y_model=0),
each with its own y_expert label.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import NamedTuple

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from probes.ova import OVAHeads, OVATrainConfig, ova_loss, compute_pos_weights
from utils.logging import get_logger
from utils.device import configure_torch_threads, get_torch_device

log = get_logger(__name__)


class TrainingData(NamedTuple):
    """
    Stacked (h^+, h^-) representations with corresponding labels.

    For N questions:
      h:        [2N, d]  — h^+ then h^-
      y_model:  [2N]     — 1 for h^+ rows, 0 for h^- rows
      y_expert: [2N]     — expert_reliable for h^+ rows; for h^- rows
                          this depends on the dataset (see prepare_training_data)
    """
    h: torch.Tensor          # [2N, d]
    y_model: torch.Tensor    # [2N], int64
    y_expert: torch.Tensor   # [2N], int64
    example_ids: np.ndarray  # [2N], for joins


def prepare_training_data(
    h_pos: np.ndarray,        # [N, d]
    h_neg: np.ndarray,        # [N, d]
    expert_reliable: np.ndarray,  # [N], bool — per record
    example_ids: np.ndarray,      # [N], str
    halueval_style: bool = False,
) -> TrainingData:
    """
    Stack pos+neg into a single training set.

    halueval_style: if True, on h^- rows we set y_expert = 1 - y_model
                    (i.e., y_expert=1 for hallucinated reps). This matches
                    HaluEval's design where the expert always knows the
                    right answer, so hallucinated reps are exactly the
                    "model wrong, expert right" deferral target.

                    For TruthfulQA / PopQA, leave halueval_style=False so
                    y_expert just mirrors expert_reliable for both reps
                    and cross-question variation drives the gap.
    """
    N, d = h_pos.shape
    h = np.concatenate([h_pos, h_neg], axis=0)
    ids = np.concatenate([example_ids, example_ids])

    # y_model: 1 for h^+, 0 for h^-
    y_model = np.concatenate([np.ones(N), np.zeros(N)]).astype(np.int64)

    # y_expert: by default mirrors expert_reliable on both reps.
    er = expert_reliable.astype(np.int64)
    y_expert = np.concatenate([er, er])
    if halueval_style:
        # On wrong-rep rows, y_expert = 1 - y_model = 1 (expert always right
        # when model hallucinates). Override the second half:
        y_expert[N:] = 1

    return TrainingData(
        h           = torch.from_numpy(h).float(),
        y_model     = torch.from_numpy(y_model),
        y_expert    = torch.from_numpy(y_expert),
        example_ids = ids,
    )


def train_ova(
    data: TrainingData,
    cfg: OVATrainConfig,
    device: str = "auto",
    seed: int = 0,
) -> OVAHeads:
    """
    Train OVAHeads. Returns the trained module on the specified device.
    """
    torch.manual_seed(seed)
    configure_torch_threads()
    device = get_torch_device(device)
    if device == "cpu":
        log.warning("No accelerator available; training on CPU")

    d = data.h.shape[1]
    model = OVAHeads(input_dim=d).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    pos_w_pred, pos_w_defer = (None, None)
    if cfg.balance_classes:
        pos_w_pred, pos_w_defer = compute_pos_weights(data.y_model, data.y_expert)
        pos_w_pred  = pos_w_pred.to(device)
        pos_w_defer = pos_w_defer.to(device)
        log.info(f"pos_weights: pred={pos_w_pred.item():.3f} defer={pos_w_defer.item():.3f}")

    h_dev  = data.h.to(device)
    ym_dev = data.y_model.to(device)
    ye_dev = data.y_expert.to(device)

    loader = DataLoader(
        TensorDataset(h_dev, ym_dev, ye_dev),
        batch_size=cfg.batch_size, shuffle=True,
    )

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for h_batch, ym_batch, ye_batch in loader:
            opt.zero_grad()
            lp, ld = model(h_batch)
            loss = ova_loss(lp, ld, ym_batch, ye_batch,
                              pos_w_pred, pos_w_defer)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * h_batch.size(0)
        epoch_loss /= len(data.h)
        if epoch == cfg.epochs - 1 or epoch % 10 == 0:
            log.info(f"epoch {epoch}: loss={epoch_loss:.4f}")

    model.eval()
    return model
