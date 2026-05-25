"""
Synthetic-data test for OVA + gap signal end-to-end.

This is the critical test: can we recover meaningful signal on toy data
where we know the answer?
"""
import numpy as np
import torch
import pytest

from probes.ova import OVAHeads, OVATrainConfig, ova_loss
from probes.training import prepare_training_data, train_ova
from gap.signal import gap_signal, compute_signals


def test_gap_signal_is_logit_difference():
    """Δ = logit_pred - logit_defer, by definition."""
    lp = torch.tensor([1.0, 2.0, -0.5])
    ld = torch.tensor([0.5, 1.5, 0.0])
    g = gap_signal(lp, ld)
    expected = lp - ld
    assert torch.allclose(g, expected)


def test_ova_heads_independence():
    """Two heads must have independent parameters."""
    model = OVAHeads(input_dim=64)
    pred_params = list(model.pred.parameters())
    defer_params = list(model.defer.parameters())
    pred_ids = {id(p) for p in pred_params}
    defer_ids = {id(p) for p in defer_params}
    # No parameter object shared between heads
    assert pred_ids.isdisjoint(defer_ids)


def test_ova_recovers_signal_on_synthetic_data():
    """
    Construct synthetic h^+ and h^- where:
      - h^+ has mean (1, 0) — the 'correct' direction
      - h^- has mean (-1, 0) — the 'wrong' direction
    A linear probe should easily separate them, and the gap should
    be reliably positive on h^+ and negative on h^-.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    N, d = 300, 32
    rng = np.random.default_rng(0)
    h_pos = rng.standard_normal((N, d))
    h_pos[:, 0] += 2.0  # shift first dim
    h_neg = rng.standard_normal((N, d))
    h_neg[:, 0] -= 2.0

    expert_reliable = np.ones(N, dtype=bool)
    example_ids = np.array([f"ex_{i}" for i in range(N)])

    data = prepare_training_data(
        h_pos=h_pos, h_neg=h_neg,
        expert_reliable=expert_reliable,
        example_ids=example_ids,
        halueval_style=True,
    )

    cfg = OVATrainConfig(epochs=20, batch_size=64)
    model = train_ova(data, cfg=cfg, seed=0, device="cpu")

    is_pos = np.concatenate([np.ones(N, dtype=bool), np.zeros(N, dtype=bool)])
    sig = compute_signals(
        model=model, h=data.h, is_pos=is_pos,
        example_ids=data.example_ids, device="cpu",
    )

    gap_pos = sig.gap[sig.is_pos]
    gap_neg = sig.gap[~sig.is_pos]

    # On this synthetic data, positive class should have higher mean gap
    assert gap_pos.mean() > gap_neg.mean(), (
        f"Gap on h^+ ({gap_pos.mean():.3f}) should exceed "
        f"gap on h^- ({gap_neg.mean():.3f})"
    )

    # AUROC for distinguishing pos from neg via gap alone should be high
    from sklearn.metrics import roc_auc_score
    labels = sig.is_pos.astype(int)
    auroc = roc_auc_score(labels, sig.gap)
    assert auroc > 0.85, f"AUROC of gap on synthetic data only {auroc:.3f}; expected >0.85"


def test_loss_decreases_during_training():
    """Trivial sanity: loss at end of training is below loss at start."""
    torch.manual_seed(0)
    N, d = 100, 16
    h_pos = torch.randn(N, d) + 1.0
    h_neg = torch.randn(N, d) - 1.0
    h = torch.cat([h_pos, h_neg])
    y_model  = torch.cat([torch.ones(N), torch.zeros(N)]).long()
    y_expert = torch.ones(2 * N).long()

    model = OVAHeads(input_dim=d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    lp, ld = model(h)
    initial = ova_loss(lp, ld, y_model, y_expert).item()

    for _ in range(50):
        opt.zero_grad()
        lp, ld = model(h)
        ova_loss(lp, ld, y_model, y_expert).backward()
        opt.step()

    lp, ld = model(h)
    final = ova_loss(lp, ld, y_model, y_expert).item()
    assert final < initial * 0.5, f"loss did not decrease: {initial:.4f} → {final:.4f}"
