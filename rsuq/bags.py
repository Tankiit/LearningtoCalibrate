"""rsuq.bags — Bag construction for the D2D input side (Stage I, Experiment 1).

Primary: MC-dropout (DECISION_bag_construction.md). Ablations: token
substitution, embedding noise. One interface:

    bag = make_bag(model, input_ids, n=N, kind="mc_dropout")   # (N, T, d)

All generators leave parameters untouched (backbone frozen throughout) and
return last-layer hidden states at every position; callers select positions.
"""

from __future__ import annotations
import torch
import torch.nn as nn


@torch.no_grad()
def mc_dropout_bag(model, input_ids, n: int) -> torch.Tensor:
    """N stochastic samples in ONE forward pass: dropout is elementwise, so
    repeating the input along the batch dimension yields n independent
    dropout masks. train() enables dropout; params already frozen so no
    other effect (no batch-norm in decoder LMs). Restores mode on exit."""
    was_training = model.training
    model.eval()
    _enable_dropout(model)                                     # ONLY dropout
    try:
        ids = input_ids.expand(n, -1) if input_ids.dim() == 2 \
              else input_ids.unsqueeze(0).expand(n, -1)      # (n, T)
        return model.base_model(ids).last_hidden_state        # (n, T, d)
    finally:
        model.train(was_training)


def _enable_dropout(model):
    """Activate only nn.Dropout submodules; leave everything else in eval."""
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()


@torch.no_grad()
def token_perturb_bag(model, input_ids, n: int, p: float = 0.1,
                      vocab_size: int | None = None,
                      protect_last: int = 1) -> torch.Tensor:
    """Ablation A: each pass substitutes tokens (except the last
    `protect_last`, so the scored position's own token is intact) with
    probability p by uniform random vocabulary tokens. Deterministic model."""
    model.eval()
    V = vocab_size or model.config.vocab_size
    x = (input_ids if input_ids.dim() == 2 else
         input_ids.unsqueeze(0)).expand(n, -1).clone()       # (n, T)
    mask = torch.rand(x.shape, device=x.device) < p
    if protect_last:
        mask[:, -protect_last:] = False
    x[mask] = torch.randint(0, V, (int(mask.sum()),), device=x.device)
    return model.base_model(x).last_hidden_state             # (n, T, d)


@torch.no_grad()
def embed_noise_bag(model, input_ids, n: int, sigma: float = 0.05
                    ) -> torch.Tensor:
    """Ablation B: Gaussian noise on input embeddings, scale sigma relative
    to the per-dimension embedding std (so sigma is unitless)."""
    model.eval()
    wte = model.get_input_embeddings()
    base = wte(input_ids)                              # (1, T, d)
    scale = sigma * base.std()
    noisy = base.expand(n, -1, -1) + scale * torch.randn(
        n, *base.shape[1:], device=base.device, dtype=base.dtype)
    return model.base_model(inputs_embeds=noisy).last_hidden_state


GENERATORS = {"mc_dropout": mc_dropout_bag,
              "token_perturb": token_perturb_bag,
              "embed_noise": embed_noise_bag}


def make_bag(model, input_ids, n: int, kind: str = "mc_dropout", **kw):
    return GENERATORS[kind](model, input_ids, n, **kw)


def bag_mean(bag: torch.Tensor) -> torch.Tensor:
    """Empirical mean embedding under the identity feature map. (N,T,d)->(T,d)."""
    return bag.mean(0)
