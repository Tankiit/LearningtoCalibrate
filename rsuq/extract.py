"""rsuq.extract — State collection using the transformers library properly.

Three idioms this module enforces:

1. CHUNK, DON'T PAD. Canonical CLM preprocessing: tokenize everything,
   concatenate, split into fixed-size blocks. Uniform lengths => full
   batching with no attention masks and no truncation waste.

2. CACHE STATES, NOT LOGITS. lm_head is (typically) tied to the input
   embeddings, so logits are one matmul from h. Caching h for 50k positions
   is ~150MB fp32; caching (50k, V) logits is ~5GB fp16. Recompute lazily:
       z = logits_from_states(model, h_batch)

3. CALL THE BASE MODEL. model.base_model(...) returns last_hidden_state
   directly — no 13-layer tuple unless last-4 pooling is requested
   (the AAAI context-vector decision), in which case hidden states are
   materialised batch-locally and reduced immediately.

Model-agnostic: AutoModelForCausalLM + get_input_embeddings(); no
GPT-2-specific attribute access (the Llama follow-up needs only a model id).
"""

from __future__ import annotations
import torch
from dataclasses import dataclass


# ----------------------------------------------------------- preprocessing
def chunk_texts(tokenizer, texts, block_size: int = 256,
                max_blocks: int | None = None) -> torch.Tensor:
    """run_clm-style: tokenize, concatenate, split into (M, block_size)."""
    enc = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    flat = [t for seq in enc for t in seq]
    M = len(flat) // block_size
    if max_blocks:
        M = min(M, max_blocks)
    return torch.tensor(flat[:M * block_size]).view(M, block_size)


# ------------------------------------------------------------- collection
@dataclass
class StateCache:
    h: torch.Tensor        # (N, d) fp32 hidden states (last layer or pooled)
    gold: torch.Tensor     # (N,) next-token ids

    def __len__(self):
        return self.h.shape[0]


@torch.inference_mode()
def collect_states(model, blocks: torch.Tensor, batch_size: int = 32,
                   device: str = "cuda", pool: str = "last",
                   dtype=None) -> StateCache:
    """Batched teacher-forced pass over uniform blocks. pool: 'last' (last
    layer) or 'last4_mean' (AAAI context-vector decision)."""
    model.eval()
    if dtype is not None:
        model.to(dtype)
    hs, ys = [], []
    base = model.base_model            # GPT2Model / LlamaModel / ...
    need_all = pool == "last4_mean"
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        if need_all:
            out = base(ids, output_hidden_states=True)
            h = torch.stack(out.hidden_states[-4:], 0).mean(0)
        else:
            h = base(ids).last_hidden_state             # (B, T, d)
        hs.append(h[:, :-1].reshape(-1, h.shape[-1]).float().cpu())
        ys.append(ids[:, 1:].reshape(-1).cpu())
    return StateCache(torch.cat(hs), torch.cat(ys))


# ------------------------------------------------------------ lazy logits
@torch.inference_mode()
def logits_from_states(model, h: torch.Tensor) -> torch.Tensor:
    """z = lm_head(h). One matmul; works for tied and untied heads.
    h: (..., d) on any device -> (..., V) on the head's device."""
    head = model.get_output_embeddings()
    return head(h.to(next(head.parameters()).device,
                     next(head.parameters()).dtype)).float()


def embedding_matrix(model) -> torch.Tensor:
    """Static input embeddings for frame construction — model-agnostic."""
    return model.get_input_embeddings().weight.detach()
