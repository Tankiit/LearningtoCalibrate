"""rsuq.loading — Model-agnostic, memory-aware loading.

Handles GPT-2 (M2/MPS, fp32) through Llama-8B (fp16/4-bit) behind one call,
so the second-model run is a config change, not a refactor.
"""
from __future__ import annotations
import torch


def load_model_and_tokenizer(model_id: str, device: str | None = None,
                             dtype: str = "auto", load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = device or ("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    kw = {}
    if dtype == "auto":
        torch_dtype = (torch.float16 if dev == "cuda" else torch.float32)
    else:
        torch_dtype = getattr(torch, dtype)
    kw["torch_dtype"] = torch_dtype
    if load_in_4bit:                       # large second model on a CUDA box
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not load_in_4bit:
        model = model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, dev
