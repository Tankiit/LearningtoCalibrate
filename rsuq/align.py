"""rsuq.align — Fast-tokenizer alignment helpers.

Uses the Rust-backed fast tokenizer's offset mapping to map between character
spans and token positions ROBUSTLY — replacing fragile encode(" "+tok) +
len==1 guessing that silently dropped multi-piece words and assumed GPT-2's
space convention.

Two jobs:
  locate_word_positions  : find the token index of a target word inside a
                           context string (E-A.2, E-B Figure 1). Handles
                           multi-subword words by returning the LAST subtoken
                           index (the position whose next-token distribution
                           is the model's prediction *after committing* the
                           word — the right place to score).
  map_span_to_tokens     : map an answer character span to token indices
                           (E-C TriviaQA/HaluEval answer-span attribution).

Requires a fast tokenizer (tokenizer.is_fast == True); raises otherwise so
the failure is loud, not a silent wrong-position bug.
"""

from __future__ import annotations
import torch


def _require_fast(tokenizer):
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "rsuq.align needs a fast tokenizer (offset mapping). Load with "
            "AutoTokenizer.from_pretrained(..., use_fast=True).")


def locate_word_positions(tokenizer, context: str, word: str,
                          which: str = "last") -> dict:
    """Return token positions of `word` as it appears in `context`.

    context MUST contain word as a substring (case-sensitive). Returns the
    char span found, the token-index span, and the single scoring index
    (`which`: 'last' subtoken by default, or 'first').

    Example:
        context = "I sat by the edge of the river bank quietly"
        word    = "bank"   -> finds char span, maps to token idx via offsets
    """
    _require_fast(tokenizer)
    start = context.find(word)
    if start < 0:
        raise ValueError(f"{word!r} not a substring of context")
    end = start + len(word)
    enc = tokenizer(context, return_offsets_mapping=True,
                    return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()        # [(c0,c1), ...]
    toks = [i for i, (c0, c1) in enumerate(offsets)
            if c0 < end and c1 > start and c1 > c0]    # overlap, skip specials
    if not toks:
        raise ValueError(f"no token overlaps span [{start},{end}) for {word!r}")
    idx = toks[-1] if which == "last" else toks[0]
    return {"input_ids": enc["input_ids"], "char_span": (start, end),
            "token_span": (toks[0], toks[-1]), "score_index": idx,
            "n_subtokens": len(toks)}


def map_span_to_tokens(tokenizer, text: str, span_start: int, span_end: int,
                       encoding=None) -> list[int]:
    """Map a character span [span_start, span_end) to the list of token
    indices covering it. For E-C: attribute a per-token width signal to the
    answer span. `encoding` may be a precomputed BatchEncoding (with
    offset_mapping) to avoid re-tokenizing."""
    _require_fast(tokenizer)
    if encoding is None:
        encoding = tokenizer(text, return_offsets_mapping=True,
                             return_tensors="pt")
    offsets = encoding["offset_mapping"][0].tolist()
    return [i for i, (c0, c1) in enumerate(offsets)
            if c0 < span_end and c1 > span_start and c1 > c0]


def find_answer_span(context: str, answer: str) -> tuple[int, int] | None:
    """Character span of the first occurrence of `answer` in `context`.
    Returns None if absent (caller decides: skip, or use generation offsets).
    For QA where the answer is appended, the span is trivially the tail; for
    extractive settings this locates it in the passage."""
    i = context.find(answer)
    return (i, i + len(answer)) if i >= 0 else None
