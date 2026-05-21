"""
Unified record schema for ALL datasets in this paper.

EVERY adapter in `data/adapters/` MUST yield records matching this schema:

    {
      'example_id':       str,    # stable hash; joins extraction+probes+results
      'question':         str,    # the prompt text
      'correct_answer':   str,    # gold answer text  (a^+ in the paper)
      'wrong_answer':     str,    # contrastive distractor (a^- in the paper)
      'category':         str,    # dataset-specific subcategory; '' if none
      'expert_reliable':  bool,   # is e(x) = 1 for this question?
      'meta':             dict,   # dataset-specific extras; never accessed by core code
    }

Why these specific fields:
  - `correct_answer` and `wrong_answer` are required because Δ(x) is computed
    from BOTH representations h^+ and h^-. No exceptions — adapters that lack
    a natural distractor must construct one (shift-by-1 over the dataset).
  - `expert_reliable` is the e(x) signal. Defaults to True for datasets with
    no per-question variation; varies per-record for TruthfulQA (category-driven)
    and PopQA (popularity-driven).
  - `category` is for slicing analyses (e.g., "does Δ work on Conspiracies?");
    not used in training.
  - `meta` is the escape hatch. PopQA's `page_views` lives here; never promote
    to top-level unless multiple datasets need it.

If you find yourself wanting to add a top-level field, ask: do at least 3 of
the 5 datasets need it? If no, it goes in `meta`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    example_id: str
    question: str
    correct_answer: str
    wrong_answer: str
    category: str = ""
    expert_reliable: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """For pandas/parquet round-tripping."""
        return {
            "example_id":      self.example_id,
            "question":        self.question,
            "correct_answer":  self.correct_answer,
            "wrong_answer":    self.wrong_answer,
            "category":        self.category,
            "expert_reliable": self.expert_reliable,
            "meta":            self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        return cls(**d)


def stable_id(*parts: str) -> str:
    """Deterministic ID from content. SHA-1[:16] survives dataset re-downloads."""
    import hashlib
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
