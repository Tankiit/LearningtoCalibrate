"""Per-dataset adapters. Each module here exposes a `load() -> list[Record]`.

Adapter modules (registered in `data.registry`):
  truthfulqa.py   halueval_qa.py   triviaqa.py   popqa.py   bioasq.py

Adapters are imported on demand by the registry, so heavy dependencies
(HF datasets, parquet, JSON fallbacks) only load when their dataset is used.
"""
