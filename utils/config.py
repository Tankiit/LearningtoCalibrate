"""YAML config loader with `extends:` composition.

A config file may contain:
  extends:
    - models/llama3_8b
    - datasets/truthfulqa
  seed: 0
  halueval_style_labels: false

`extends` paths are resolved relative to the configs/ directory (the parent
of the file being loaded). Items earlier in the list are overridden by
later ones, and the file's own keys override its `extends`.

The returned `Config` is a dict subclass, so:
  - cfg.model_key       — attribute access (raises KeyError if missing)
  - cfg["model_key"]    — item access
  - cfg.get("ova_train") — dict-style get
  - dict(cfg)           — round-trips through the standard dict ctor

`assert_required` is a small helper for fail-fast checks at the top of
each step script.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """Dict with attribute access. Use `Config(d)` from any plain dict."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _resolve_extends_path(token: str, configs_root: Path) -> Path:
    """`token` may be 'models/llama3_8b' or 'models/llama3_8b.yaml'."""
    p = configs_root / token
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    return p


def _load_raw(path: Path, configs_root: Path, _seen: set[Path]) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Cyclic extends through {path}")
    _seen.add(path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping at top level")

    extends = raw.pop("extends", []) or []
    if isinstance(extends, str):
        extends = [extends]

    merged: dict = {}
    for token in extends:
        ext_path = _resolve_extends_path(token, configs_root)
        merged.update(_load_raw(ext_path, configs_root, _seen))
    merged.update(raw)
    return merged


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving `extends:` relative to its configs/ root."""
    p = Path(path).resolve()
    # Walk up until we find a directory called 'configs' so extends resolve
    # relative to it. Fall back to the file's directory if not found.
    configs_root = p.parent
    for ancestor in p.parents:
        if ancestor.name == "configs":
            configs_root = ancestor
            break
    return Config(_load_raw(p, configs_root, set()))


def assert_required(
    cfg: dict, keys: Iterable[str], context: str = "config",
) -> None:
    """Fail fast if any required keys are missing or None."""
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise KeyError(
            f"{context}: missing required keys {missing}. Got keys: {sorted(cfg.keys())}"
        )
