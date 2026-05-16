"""Tiny logging helper.

A single `get_logger(name)` wrapper around stdlib logging so every module
gets the same format without each one re-doing basicConfig. Format is
intentionally compact for batch sweep output.
"""
from __future__ import annotations
import logging
import os
import sys

_CONFIGURED = False


def _configure_root_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("OVA_ARR_LOGLEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_once()
    return logging.getLogger(name)
