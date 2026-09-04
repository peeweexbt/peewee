"""Load config.yaml + .env into one convenient object."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v


def load(path: str | Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    # env overrides that are handy in deployment
    if os.getenv("PEEWEE_PAINTER"):
        cfg["painter"]["backend"] = os.environ["PEEWEE_PAINTER"]
    cfg["_root"] = str(ROOT)
    return cfg


def env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v not in ("", None) else default
