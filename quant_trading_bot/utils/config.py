"""Configuration loading utilities.

The bot is configured through a YAML file (``configs/config.yaml`` by
default).  This module centralises loading and provides a small helper
that substitutes environment variables (``${VAR_NAME}``) so secrets
(e.g. API keys) can be injected at runtime.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _substitute_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment values."""

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


@dataclass(frozen=True)
class Config:
    """Immutable configuration container."""

    raw: Dict[str, Any]

    # --- typed accessors -------------------------------------------------
    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def indicators(self) -> Dict[str, Any]:
        return self.raw["indicators"]

    @property
    def strategies(self) -> Dict[str, Any]:
        return self.raw["strategies"]

    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw["risk"]

    @property
    def backtest(self) -> Dict[str, Any]:
        return self.raw["backtest"]

    @property
    def ml(self) -> Dict[str, Any]:
        return self.raw["ml"]

    @property
    def visualization(self) -> Dict[str, Any]:
        return self.raw["visualization"]

    def section(self, name: str) -> Dict[str, Any]:
        """Return an arbitrary top-level section by name."""
        if name not in self.raw:
            raise KeyError(f"Config section '{name}' not found")
        return self.raw[name]


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    """Load a YAML config file and return a :class:`Config` instance.

    Parameters
    ----------
    path:
        Path to the YAML file.  Relative paths are interpreted from the
        current working directory.

    Returns
    -------
    Config
        Frozen dataclass holding the parsed configuration.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")

    return Config(raw=_substitute_env(raw))
