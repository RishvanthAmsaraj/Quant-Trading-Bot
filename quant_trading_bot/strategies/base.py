"""Base classes for trading strategies.

A :class:`Strategy` consumes a feature-enriched OHLCV DataFrame and
returns a position ``pandas.Series`` whose values are drawn from
``{-1, 0, +1}`` (short, flat, long).  The downstream backtester is
agnostic to *how* the signal was produced, which makes strategies easy
to mix-and-match.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd


@dataclass
class StrategyMetadata:
    """Human-readable information about a strategy."""

    name: str
    family: str  # "momentum" | "mean_reversion" | "trend" | "ml"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all signal-generating strategies."""

    metadata: StrategyMetadata

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        """Return a ``pandas.Series`` of positions in ``{-1, 0, +1}``.

        Implementations are expected to be vectorised and to align the
        output index with ``df.index``.
        """

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v}" for k, v in self.metadata.parameters.items())
        return f"{self.metadata.name}({params})"
