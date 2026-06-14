"""Ensemble strategy that combines multiple sub-strategies via voting."""

from __future__ import annotations

from typing import List

import pandas as pd

from .base import Strategy, StrategyMetadata


class EnsembleStrategy(Strategy):
    """Average the signals of multiple strategies.

    Each sub-strategy votes +1 / -1 / 0.  The ensemble returns the
    rounded average, so majority rules: a 2-1 long vote becomes +1.
    """

    def __init__(self, strategies: List[Strategy], min_agreement: int = 1) -> None:
        if not strategies:
            raise ValueError("Ensemble requires at least one sub-strategy")
        self.strategies = strategies
        self.min_agreement = min_agreement
        self.metadata = StrategyMetadata(
            name="Ensemble",
            family="ensemble",
            description=(
                f"Majority vote across {len(strategies)} strategies: "
                + ", ".join(s.metadata.name for s in strategies)
            ),
            parameters={
                "min_agreement": min_agreement,
                "components": [s.metadata.name for s in strategies],
            },
        )

    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        votes = pd.DataFrame(
            {s.metadata.name: s.generate_signals(df, ticker) for s in self.strategies}
        )
        summed = votes.sum(axis=1)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[summed >= self.min_agreement] = 1
        signal[summed <= -self.min_agreement] = -1
        return signal
