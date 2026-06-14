"""Momentum strategies.

A momentum strategy goes long when an instrument's recent return is
strongly positive and short when it is strongly negative.  The lookback
window and threshold are configurable.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyMetadata


class MomentumStrategy(Strategy):
    """Long/short based on rolling-window returns."""

    def __init__(self, lookback: int = 20, threshold: float = 0.02) -> None:
        self.lookback = lookback
        self.threshold = threshold
        self.metadata = StrategyMetadata(
            name="Momentum",
            family="momentum",
            description=(
                "Long when N-day return > threshold, short when < -threshold, "
                "otherwise flat."
            ),
            parameters={"lookback": lookback, "threshold": threshold},
        )

    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        suffix = f"_{ticker}" if ticker else ""
        close_col = f"Close{suffix}"
        if close_col not in df.columns:
            raise KeyError(f"Column {close_col!r} missing from DataFrame")
        close = df[close_col]
        momentum = close.pct_change(periods=self.lookback)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[momentum > self.threshold] = 1
        signal[momentum < -self.threshold] = -1
        return signal
