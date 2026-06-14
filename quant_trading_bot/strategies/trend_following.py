"""Trend-following strategies.

These systems ride medium-term trends.  Signals are generated from a
fast/slow moving-average crossover, gated by the ADX to avoid taking
positions in choppy markets.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyMetadata


class TrendFollowingStrategy(Strategy):
    """Fast/slow moving-average crossover with an ADX trend filter."""

    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        adx_threshold: float = 25.0,
    ) -> None:
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.adx_threshold = adx_threshold
        self.metadata = StrategyMetadata(
            name="TrendFollowing",
            family="trend",
            description=(
                "Long when fast MA > slow MA and ADX > threshold, "
                "short when fast MA < slow MA and ADX > threshold."
            ),
            parameters={
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "adx_threshold": adx_threshold,
            },
        )

    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        suffix = f"_{ticker}" if ticker else ""
        fast_col = f"SMA_fast{suffix}" if f"SMA_fast{suffix}" in df.columns else f"EMA_fast{suffix}"
        slow_col = f"SMA_slow{suffix}" if f"SMA_slow{suffix}" in df.columns else f"EMA_slow{suffix}"
        adx_col = f"ADX{suffix}"
        for col in (fast_col, slow_col, adx_col):
            if col not in df.columns:
                raise KeyError(
                    f"Column {col!r} missing - did you run add_all_indicators?"
                )

        fast = df[fast_col]
        slow = df[slow_col]
        adx = df[adx_col]

        long_cond = (fast > slow) & (adx > self.adx_threshold)
        short_cond = (fast < slow) & (adx > self.adx_threshold)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[long_cond] = 1
        signal[short_cond] = -1
        return signal
