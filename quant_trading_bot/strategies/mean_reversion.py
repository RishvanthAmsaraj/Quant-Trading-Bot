"""Mean-reversion strategies.

These strategies assume that price deviations from a fair value mean
will revert.  Signals are generated using RSI and Bollinger Band
extremes.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyMetadata


class MeanReversionStrategy(Strategy):
    """Buy oversold, sell overbought.

    Combines RSI thresholds with Bollinger Band touches.  Either
    indicator firing produces a long/short signal, depending on
    configuration.
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_threshold: float = 0.05,
    ) -> None:
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_threshold = bb_threshold
        self.metadata = StrategyMetadata(
            name="MeanReversion",
            family="mean_reversion",
            description=(
                "Long when RSI is oversold or price tags lower Bollinger Band, "
                "short when RSI is overbought or price tags upper band."
            ),
            parameters={
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
                "bb_threshold": bb_threshold,
            },
        )

    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        suffix = f"_{ticker}" if ticker else ""
        rsi_col = f"RSI{suffix}"
        bb_pct_b = f"BB_percent_b{suffix}"
        for col in (rsi_col, bb_pct_b):
            if col not in df.columns:
                raise KeyError(f"Column {col!r} missing - did you run add_all_indicators?")

        rsi = df[rsi_col]
        pct_b = df[bb_pct_b]

        signal = pd.Series(0, index=df.index, dtype=int)
        # Long: oversold RSI or price near/below lower BB
        long_cond = (rsi < self.rsi_oversold) | (pct_b < self.bb_threshold)
        # Short: overbought RSI or price near/above upper BB
        short_cond = (rsi > self.rsi_overbought) | (pct_b > (1.0 - self.bb_threshold))
        signal[long_cond] = 1
        signal[short_cond] = -1
        return signal
