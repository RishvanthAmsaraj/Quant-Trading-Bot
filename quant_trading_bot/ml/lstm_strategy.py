"""Strategy adapter that turns LSTM predictions into trading signals."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..strategies.base import Strategy, StrategyMetadata
from .lstm_model import LSTMPricePredictor


class LSTMStrategy(Strategy):
    """Long when LSTM predicts next bar will be higher, else flat.

    The strategy is *long-only* by design: predicting the absolute
    direction of a short is harder and the goal is to demonstrate how
    a learned signal plugs into the same backtester as rule-based
    strategies.
    """

    def __init__(self, predictor: LSTMPricePredictor, threshold: float = 0.0) -> None:
        self.predictor = predictor
        self.threshold = threshold
        self.metadata = StrategyMetadata(
            name="LSTM",
            family="ml",
            description=(
                "Long when LSTM-predicted next close > last close + threshold, "
                "otherwise flat."
            ),
            parameters={"threshold": threshold, "lookback": predictor.lookback},
        )
        self._predictions: Optional[pd.Series] = None

    def train(self, close: pd.Series, **kwargs) -> None:
        """Train the underlying LSTM on ``close``."""
        self.predictor.fit(close, **kwargs)
        self._predictions = self.predictor.predict(close)

    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        suffix = f"_{ticker}" if ticker else ""
        close_col = f"Close{suffix}" if f"Close{suffix}" in df.columns else "Close"
        if close_col not in df.columns:
            raise KeyError(f"Close column '{close_col}' missing from DataFrame")

        if self._predictions is None:
            self._predictions = self.predictor.predict(df[close_col])

        close = df[close_col]
        pred = self._predictions.reindex(close.index).ffill()
        delta = (pred - close) / close
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[delta > self.threshold] = 1
        return signal
