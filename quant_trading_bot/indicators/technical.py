"""Vectorised implementations of common technical indicators.

All functions take a ``pandas.Series`` of prices (or a DataFrame of
OHLCV) and return a new ``pandas.Series`` / ``DataFrame`` aligned with
the input index.  Implementations follow standard textbook formulas so
they exactly match the values produced by TradingView / TA-Lib.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (adjust=False for reproducibility)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# Momentum / oscillators
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing: equivalent to EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    return rsi_series.fillna(50.0)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving-Average Convergence/Divergence.

    Returns a DataFrame with columns ``macd``, ``signal`` and ``hist``.
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": macd_line - signal_line,
        }
    )


# --------------------------------------------------------------------------- #
# Volatility / bands
# --------------------------------------------------------------------------- #
def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: middle, upper, lower + bandwidth & %B."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    percent_b = (series - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }
    )


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# Trend strength
# --------------------------------------------------------------------------- #
def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Average Directional Index.

    Returns DataFrame with columns ``+DI``, ``-DI`` and ``ADX``.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = atr(high, low, close, period=period)  # uses Wilder smoothing
    atr_value = tr.ewm(alpha=1.0 / period, adjust=False).mean()  # smoothed TR

    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1.0 / period, adjust=False
    ).mean() / atr_value
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1.0 / period, adjust=False
    ).mean() / atr_value

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx_val})


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
def add_all_indicators(
    df: pd.DataFrame,
    ticker: str = "",
    sma_fast: int = 20,
    sma_slow: int = 50,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_period: int = 14,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Attach a full panel of indicators to an OHLCV frame.

    Parameters
    ----------
    df:
        DataFrame containing ``Open``, ``High``, ``Low``, ``Close`` and
        ``Volume`` columns (suffixed with the ticker if multi-asset).
    ticker:
        Optional ticker name used to disambiguate the columns.
    """
    suffix = f"_{ticker}" if ticker else ""

    def col(name: str) -> pd.Series:
        full = f"{name}{suffix}"
        if full in df.columns:
            return df[full]
        if name in df.columns:
            return df[name]
        raise KeyError(f"Column {name!r} (or {full!r}) missing from DataFrame")

    close = col("Close")
    high = col("High")
    low = col("Low")

    out = df.copy()
    out[f"SMA_fast{suffix}"] = sma(close, sma_fast)
    out[f"SMA_slow{suffix}"] = sma(close, sma_slow)
    out[f"EMA_fast{suffix}"] = ema(close, ema_fast)
    out[f"EMA_slow{suffix}"] = ema(close, ema_slow)
    out[f"RSI{suffix}"] = rsi(close, rsi_period)

    macd_df = macd(close, ema_fast, ema_slow, macd_signal)
    out[f"MACD{suffix}"] = macd_df["macd"]
    out[f"MACD_signal{suffix}"] = macd_df["signal"]
    out[f"MACD_hist{suffix}"] = macd_df["hist"]

    bb = bollinger_bands(close, bb_period, bb_std)
    out[f"BB_upper{suffix}"] = bb["bb_upper"]
    out[f"BB_middle{suffix}"] = bb["bb_middle"]
    out[f"BB_lower{suffix}"] = bb["bb_lower"]
    out[f"BB_bandwidth{suffix}"] = bb["bb_bandwidth"]
    out[f"BB_percent_b{suffix}"] = bb["bb_percent_b"]

    out[f"ATR{suffix}"] = atr(high, low, close, atr_period)
    adx_df = adx(high, low, close, adx_period)
    out[f"ADX{suffix}"] = adx_df["ADX"]
    out[f"+DI{suffix}"] = adx_df["+DI"]
    out[f"-DI{suffix}"] = adx_df["-DI"]
    return out
