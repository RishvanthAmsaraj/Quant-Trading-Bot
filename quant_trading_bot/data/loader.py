"""Market data ingestion built on top of :mod:`yfinance`.

The loader transparently caches downloaded data as CSV files so repeated
backtests do not hammer the Yahoo Finance endpoint.  All downstream
modules expect a tidy OHLCV ``pandas.DataFrame`` indexed by date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
import yfinance as yf

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class MarketDataLoader:
    """Download, cache and validate OHLCV data from Yahoo Finance."""

    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

    def __init__(self, cache_dir: Union[str, Path] = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("MarketDataLoader initialised (cache=%s)", self.cache_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def download(
        self,
        tickers: Union[str, Iterable[str]],
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return a tidy OHLCV DataFrame for ``tickers``.

        Parameters
        ----------
        tickers:
            A single ticker or an iterable of tickers.
        period:
            Lookback window (e.g. ``"1y"``, ``"6mo"``, ``"max"``).
        interval:
            Bar size (``"1d"``, ``"1h"``, ``"15m"`` …).
        use_cache:
            If ``True`` (default) load from disk when available.
        """
        ticker_list = self._normalise_tickers(tickers)
        frames: List[pd.DataFrame] = []
        for ticker in ticker_list:
            frames.append(self._download_one(ticker, period, interval, use_cache))
        combined = pd.concat(frames, axis=1)
        # Drop duplicate columns that can occur when re-downloading
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined.sort_index()

    def load_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load a cached CSV file and validate it."""
        df = pd.read_csv(Path(path), index_col=0, parse_dates=True)
        self._validate(df)
        return df

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_tickers(tickers: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(tickers, str):
            return [t.strip().upper() for t in tickers.split(",") if t.strip()]
        return [str(t).strip().upper() for t in tickers if str(t).strip()]

    def _cache_path(self, ticker: str, period: str, interval: str) -> Path:
        safe_period = period.replace("/", "-")
        return self.cache_dir / f"{ticker}_{interval}_{safe_period}.csv"

    def _download_one(
        self,
        ticker: str,
        period: str,
        interval: str,
        use_cache: bool,
    ) -> pd.DataFrame:
        path = self._cache_path(ticker, period, interval)
        if use_cache and path.exists():
            LOGGER.debug("Cache hit for %s (%s)", ticker, path.name)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            LOGGER.info("Downloading %s from Yahoo Finance …", ticker)
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                raise RuntimeError(f"No data returned for ticker '{ticker}'")
            # Flatten multi-index columns that yfinance sometimes returns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            self._validate(data)
            data.to_csv(path)
            df = data

        # Add a top-level ticker column suffix when multiple tickers
        # are requested.  Single ticker is returned as-is.
        df = df.copy()
        df.columns = [f"{c}_{ticker}" if not c.endswith(f"_{ticker}") else c for c in df.columns]
        return df

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV data is missing required columns: {missing}")
        if df.empty:
            raise ValueError("Downloaded OHLCV frame is empty")
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def get_close(self, df: pd.DataFrame, ticker: Optional[str] = None) -> pd.Series:
        """Return the close-price series for ``ticker``.

        When the DataFrame contains multiple tickers (suffixed columns)
        pass the ticker explicitly.  When there is a single ticker, the
        first ``Close`` column is returned.
        """
        close_cols = [c for c in df.columns if c.startswith("Close")]
        if ticker is None:
            if not close_cols:
                raise KeyError("No 'Close' column found in DataFrame")
            return df[close_cols[0]]
        match = [c for c in close_cols if c.endswith(f"_{ticker}")]
        if not match:
            raise KeyError(f"No close column for ticker {ticker}")
        return df[match[0]]
