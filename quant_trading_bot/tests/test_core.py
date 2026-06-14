"""Unit tests covering indicators, strategies, risk and backtester.

The tests are deliberately self-contained: they build a synthetic price
series so they don't depend on a network or on ``yfinance`` being able
to reach Yahoo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_trading_bot.indicators import (  # noqa: E402
    add_all_indicators,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)
from quant_trading_bot.risk import RiskManager  # noqa: E402
from quant_trading_bot.backtester import Backtester  # noqa: E402
from quant_trading_bot.strategies import (  # noqa: E402
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from quant_trading_bot.utils import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
def _synthetic_prices(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    return df


# --------------------------------------------------------------------------- #
def test_sma_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert out.iloc[-1] == pytest.approx(4.0)
    assert out.iloc[0:2].isna().all()


def test_ema_is_exponential():
    s = pd.Series(np.ones(10), dtype=float)
    out = ema(s, 3)
    # EMA of constant series is the constant
    assert out.dropna().iloc[-1] == pytest.approx(1.0)


def test_rsi_bounds():
    s = pd.Series(np.linspace(100, 50, 200))
    r = rsi(s, 14)
    assert r.dropna().between(0, 100).all()


def test_macd_shape():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 200)))
    m = macd(s)
    assert {"macd", "signal", "hist"} <= set(m.columns)
    assert len(m) == len(s)


def test_bollinger_band_ordering():
    s = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 200)))
    bb = bollinger_bands(s, 20, 2.0)
    valid = bb.dropna()
    # BB ordering only guaranteed once the rolling window is full
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_lower"] <= valid["bb_middle"]).all()


def test_add_all_indicators_adds_columns():
    df = _synthetic_prices()
    out = add_all_indicators(df)
    for col in ["SMA_fast", "RSI", "MACD", "BB_upper", "ADX", "ATR"]:
        assert col in out.columns, f"Missing indicator: {col}"


# --------------------------------------------------------------------------- #
def test_momentum_signals_shape_and_values():
    df = add_all_indicators(_synthetic_prices())
    strat = MomentumStrategy(lookback=20, threshold=0.02)
    sig = strat.generate_signals(df)
    assert set(sig.unique()) <= {-1, 0, 1}
    assert len(sig) == len(df)


def test_trend_following_signals():
    df = add_all_indicators(_synthetic_prices())
    sig = TrendFollowingStrategy().generate_signals(df)
    assert set(sig.unique()) <= {-1, 0, 1}


def test_mean_reversion_signals():
    df = add_all_indicators(_synthetic_prices())
    sig = MeanReversionStrategy().generate_signals(df)
    assert set(sig.unique()) <= {-1, 0, 1}


# --------------------------------------------------------------------------- #
def test_risk_manager_kelly_capped():
    rm = RiskManager(initial_capital=100_000, position_sizing="kelly", kelly_fraction=0.5)
    # 60% win rate, 1.5x payoff -> full Kelly ~ 0.267 -> 0.5 * 0.267 = 0.133
    size = rm.size_position(100, +1, win_rate=0.6, payoff=1.5)
    assert size > 0
    assert size < rm.initial_capital * 0.5  # capped by max_position_pct


def test_risk_manager_stop_loss():
    rm = RiskManager(initial_capital=100_000, stop_loss_pct=0.05)
    rm.state.cash = 100_000
    rm.state.position_qty = 100
    rm.state.position_avg_price = 100
    rm.state.side = __import__("quant_trading_bot.risk.manager", fromlist=["PositionSide"]).PositionSide.LONG
    rm.highest_since_entry = 100
    # 6% drop triggers stop
    assert rm.check_exit(94) == "stop_loss"


# --------------------------------------------------------------------------- #
def test_backtester_runs_and_returns_metrics():
    df = add_all_indicators(_synthetic_prices())
    rm = RiskManager(initial_capital=100_000, position_sizing="fixed_fractional", fixed_fraction=0.2)
    bt = Backtester(strategy=MomentumStrategy(lookback=10, threshold=0.01), risk_manager=rm)
    result = bt.run(df)
    assert len(result.equity) == len(df)
    assert isinstance(result.metrics.sharpe_ratio, float)
    assert result.metrics.n_trades >= 0


# --------------------------------------------------------------------------- #
def test_config_loads():
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    assert "tickers" in cfg.data
    assert "strategies" in cfg.raw
