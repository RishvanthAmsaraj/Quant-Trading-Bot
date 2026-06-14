"""Performance metrics for backtests.

All functions operate on the *return* series produced by a backtest
(per-bar simple returns).  Metrics are designed to be unambiguous and
match those reported by industry-standard libraries (e.g. quantstats).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for the headline performance metrics of a backtest."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int       # bars
    volatility: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    n_trades: int
    exposure: float                  # fraction of bars in market

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_metrics(
    returns: pd.Series,
    trades: pd.DataFrame,
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Compute a full metrics panel from backtest outputs.

    Parameters
    ----------
    returns:
        Per-bar simple returns of the strategy (not the asset).
    trades:
        Trade log DataFrame (one row per closed trade) with at least
        a ``pnl`` column.
    equity:
        Equity curve (portfolio value over time).
    risk_free_rate:
        Annualised risk-free rate (e.g. 0.02).
    periods_per_year:
        Number of bars per year (252 for daily).
    """
    returns = pd.Series(returns).dropna()
    equity = pd.Series(equity).dropna()

    if returns.empty:
        return _empty_metrics()

    # ------------------------------------------------------------------ #
    # Return / risk
    # ------------------------------------------------------------------ #
    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
    n_periods = len(returns)
    years = n_periods / periods_per_year
    # CAGR is undefined for negative or zero ending equity
    if equity.iloc[-1] > 0 and equity.iloc[0] > 0:
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(years, 1e-9)) - 1.0)
    else:
        cagr = float("nan")

    excess = returns - (risk_free_rate / periods_per_year)
    vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(excess.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year)) if returns.std(ddof=0) > 0 else 0.0

    downside = returns.clip(upper=0.0)
    downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))
    sortino = float(excess.mean() * np.sqrt(periods_per_year) / downside_std) if downside_std > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Drawdown
    # ------------------------------------------------------------------ #
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min())

    # Duration of max drawdown
    dd_duration = 0
    current = 0
    for v in drawdown.values:
        if v < 0:
            current += 1
            dd_duration = max(dd_duration, current)
        else:
            current = 0

    calmar = float(cagr / abs(max_dd)) if max_dd < 0 and not (cagr != cagr) else 0.0  # NaN-safe

    # ------------------------------------------------------------------ #
    # Trade-based metrics
    # ------------------------------------------------------------------ #
    if "pnl" in trades.columns and len(trades) > 0:
        pnl = trades["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
        avg_trade = float(pnl.mean())
        n_trades = int(len(pnl))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
        n_trades = 0

    # Exposure: fraction of bars where we held a position
    # (we approximate this from trades: 1 - flat bars / total bars)
    exposure = float((returns != 0).mean())

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=int(dd_duration),
        volatility=vol,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade,
        n_trades=n_trades,
        exposure=exposure,
    )


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.0,
        cagr=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration=0,
        volatility=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_return=0.0,
        n_trades=0,
        exposure=0.0,
    )
