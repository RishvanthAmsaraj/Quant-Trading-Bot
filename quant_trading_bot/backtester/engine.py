"""Vectorised event-driven backtester.

The engine walks through a price series bar by bar, applies the
strategy signal, sizes the position with the :class:`RiskManager`,
charges commissions/slippage and keeps a trade log.  It is intentionally
simple so that the behaviour is easy to reason about and audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..risk.manager import PositionSide, RiskManager
from ..strategies.base import Strategy
from ..utils.logging import get_logger
from .metrics import PerformanceMetrics, compute_metrics

LOGGER = get_logger(__name__)


@dataclass
class BacktestResult:
    """Structured result of a backtest run."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: PerformanceMetrics
    strategy_name: str
    ticker: str
    params: Dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"\n=== Backtest: {self.strategy_name} on {self.ticker} ===\n"
            f"  Total return : {m.total_return * 100:8.2f}%\n"
            f"  CAGR         : {m.cagr * 100:8.2f}%\n"
            f"  Sharpe ratio : {m.sharpe_ratio:8.3f}\n"
            f"  Sortino ratio: {m.sortino_ratio:8.3f}\n"
            f"  Max drawdown : {m.max_drawdown * 100:8.2f}%\n"
            f"  Win rate     : {m.win_rate * 100:8.2f}%\n"
            f"  Profit factor: {m.profit_factor:8.3f}\n"
            f"  N trades     : {m.n_trades:8d}\n"
            f"  Exposure     : {m.exposure * 100:8.2f}%\n"
        )


class Backtester:
    """Event-driven backtester for a single instrument."""

    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        exit_cooldown_bars: int = 2,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        # Bars to stay flat before re-entering the SAME side after an exit.
        # Kills stop-loss churn: without it a persistent signal re-opens the
        # position right after every stop, paying commissions on whipsaws.
        self.exit_cooldown_bars = exit_cooldown_bars

    # ------------------------------------------------------------------ #
    def run(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """Run the backtest and return a :class:`BacktestResult`."""
        suffix = f"_{ticker}" if ticker else ""
        close_col = f"Close{suffix}" if f"Close{suffix}" in df.columns else "Close"
        if close_col not in df.columns:
            raise KeyError(f"Close column '{close_col}' not in DataFrame")

        prices = df[close_col].astype(float).copy()
        signals = self.strategy.generate_signals(df, ticker).astype(int).reindex(prices.index).fillna(0)

        if initial_capital is not None:
            self.risk_manager.initial_capital = initial_capital
        self.risk_manager.reset()

        equity = pd.Series(index=prices.index, dtype=float)
        positions = pd.Series(0, index=prices.index, dtype=int)
        returns = pd.Series(0.0, index=prices.index, dtype=float)
        trades: List[Dict] = []

        prev_equity = self.risk_manager.state.equity
        entry_price: Optional[float] = None
        entry_time: Optional[pd.Timestamp] = None
        entry_qty: float = 0.0
        # Re-entry cooldown state: bar index + side of the last exit
        last_exit_bar: int = -10**9
        last_exit_side: int = 0

        for bar_i, (ts, price) in enumerate(prices.items()):
            if np.isnan(price):
                equity.loc[ts] = prev_equity
                continue

            # 1) Update mark-to-market equity
            mtm_equity = self._mark_to_market(price)
            equity.loc[ts] = mtm_equity
            self.risk_manager.state.equity = mtm_equity
            self.risk_manager.state.peak_equity = max(
                self.risk_manager.state.peak_equity, mtm_equity
            )

            # 2) Check exit conditions first (stops / targets)
            exit_reason = self.risk_manager.check_exit(price)
            if exit_reason and self.risk_manager.state.side != PositionSide.FLAT:
                last_exit_side = self._side_value(self.risk_manager.state.side)
                last_exit_bar = bar_i
                pnl = self._close_position(price, ts, reason=exit_reason, trades=trades, entry_price=entry_price, entry_time=entry_time, qty=entry_qty)
                entry_price = None
                entry_time = None
                entry_qty = 0.0

            # 3) Generate desired position
            target_signal = int(signals.loc[ts])
            # Cooldown: don't re-open the same side right after an exit
            # (prevents churning into the same signal post stop/target).
            if (
                self.exit_cooldown_bars > 0
                and target_signal != 0
                and target_signal == last_exit_side
                and (bar_i - last_exit_bar) < self.exit_cooldown_bars
            ):
                target_signal = 0
            if self.risk_manager.state.side == PositionSide.FLAT and target_signal != 0:
                size_value = self.risk_manager.size_position(price, target_signal)
                qty = size_value / max(price, 1e-9)
                self._open_position(price, ts, target_signal, qty, trades=trades)
                entry_price = price * (1 + np.sign(target_signal) * self.slippage)
                entry_time = ts
                entry_qty = qty

            positions.loc[ts] = (
                1 if self.risk_manager.state.side == PositionSide.LONG
                else -1 if self.risk_manager.state.side == PositionSide.SHORT
                else 0
            )

            # 4) Compute per-bar return
            if prev_equity > 0:
                returns.loc[ts] = (mtm_equity / prev_equity) - 1.0
            prev_equity = mtm_equity

        # If still in a position at end of test, close it at last price
        if self.risk_manager.state.side != PositionSide.FLAT and not np.isnan(prices.iloc[-1]):
            self._close_position(prices.iloc[-1], prices.index[-1], reason="end_of_data",
                                 trades=trades, entry_price=entry_price, entry_time=entry_time, qty=entry_qty)

        equity = equity.ffill().fillna(self.risk_manager.initial_capital)
        trades_df = pd.DataFrame(trades)

        metrics = compute_metrics(
            returns=returns,
            trades=trades_df,
            equity=equity,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.trading_days_per_year,
        )

        return BacktestResult(
            equity=equity,
            returns=returns,
            positions=positions,
            trades=trades_df,
            metrics=metrics,
            strategy_name=self.strategy.metadata.name,
            ticker=ticker,
            params=self.strategy.metadata.parameters,
        )

    # ------------------------------------------------------------------ #
    # Position helpers
    # ------------------------------------------------------------------ #
    def _mark_to_market(self, price: float) -> float:
        s = self.risk_manager.state
        if s.side == PositionSide.FLAT or s.position_qty == 0:
            return s.cash
        if s.side == PositionSide.LONG:
            return s.cash + s.position_qty * price
        # SHORT: cash already holds the sale proceeds (+qty * entry); the
        # liability is the current cost to buy the shares back.
        return s.cash - s.position_qty * price

    def _open_position(
        self, price: float, ts: pd.Timestamp, signal: int, qty: float, trades: List[Dict]
    ) -> None:
        if qty <= 0:
            return
        fill_price = price * (1 + np.sign(signal) * self.slippage)
        cost = qty * fill_price
        commission = cost * self.commission
        s = self.risk_manager.state
        if signal > 0:
            # LONG: buy the shares — cash out.
            s.cash -= cost + commission
        else:
            # SHORT: sell borrowed shares — the sale proceeds arrive as cash.
            s.cash += cost - commission
        s.position_qty = qty
        s.position_avg_price = fill_price
        s.side = PositionSide.LONG if signal > 0 else PositionSide.SHORT
        self.risk_manager.highest_since_entry = fill_price
        self.risk_manager.lowest_since_entry = fill_price
        # Keep the opening leg on the engine (not the log): a completed row
        # is appended once at close so the log/win-rate see one row per trade.
        self._open_trade = {
            "entry_date": ts,
            "side": s.side.value,
            "entry_price": fill_price,
            "qty": qty,
        }

    def _close_position(
        self,
        price: float,
        ts: pd.Timestamp,
        reason: str,
        trades: List[Dict],
        entry_price: Optional[float],
        entry_time: Optional[pd.Timestamp],
        qty: float,
    ) -> float:
        s = self.risk_manager.state
        if s.side == PositionSide.FLAT or qty <= 0:
            return 0.0
        fill_price = price * (1 - np.sign(self._side_value(s.side)) * self.slippage)
        proceeds = qty * fill_price
        commission = proceeds * self.commission
        if s.side == PositionSide.LONG:
            pnl = (fill_price - entry_price) * qty - commission
            s.cash += proceeds - commission
        else:  # SHORT: buy the shares back — cash out.
            pnl = (entry_price - fill_price) * qty - commission
            s.cash -= proceeds + commission
        return_pct = pnl / max(entry_price * qty, 1e-9)
        trades.append(
            {
                "entry_date": entry_time,
                "exit_date": ts,
                "side": s.side.value,
                "entry_price": entry_price,
                "exit_price": fill_price,
                "qty": qty,
                "pnl": pnl,
                "return_pct": return_pct,
                "reason": reason,
            }
        )
        s.position_qty = 0.0
        s.position_avg_price = 0.0
        s.side = PositionSide.FLAT
        self.risk_manager.highest_since_entry = None
        self.risk_manager.lowest_since_entry = None
        return pnl

    @staticmethod
    def _side_value(side: PositionSide) -> int:
        return 1 if side == PositionSide.LONG else -1
