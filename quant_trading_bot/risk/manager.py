"""Risk management helpers.

This module is intentionally side-effect-free: it consumes a price
series, a raw signal and the current portfolio state, and returns a
sized, risk-adjusted target position.  The backtester (or live trader)
is responsible for actually executing that target.

Implemented controls
--------------------
* Position sizing: Kelly criterion, fixed fractional, equal weight.
* Stop loss / take profit (per trade).
* Trailing stop (per open position).
* Portfolio-level drawdown kill switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class PositionSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass
class PortfolioState:
    """Mutable container describing the current portfolio."""

    cash: float
    equity: float
    position_qty: float = 0.0
    position_avg_price: float = 0.0
    peak_equity: float = 0.0
    side: PositionSide = PositionSide.FLAT

    def __post_init__(self) -> None:
        if self.peak_equity == 0.0:
            self.peak_equity = self.equity


class RiskManager:
    """Apply position sizing and stop-loss / take-profit rules."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_sizing: str = "fixed_fractional",
        kelly_fraction: float = 0.25,
        fixed_fraction: float = 0.10,
        max_position_pct: float = 0.20,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        trailing_stop_pct: float = 0.07,
        max_drawdown_pct: float = 0.20,
    ) -> None:
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.kelly_fraction = kelly_fraction
        self.fixed_fraction = fixed_fraction
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct

        self.state = PortfolioState(cash=initial_capital, equity=initial_capital)
        self.highest_since_entry: Optional[float] = None
        self.lowest_since_entry: Optional[float] = None
        self.trading_halted: bool = False  # set by drawdown kill switch
        self.halt_reason: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Position sizing
    # ------------------------------------------------------------------ #
    def size_position(
        self, price: float, signal: int, win_rate: float = 0.55, payoff: float = 1.5
    ) -> float:
        """Return the dollar value to allocate for ``signal`` in {-1, 0, +1}."""
        if self.trading_halted or signal == 0 or price <= 0:
            return 0.0
        equity = self.state.equity
        if equity <= 0:
            return 0.0

        if self.position_sizing == "kelly":
            # Full Kelly: f* = (p*b - q) / b,  b = payoff ratio
            p = np.clip(win_rate, 0.01, 0.99)
            q = 1.0 - p
            b = max(payoff, 0.01)
            kelly = max(0.0, (p * b - q) / b)
            fraction = self.kelly_fraction * kelly
        elif self.position_sizing == "fixed_fractional":
            fraction = self.fixed_fraction
        elif self.position_sizing == "equal_weight":
            fraction = 1.0  # caller will scale to position cap
        else:
            raise ValueError(f"Unknown position_sizing='{self.position_sizing}'")

        fraction = min(fraction, self.max_position_pct)
        return equity * fraction

    # ------------------------------------------------------------------ #
    # Stop checks (return True if position should be closed)
    # ------------------------------------------------------------------ #
    def check_exit(self, price: float) -> Optional[str]:
        """Return a reason string if the current position should be closed."""
        if self.state.side == PositionSide.FLAT:
            return None

        # Portfolio-level drawdown kill switch (latches on first trigger)
        if not self.trading_halted and self.state.equity <= self.state.peak_equity * (1.0 - self.max_drawdown_pct):
            self.trading_halted = True
            self.halt_reason = "max_drawdown"
            return "max_drawdown"

        if self.state.position_avg_price <= 0:
            return None

        if self.state.side == PositionSide.LONG:
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price
            change = (price - self.state.position_avg_price) / self.state.position_avg_price
            if change <= -self.stop_loss_pct:
                return "stop_loss"
            if change >= self.take_profit_pct:
                return "take_profit"
            if (
                self.highest_since_entry is not None
                and self.highest_since_entry > 0
                and price <= self.highest_since_entry * (1.0 - self.trailing_stop_pct)
            ):
                return "trailing_stop"
        else:  # SHORT
            if self.lowest_since_entry is None or price < self.lowest_since_entry:
                self.lowest_since_entry = price
            change = (self.state.position_avg_price - price) / self.state.position_avg_price
            if change <= -self.stop_loss_pct:
                return "stop_loss"
            if change >= self.take_profit_pct:
                return "take_profit"
            if (
                self.lowest_since_entry is not None
                and self.lowest_since_entry > 0
                and price >= self.lowest_since_entry * (1.0 + self.trailing_stop_pct)
            ):
                return "trailing_stop"
        return None

    # ------------------------------------------------------------------ #
    # State helpers used by the backtester
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = PortfolioState(cash=self.initial_capital, equity=self.initial_capital)
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.trading_halted = False
        self.halt_reason = None
