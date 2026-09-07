"""Engine sanity checks — verifies the backtester's core accounting.

Run with:  python scripts/engine_sanity_check.py

Each case uses a synthetic price series and a fixed-signal strategy so the
expected outcome is computable by hand. Failures mean the engine's math is
broken (cash accounting, mark-to-market, trade log, or win-rate bookkeeping).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_trading_bot.backtester.engine import Backtester
from quant_trading_bot.risk.manager import RiskManager
from quant_trading_bot.strategies.base import Strategy, StrategyMetadata

PASS, FAIL = "PASS", "FAIL"
failures: list[str] = []


def check(label: str, actual: float, expected: float, tol: float = 0.01) -> None:
    ok = abs(actual - expected) <= tol
    print(f"  [{PASS if ok else FAIL}] {label}: got {actual:.4f}, expected {expected:.4f}")
    if not ok:
        failures.append(label)


class FixedSignals(Strategy):
    def __init__(self, name: str, sig: int) -> None:
        self.sig = sig
        self.metadata = StrategyMetadata(name=name, family="test", parameters={})

    def generate_signals(self, df: pd.DataFrame, ticker: str = "") -> pd.Series:
        return pd.Series(self.sig, index=df.index, dtype=int)


def run(sig: int, prices: list[float], commission: float = 0.0, slippage: float = 0.0):
    df = pd.DataFrame({"Close": prices})
    bt = Backtester(
        FixedSignals(f"fixed_{sig}", sig),
        RiskManager(initial_capital=100_000),
        commission=commission,
        slippage=slippage,
    )
    return bt.run(df, ticker="")


def main() -> int:
    n = 60

    # ---- 1. Short through a clean 100->50 decline, no costs ----
    print("1. Short through a 100->50 decline (no costs)")
    res = run(-1, np.linspace(100.0, 50.0, n).tolist())
    # 10% of equity per trade, TP legs of +15% each with the 2-bar
    # re-entry cooldown -> 4 trades, ~+5.84% total
    check("total_return ≈ +5.84%", res.metrics.total_return, 0.0584, tol=0.004)
    check("win_rate = 100%", res.metrics.win_rate, 1.0)
    check("one row per trade (no phantom 'open' rows)",
          float(len(res.trades)), 4.0, tol=1.0)
    if len(res.trades):
        check("trade pnl > 0 (shorts profit in a decline)",
              float(res.trades["pnl"].iloc[0]), 1525.42, tol=40.0)

    # ---- 2. Short through a 100->150 rise, no costs (bounded losses) ----
    print("2. Short through a 100->150 rise (no costs)")
    res = run(-1, np.linspace(100.0, 150.0, n).tolist())
    check("total_return ≈ -3.27% (stop-managed)", res.metrics.total_return, -0.0327, tol=0.004)
    check("equity stays positive", float(res.equity.min()), 96725.49, tol=200.0)

    # ---- 3. Mixed flat-ish prices WITH costs: continuity + hand-check ----
    print("3. Mixed prices with costs (hand-verifiable round trip)")
    prices = [100.0] * 10 + [90.0] * 10 + [95.0] * 10
    res = run(-1, prices, commission=0.001, slippage=0.0005)
    check("no accounting jumps (|bar return| <= 15%)",
          float((np.abs(res.returns) > 0.15).sum()), 0.0)
    check("total_return ≈ +0.48%", res.metrics.total_return, 0.0048, tol=0.0015)
    check("exactly 1 trade", float(len(res.trades)), 1.0)
    row = res.trades.iloc[0]
    # entry 100, exit 95, qty = 10000/100 = 100 shares
    # pnl = (100-95)*100 - commission(~9.5 entry + ~9.55 exit) ~= 480.95
    check("trade pnl ≈ +480.95", float(row["pnl"]), 480.95, tol=2.0)
    check("dates present", int(("entry_date" in res.trades.columns) and ("exit_date" in res.trades.columns)), 1.0, tol=0.0)

    # ---- 4. Long regression guard ----
    print("4. Long through a 100->150 rise (no costs)")
    res = run(1, np.linspace(100.0, 150.0, n).tolist())
    check("total_return ≈ +4.09%", res.metrics.total_return, 0.0409, tol=0.004)
    check("win_rate = 100%", res.metrics.win_rate, 1.0)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("All engine sanity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
