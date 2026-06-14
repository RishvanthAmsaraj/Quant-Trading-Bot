"""
Quant Trading Bot — Quick Walkthrough
=====================================

This is the *executable* companion to the README.  It shows, in a few
cells, how to:

1. Load a single ticker with ``MarketDataLoader``
2. Add technical indicators
3. Run a momentum strategy through the backtester
4. Plot the result

Run it with::

    python quant_trading_bot/notebooks/example_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when the file is executed directly
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from quant_trading_bot.backtester import Backtester  # noqa: E402
from quant_trading_bot.data import MarketDataLoader  # noqa: E402
from quant_trading_bot.indicators import add_all_indicators  # noqa: E402
from quant_trading_bot.risk import RiskManager  # noqa: E402
from quant_trading_bot.strategies import MomentumStrategy  # noqa: E402


def main() -> None:
    print(">> Loading data …")
    loader = MarketDataLoader(cache_dir="data/cache")
    raw = loader.download("AAPL", period="1y", interval="1d")

    # For a single ticker the loader returns suffixed columns; strip suffix
    raw.columns = [c.replace("_AAPL", "") for c in raw.columns]

    print(">> Computing indicators …")
    df = add_all_indicators(raw)

    print(">> Backtesting momentum strategy …")
    rm = RiskManager(initial_capital=100_000, position_sizing="fixed_fractional", fixed_fraction=0.2)
    bt = Backtester(strategy=MomentumStrategy(lookback=20, threshold=0.02), risk_manager=rm)
    result = bt.run(df)
    print(result.summary())

    print(">> Plotting …")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.equity.index, result.equity, label="Momentum equity")
    benchmark = raw["Close"] * (100_000 / raw["Close"].iloc[0])
    ax.plot(benchmark.index, benchmark, label="Buy & Hold", linestyle="--")
    ax.set_title("Momentum Strategy — AAPL")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    fig.tight_layout()
    out = Path("results/plots/example_walkthrough.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f">> Saved chart to {out}")


if __name__ == "__main__":
    main()
