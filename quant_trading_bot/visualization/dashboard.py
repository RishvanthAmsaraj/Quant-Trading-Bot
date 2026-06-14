"""Visualisation utilities (matplotlib + plotly).

The dashboard produces a small, consistent set of figures that explain
what a backtest did:

* Price chart with indicators and buy/sell markers
* Equity curve vs. buy & hold
* Drawdown curve
* Rolling Sharpe ratio
* Performance summary table
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtester.engine import BacktestResult
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class Dashboard:
    """Persist a standard set of backtest visualisations."""

    def __init__(self, output_dir: str | Path = "results/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"figure.figsize": (12, 7), "axes.grid": True})

    # ------------------------------------------------------------------ #
    def save_all(
        self,
        result: BacktestResult,
        benchmark_equity: Optional[pd.Series] = None,
        title_suffix: str = "",
    ) -> list[Path]:
        """Save every figure for a backtest and return the file paths."""
        saved: list[Path] = []
        for fn, kwargs in [
            (self.plot_price_with_signals, {"result": result}),
            (self.plot_equity_vs_benchmark, {"result": result, "benchmark_equity": benchmark_equity}),
            (self.plot_drawdown, {"result": result}),
            (self.plot_rolling_sharpe, {"result": result}),
        ]:
            try:
                path = fn(**kwargs)
                if path is not None:
                    saved.append(path)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to render %s: %s", fn.__name__, exc)
        LOGGER.info("Saved %d dashboard figures to %s", len(saved), self.output_dir)
        return saved

    # ------------------------------------------------------------------ #
    def plot_price_with_signals(self, result: BacktestResult) -> Path:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(result.equity.index, result.equity, color="#1f77b4", label="Equity")

        if not result.trades.empty:
            for _, tr in result.trades.iterrows():
                color = "#2ca02c" if tr["pnl"] > 0 else "#d62728"
                ax.axvline(tr["exit_time"], color=color, alpha=0.08, linewidth=1)

        ax.set_title(f"Equity curve — {result.strategy_name} on {result.ticker}")
        ax.set_ylabel("Portfolio value ($)")
        ax.legend()
        fig.tight_layout()
        path = self.output_dir / f"equity_{result.strategy_name}_{result.ticker}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def plot_equity_vs_benchmark(
        self,
        result: BacktestResult,
        benchmark_equity: Optional[pd.Series] = None,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(result.equity.index, result.equity, label="Strategy", linewidth=2)
        if benchmark_equity is not None and len(benchmark_equity) > 0:
            scaled = benchmark_equity / benchmark_equity.iloc[0] * result.equity.iloc[0]
            ax.plot(scaled.index, scaled, label="Buy & Hold", linewidth=1.5, linestyle="--")
        ax.set_title("Strategy vs. Buy & Hold")
        ax.set_ylabel("Portfolio value ($)")
        ax.legend()
        fig.tight_layout()
        path = self.output_dir / f"benchmark_{result.strategy_name}_{result.ticker}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def plot_drawdown(self, result: BacktestResult) -> Path:
        fig, ax = plt.subplots(figsize=(13, 4))
        running_max = result.equity.cummax()
        drawdown = result.equity / running_max - 1.0
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.4)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown %")
        fig.tight_layout()
        path = self.output_dir / f"drawdown_{result.strategy_name}_{result.ticker}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def plot_rolling_sharpe(self, result: BacktestResult, window: int = 63) -> Path:
        fig, ax = plt.subplots(figsize=(13, 4))
        rolling = (
            result.returns.rolling(window).mean() / (result.returns.rolling(window).std() + 1e-9)
        ) * np.sqrt(252)
        ax.plot(rolling.index, rolling, label=f"Rolling Sharpe ({window}d)", color="#9467bd")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title("Rolling Sharpe Ratio")
        ax.legend()
        fig.tight_layout()
        path = self.output_dir / f"rolling_sharpe_{result.strategy_name}_{result.ticker}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def plot_indicators(self, df: pd.DataFrame, ticker: str = "") -> Path:
        """Static helper that plots price + Bollinger + RSI + MACD."""
        suffix = f"_{ticker}" if ticker else ""
        close = df[f"Close{suffix}"] if f"Close{suffix}" in df.columns else df["Close"]
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        axes[0].plot(close.index, close, label="Close", color="#1f77b4")
        for col in [f"BB_upper{suffix}", f"BB_middle{suffix}", f"BB_lower{suffix}"]:
            if col in df.columns:
                axes[0].plot(df.index, df[col], label=col.replace(suffix, ""), linewidth=0.8)
        axes[0].set_title(f"Price & Bollinger Bands ({ticker or 'asset'})")
        axes[0].legend(loc="upper left", fontsize=8)

        if f"RSI{suffix}" in df.columns:
            axes[1].plot(df.index, df[f"RSI{suffix}"], color="#ff7f0e")
            axes[1].axhline(70, color="red", linestyle="--", linewidth=0.7)
            axes[1].axhline(30, color="green", linestyle="--", linewidth=0.7)
            axes[1].set_title("RSI")
            axes[1].set_ylim(0, 100)

        if f"MACD{suffix}" in df.columns:
            axes[2].plot(df.index, df[f"MACD{suffix}"], label="MACD", color="#1f77b4")
            axes[2].plot(df.index, df[f"MACD_signal{suffix}"], label="Signal", color="#ff7f0e")
            axes[2].bar(df.index, df[f"MACD_hist{suffix}"], label="Hist", color="grey", alpha=0.4)
            axes[2].set_title("MACD")
            axes[2].legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        path = self.output_dir / f"indicators_{ticker or 'asset'}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def plotly_dashboard(
        self,
        result: BacktestResult,
        benchmark_equity: Optional[pd.Series] = None,
        filename: str = "dashboard.html",
    ) -> Optional[Path]:
        """Build an interactive plotly dashboard.  Returns ``None`` if plotly
        is not installed."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            LOGGER.warning("plotly is not installed - skipping interactive dashboard")
            return None

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.05,
            subplot_titles=("Equity vs Buy & Hold", "Drawdown", "Rolling Sharpe (63d)"),
        )

        fig.add_trace(
            go.Scatter(x=result.equity.index, y=result.equity, name="Strategy", line=dict(color="#1f77b4")),
            row=1, col=1,
        )
        if benchmark_equity is not None and len(benchmark_equity) > 0:
            scaled = benchmark_equity / benchmark_equity.iloc[0] * result.equity.iloc[0]
            fig.add_trace(
                go.Scatter(x=scaled.index, y=scaled, name="Buy & Hold", line=dict(dash="dash", color="grey")),
                row=1, col=1,
            )

        running_max = result.equity.cummax()
        drawdown = result.equity / running_max - 1.0
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown", line=dict(color="#d62728")),
            row=2, col=1,
        )

        rolling = (
            result.returns.rolling(63).mean() / (result.returns.rolling(63).std() + 1e-9)
        ) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling.index, y=rolling, name="Rolling Sharpe", line=dict(color="#9467bd")),
            row=3, col=1,
        )

        fig.update_layout(
            title=f"Quant Trading Bot Dashboard — {result.strategy_name} on {result.ticker}",
            height=900,
            template="plotly_white",
        )
        out_path = self.output_dir / filename
        fig.write_html(out_path)
        LOGGER.info("Saved interactive dashboard to %s", out_path)
        return out_path
