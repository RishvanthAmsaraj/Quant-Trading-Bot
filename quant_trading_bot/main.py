"""End-to-end orchestrator for the quant trading bot.

The :func:`run` function wires together every component of the system:

1. Load configuration
2. Download OHLCV data (with caching)
3. Compute technical indicators
4. Train the LSTM model (if enabled)
5. Generate signals for each strategy
6. Run backtests with risk controls
7. Compute performance metrics
8. Render matplotlib + plotly dashboards
9. Persist results to disk

The CLI exposes convenient flags for picking tickers, strategies and
whether to train the LSTM.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .backtester import Backtester
from .data import MarketDataLoader
from .indicators import add_all_indicators
from .ml import LSTMStrategy
from .risk import RiskManager
from .strategies import (
    EnsembleStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from .utils import get_logger, load_config
from .visualization import Dashboard

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Strategy factory
# --------------------------------------------------------------------------- #
def _build_strategies(cfg) -> Dict[str, object]:
    s = cfg.strategies
    strategies: Dict[str, object] = {
        "Momentum": MomentumStrategy(
            lookback=s["momentum"]["lookback"],
            threshold=s["momentum"]["threshold"],
        ),
        "MeanReversion": MeanReversionStrategy(
            rsi_oversold=s["mean_reversion"]["rsi_oversold"],
            rsi_overbought=s["mean_reversion"]["rsi_overbought"],
            bb_threshold=s["mean_reversion"]["bb_threshold"],
        ),
        "TrendFollowing": TrendFollowingStrategy(
            fast_ma=s["trend_following"]["fast_ma"],
            slow_ma=s["trend_following"]["slow_ma"],
            adx_threshold=s["trend_following"]["adx_threshold"],
        ),
    }
    strategies["Ensemble"] = EnsembleStrategy(
        [strategies["Momentum"], strategies["TrendFollowing"]],  # type: ignore[list-item]
        min_agreement=1,
    )
    return strategies


# --------------------------------------------------------------------------- #
def _build_risk_manager(cfg) -> RiskManager:
    r = cfg.risk
    return RiskManager(
        initial_capital=r["initial_capital"],
        position_sizing=r["position_sizing"],
        kelly_fraction=r["kelly_fraction"],
        fixed_fraction=r["fixed_fraction"],
        max_position_pct=r["max_position_pct"],
        stop_loss_pct=r["stop_loss_pct"],
        take_profit_pct=r["take_profit_pct"],
        trailing_stop_pct=r["trailing_stop_pct"],
        max_drawdown_pct=r["max_drawdown_pct"],
    )


# --------------------------------------------------------------------------- #
def run(
    config_path: str = "configs/config.yaml",
    tickers: List[str] | None = None,
    strategies: List[str] | None = None,
    train_lstm: bool | None = None,
    output_dir: str = "results",
) -> Dict[str, dict]:
    """Run the full quant-trading pipeline."""
    cfg = load_config(config_path)
    tickers = tickers or cfg.data["tickers"]
    strategies_requested = strategies or ["Momentum", "MeanReversion", "TrendFollowing", "Ensemble"]
    if train_lstm is None:
        train_lstm = cfg.ml.get("enabled", False)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_dir = output_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading data for %d ticker(s): %s", len(tickers), tickers)
    loader = MarketDataLoader(cache_dir=cfg.data["cache_dir"])
    raw = loader.download(
        tickers=tickers,
        period=cfg.data["period"],
        interval=cfg.data["interval"],
    )

    # Per-ticker dataframes with indicators
    ind_cfg = cfg.indicators
    per_ticker: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        suffix = f"_{t}"
        cols = [c for c in raw.columns if c.endswith(suffix)]
        if not cols:
            LOGGER.warning("No columns found for %s - skipping", t)
            continue
        df_t = raw[cols].copy()
        df_t.columns = [c.replace(suffix, "") for c in cols]
        df_t = add_all_indicators(
            df_t, ticker="",
            sma_fast=ind_cfg["sma_fast"],
            sma_slow=ind_cfg["sma_slow"],
            ema_fast=ind_cfg["ema_fast"],
            ema_slow=ind_cfg["ema_slow"],
            rsi_period=ind_cfg["rsi_period"],
            macd_signal=ind_cfg["macd_signal"],
            bb_period=ind_cfg["bb_period"],
            bb_std=ind_cfg["bb_std"],
            atr_period=ind_cfg["atr_period"],
            adx_period=ind_cfg["adx_period"],
        )
        per_ticker[t] = df_t

    # Strategies + risk
    strat_map = _build_strategies(cfg)
    chosen = {name: strat_map[name] for name in strategies_requested if name in strat_map}
    if not chosen:
        raise ValueError(f"No valid strategies requested: {strategies_requested}")

    # Optional LSTM
    if train_lstm and "LSTM" not in chosen:
        first_ticker = next(iter(per_ticker))
        try:
            predictor = _build_lstm(cfg)
            LOGGER.info("Training LSTM on %s …", first_ticker)
            lstm_strat = LSTMStrategy(predictor=predictor)
            lstm_strat.train(
                per_ticker[first_ticker]["Close"],
                epochs=cfg.ml["epochs"],
                batch_size=cfg.ml["batch_size"],
                train_split=cfg.ml["train_split"],
                early_stopping_patience=cfg.ml["early_stopping_patience"],
            )
            chosen["LSTM"] = lstm_strat
        except Exception as exc:  # pragma: no cover
            LOGGER.error("LSTM training failed: %s", exc)

    # Backtest loop
    risk = _build_risk_manager(cfg)
    backtester = Backtester(
        strategy=None,  # type: ignore[arg-type]
        risk_manager=risk,
        commission=cfg.backtest["commission"],
        slippage=cfg.backtest["slippage"],
        risk_free_rate=cfg.backtest["risk_free_rate"],
        trading_days_per_year=cfg.backtest["trading_days_per_year"],
    )

    dashboard = Dashboard(output_dir=plot_dir)
    results_summary: Dict[str, dict] = {}

    for t, df_t in per_ticker.items():
        benchmark_equity = df_t["Close"] * (cfg.risk["initial_capital"] / df_t["Close"].iloc[0])
        for name, strat in chosen.items():
            backtester.strategy = strat
            try:
                result = backtester.run(df_t, ticker=t)
            except Exception as exc:  # pragma: no cover
                LOGGER.error("Backtest failed for %s / %s: %s", name, t, exc)
                continue
            print(result.summary())
            dashboard.save_all(result, benchmark_equity=benchmark_equity)
            if cfg.visualization.get("plotly_dashboard", True):
                dashboard.plotly_dashboard(result, benchmark_equity=benchmark_equity,
                                          filename=f"dashboard_{name}_{t}.html")
            results_summary[f"{name}__{t}"] = {
                "strategy": name,
                "ticker": t,
                "metrics": asdict(result.metrics),
                "params": result.params,
            }

    # Save full summary
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2, default=str))
    LOGGER.info("Wrote summary to %s", summary_path)
    return results_summary


def _build_lstm(cfg):
    from .ml import LSTMPricePredictor

    m = cfg.ml
    return LSTMPricePredictor(
        lookback=m["lookback"],
        hidden_units=m["hidden_units"],
        dropout=m["dropout"],
        learning_rate=m["learning_rate"],
    )


# --------------------------------------------------------------------------- #
def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the quant trading bot")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    p.add_argument("--tickers", nargs="*", help="Tickers to trade (overrides config)")
    p.add_argument(
        "--strategies",
        nargs="*",
        choices=["Momentum", "MeanReversion", "TrendFollowing", "Ensemble"],
        help="Strategies to run",
    )
    p.add_argument("--lstm", dest="lstm", action="store_true", help="Train the LSTM model")
    p.add_argument("--no-lstm", dest="lstm", action="store_false", help="Skip the LSTM model")
    p.set_defaults(lstm=None)
    p.add_argument("--output", default="results", help="Output directory")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(
            config_path=args.config,
            tickers=args.tickers,
            strategies=args.strategies,
            train_lstm=args.lstm,
            output_dir=args.output,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
