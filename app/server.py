"""
Quant Trading Bot — Flask Backend Server

Provides the backtest API endpoint and serves the frontend.
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from quant_trading_bot.data import MarketDataLoader
from quant_trading_bot.indicators import add_all_indicators
from quant_trading_bot.strategies import (
    EnsembleStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from quant_trading_bot.risk import RiskManager
from quant_trading_bot.backtester import Backtester
from quant_trading_bot.utils import load_config

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

cfg_path = _PROJECT_DIR / "configs" / "config.yaml"
cfg = load_config(str(cfg_path))


# -------------------------------------------------------------------
# Serve the main page
# -------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(_APP_DIR), "index.html")


# -------------------------------------------------------------------
# Serve Plotly dashboards
# -------------------------------------------------------------------
@app.route("/plots/<path:filename>")
def serve_plot(filename):
    plots_dir = _PROJECT_DIR / "results" / "plots"
    return send_from_directory(str(plots_dir), filename)


# -------------------------------------------------------------------
# Backtest API
# -------------------------------------------------------------------
@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    data = request.get_json(silent=True) or {}

    tickers = [t.strip().upper() for t in data.get("tickers", ",".join(cfg.data["tickers"])).split(",") if t.strip()]
    period = data.get("period", "1y")
    strategy_names = data.get("strategies", ["Momentum", "MeanReversion", "TrendFollowing", "Ensemble"])
    initial_capital = float(data.get("initial_capital", cfg.risk["initial_capital"]))
    position_sizing = data.get("position_sizing", cfg.risk["position_sizing"])
    stop_loss = float(data.get("stop_loss", cfg.risk["stop_loss_pct"]))
    take_profit = float(data.get("take_profit", cfg.risk["take_profit_pct"]))

    try:
        # --- Load data ---
        loader = MarketDataLoader(cache_dir=str(_PROJECT_DIR / "data" / "cache"))
        raw = loader.download(tickers=tickers, period=period)

        ind_cfg = cfg.indicators
        per_ticker = {}
        for t in tickers:
            suffix = f"_{t}"
            cols = [c for c in raw.columns if c.endswith(suffix)]
            if not cols:
                continue
            df_t = raw[cols].copy()
            df_t = add_all_indicators(
                df_t, ticker=t,
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

        # --- Build strategies ---
        strategy_map = {
            "Momentum": MomentumStrategy(
                lookback=cfg.strategies["momentum"]["lookback"],
                threshold=cfg.strategies["momentum"]["threshold"],
            ),
            "MeanReversion": MeanReversionStrategy(
                rsi_oversold=cfg.strategies["mean_reversion"]["rsi_oversold"],
                rsi_overbought=cfg.strategies["mean_reversion"]["rsi_overbought"],
                bb_threshold=cfg.strategies["mean_reversion"]["bb_threshold"],
            ),
            "TrendFollowing": TrendFollowingStrategy(
                fast_ma=cfg.strategies["trend_following"]["fast_ma"],
                slow_ma=cfg.strategies["trend_following"]["slow_ma"],
                adx_threshold=cfg.strategies["trend_following"]["adx_threshold"],
            ),
        }
        strategy_map["Ensemble"] = EnsembleStrategy(
            [strategy_map["Momentum"], strategy_map["TrendFollowing"]],
            min_agreement=1,
        )
        chosen = {n: strategy_map[n] for n in strategy_names if n in strategy_map}

        # --- Risk ---
        risk_mgr = RiskManager(
            initial_capital=initial_capital,
            position_sizing=position_sizing,
            kelly_fraction=cfg.risk["kelly_fraction"],
            fixed_fraction=cfg.risk["fixed_fraction"],
            max_position_pct=cfg.risk["max_position_pct"],
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            trailing_stop_pct=cfg.risk["trailing_stop_pct"],
            max_drawdown_pct=cfg.risk["max_drawdown_pct"],
        )

        # --- Backtest ---
        backtester = Backtester(
            strategy=None,
            risk_manager=risk_mgr,
            commission=cfg.backtest["commission"],
            slippage=cfg.backtest["slippage"],
            risk_free_rate=cfg.backtest["risk_free_rate"],
            trading_days_per_year=cfg.backtest["trading_days_per_year"],
        )

        results_list = []
        trade_logs = {}
        plots_available = []

        for t, df_t in per_ticker.items():
            close_col = f"Close_{t}"
            for name, strat in chosen.items():
                backtester.strategy = strat
                try:
                    result = backtester.run(df_t, ticker=t)
                except Exception as exc:
                    results_list.append({
                        "strategy": name,
                        "ticker": t,
                        "error": str(exc),
                    })
                    continue

                m = result.metrics

                # Check if plot exists
                plot_path = _PROJECT_DIR / "results" / "plots" / f"dashboard_{name}_{t}.html"
                has_plot = plot_path.exists()

                def _safe_round(v, d=3):
                    try:
                        return round(float(v), d)
                    except (ValueError, OverflowError, TypeError):
                        return 0.0

                entry = {
                    "strategy": name,
                    "ticker": t,
                    "total_return": _safe_round(m.total_return * 100, 2),
                    "cagr": _safe_round(m.cagr * 100, 2),
                    "sharpe": _safe_round(m.sharpe_ratio, 3),
                    "sortino": _safe_round(m.sortino_ratio, 3),
                    "max_drawdown": _safe_round(m.max_drawdown * 100, 2),
                    "win_rate": _safe_round(m.win_rate * 100, 2),
                    "profit_factor": _safe_round(m.profit_factor, 3),
                    "n_trades": int(m.n_trades),
                    "exposure": _safe_round(m.exposure * 100, 2),
                    "calmar": _safe_round(m.calmar_ratio, 3),
                    "volatility": _safe_round(m.volatility * 100, 2),
                    "avg_trade": _safe_round(m.avg_trade_return, 2),
                    "has_plot": has_plot,
                }
                results_list.append(entry)

                # Trade log
                if hasattr(result, "trades") and result.trades is not None and len(result.trades):
                    trades = result.trades
                    trade_logs[f"{name}_{t}"] = []
                    for _, row in trades.iterrows():
                        def _sf(v):
                            try:
                                r = float(v)
                                return 0.0 if (r != r or r == float('inf') or r == float('-inf')) else round(r, 2)
                            except (ValueError, TypeError, OverflowError):
                                return 0.0
                        trade_logs[f"{name}_{t}"].append({
                            "entry_date": str(row.get("entry_date", "")),
                            "exit_date": str(row.get("exit_date", "")),
                            "side": str(row.get("side", "")),
                            "entry_price": _sf(row.get("entry_price", 0)),
                            "exit_price": _sf(row.get("exit_price", 0)),
                            "pnl": _sf(row.get("pnl", 0)),
                            "return_pct": _sf(row.get("return_pct", 0)),
                        })

                # Equity curve
                equity = result.equity
                equity_data = []
                if equity is not None and len(equity):
                    for i, (dt, val) in enumerate(equity.items()):
                        try:
                            v = float(val)
                            v = 0.0 if (v != v or v == float('inf') or v == float('-inf')) else round(v, 2)
                            equity_data.append({"date": str(dt)[:10], "equity": v})
                        except (ValueError, TypeError, OverflowError):
                            pass
                    # For the plot inline
                    plots_available.append({
                        "strategy": name,
                        "ticker": t,
                        "plot_url": f"/plots/dashboard_{name}_{t}.html" if has_plot else None,
                        "equity": equity_data,
                    })

        response = {
            "success": True,
            "results": results_list,
            "trades": trade_logs,
            "plots": plots_available,
            "summary": _compute_summary(results_list),
        }
        return jsonify(response)

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# Raw data endpoint
# -------------------------------------------------------------------
@app.route("/api/data/<ticker>", methods=["GET"])
def get_raw_data(ticker):
    period = request.args.get("period", "1y")
    ticker = ticker.upper()
    try:
        loader = MarketDataLoader(cache_dir=str(_PROJECT_DIR / "data" / "cache"))
        raw = loader.download(tickers=[ticker], period=period)
        suffix = f"_{ticker}"
        cols = [c for c in raw.columns if c.endswith(suffix)]
        if not cols:
            return jsonify({"success": False, "error": f"No data for {ticker}"}), 404
        df = raw[cols].copy()
        ind_cfg = cfg.indicators
        df = add_all_indicators(
            df, ticker=ticker,
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
        df = df.reset_index()
        df["Date"] = df["Date"].astype(str)
        # Replace NaN/Inf with None for safe JSON
        import math as _math
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                    rec[k] = None
        return jsonify({"success": True, "data": records, "columns": list(df.columns)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# Download endpoint
# -------------------------------------------------------------------
@app.route("/api/download", methods=["POST"])
def download_csv():
    data = request.get_json(silent=True) or {}
    results = data.get("results", [])
    if not results:
        return jsonify({"error": "No results"}), 400

    df = pd.DataFrame(results)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="backtest_results.csv",
    )


# -------------------------------------------------------------------
# Summary helper
# -------------------------------------------------------------------
def _safe_val(v):
    try:
        r = float(v)
        return 0.0 if (r != r or r == float('inf') or r == float('-inf')) else r
    except (ValueError, TypeError, OverflowError):
        return 0.0


def _compute_summary(results):
    if not results:
        return {}
    clean = [r for r in results if "error" not in r]
    sharpe_vals = [_safe_val(r["sharpe"]) for r in clean]
    return_vals = [_safe_val(r["total_return"]) for r in clean]
    trades_vals = [r["n_trades"] for r in clean]

    best_sharpe = max(clean, key=lambda r: _safe_val(r["sharpe"]))
    best_return = max(clean, key=lambda r: _safe_val(r["total_return"]))
    most_trades = max(clean, key=lambda r: r["n_trades"])

    return {
        "n_runs": len(clean),
        "avg_sharpe": round(float(pd.Series(sharpe_vals).mean()), 3),
        "avg_return": round(float(pd.Series(return_vals).mean()), 2),
        "best_sharpe": {
            "strategy": best_sharpe["strategy"],
            "ticker": best_sharpe["ticker"],
            "value": best_sharpe["sharpe"],
        },
        "best_return": {
            "strategy": best_return["strategy"],
            "ticker": best_return["ticker"],
            "value": best_return["total_return"],
        },
        "most_trades": {
            "strategy": most_trades["strategy"],
            "ticker": most_trades["ticker"],
            "value": most_trades["n_trades"],
        },
    }


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=True)
