# 🤖 Quant Trading Bot

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-yellow.svg)](#-running-the-tests)

An end-to-end **algorithmic trading system** built in Python.  It
combines classical quantitative techniques (technical indicators,
mean-reversion / momentum / trend-following strategies, Kelly-criterion
position sizing, full backtesting with risk-adjusted performance
metrics) with a **deep-learning** module (LSTM-based price
prediction) — the same kind of stack you'd see at a prop trading firm
or quant hedge fund.

The bot is designed to be **readable**, **configurable**, and
**extensible**: every strategy, indicator and risk limit is defined in
`configs/config.yaml`, and new strategies plug in by subclassing
`Strategy` and registering them in `main.py`.

---

## ✨ Features

| Module                | What it does |
|-----------------------|--------------|
| `data`                | Live OHLCV downloads from Yahoo Finance with on-disk CSV caching |
| `indicators`          | RSI, MACD, Bollinger Bands, SMA/EMA, ATR, ADX (Wilder smoothing) |
| `strategies`          | Momentum, Mean Reversion, Trend Following, Ensemble voting |
| `risk`                | Kelly-criterion / fixed-fractional sizing, stop-loss, take-profit, trailing stop, drawdown kill switch |
| `backtester`          | Event-driven engine, commission + slippage, full trade log |
| `metrics`             | Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, CAGR, exposure |
| `ml`                  | TensorFlow/Keras LSTM price predictor, pluggable into the same backtester |
| `visualization`       | Matplotlib static plots + interactive Plotly HTML dashboard |
| `utils`               | YAML config loader, structured logging |
| `tests`               | Pytest unit tests for every module |

---

## 📦 Project layout

```
quant-trading-bot/
├── configs/
│   └── config.yaml            # All tuneable parameters
├── quant_trading_bot/
│   ├── data/loader.py         # yfinance wrapper + cache
│   ├── indicators/            # RSI, MACD, BB, ATR, ADX …
│   ├── strategies/            # Momentum / MeanReversion / Trend / Ensemble
│   ├── risk/manager.py        # Position sizing + stops
│   ├── backtester/            # Engine + performance metrics
│   ├── ml/                    # LSTM predictor + strategy adapter
│   ├── visualization/         # Matplotlib + Plotly dashboard
│   ├── utils/                 # Config + logging
│   ├── tests/                 # Pytest suite
│   ├── notebooks/             # Walkthrough examples
│   └── main.py                # End-to-end CLI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick start

### 1. Clone & install

```bash
git clone https://github.com/RishvanthAmsaraj/quant-trading-bot.git
cd quant-trading-bot

# Optional: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run a full backtest

```bash
python -m quant_trading_bot.main
```

This will:

* download 2 years of daily OHLCV for `AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM`
* compute the full panel of technical indicators
* run **Momentum**, **MeanReversion**, **TrendFollowing** and the
  **Ensemble** strategy against each ticker
* print a metrics summary (Sharpe, max drawdown, win rate …) for every
  backtest
* write matplotlib PNGs and an interactive Plotly HTML dashboard to
  `results/plots/`
* write a `results/summary.json` aggregating everything

### 3. Run with custom tickers

```bash
python -m quant_trading_bot.main --tickers AAPL MSFT NVDA
```

### 4. Train the LSTM (CPU-friendly, ~2-3 min/ticker)

```bash
python -m quant_trading_bot.main --lstm
```

### 5. Try the quick walkthrough

```bash
python quant_trading_bot/notebooks/example_walkthrough.py
```

---

## ⚙️ Configuration

All knobs are in `configs/config.yaml`:

```yaml
data:
  tickers: [AAPL, MSFT, GOOGL, ...]
  period: "2y"
  interval: "1d"

indicators:
  sma_fast: 20
  rsi_period: 14
  bb_period: 20
  bb_std: 2.0

strategies:
  momentum:        { lookback: 20, threshold: 0.02 }
  mean_reversion:  { rsi_oversold: 30, rsi_overbought: 70, bb_threshold: 0.05 }
  trend_following: { fast_ma: 20, slow_ma: 50, adx_threshold: 25 }

risk:
  initial_capital: 100000
  position_sizing: kelly      # kelly | fixed_fractional | equal_weight
  kelly_fraction: 0.25
  stop_loss_pct: 0.05
  take_profit_pct: 0.15
  trailing_stop_pct: 0.07
  max_drawdown_pct: 0.20       # portfolio kill switch

backtest:
  commission: 0.001
  slippage: 0.0005
  risk_free_rate: 0.02

ml:
  enabled: false
  model: lstm
  lookback: 60
  epochs: 30
  hidden_units: 64
```

---

## 🧠 Architecture overview

```
                    ┌────────────┐
                    │  config    │
                    └─────┬──────┘
                          ▼
                    ┌────────────┐
                    │  yfinance  │  (cached CSV)
                    └─────┬──────┘
                          ▼
                    ┌────────────┐
                    │ indicators │
                    └─────┬──────┘
                          ▼
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   MomentumStrategy  MeanReversion   TrendFollowing
         └────────────────┬────────────────┘
                          ▼
                    ┌────────────┐
                    │  risk      │  (sizing + stops)
                    └─────┬──────┘
                          ▼
                    ┌────────────┐
                    │ backtester │  (PnL, equity, trades)
                    └─────┬──────┘
                          ▼
                    ┌────────────┐
                    │  metrics   │  (Sharpe, drawdown …)
                    └─────┬──────┘
                          ▼
                    ┌────────────┐
                    │ dashboard  │  (matplotlib + plotly)
                    └────────────┘

  ┌────────────┐
  │  LSTM ml   │── plugs into the same strategy interface
  └────────────┘
```

---

## 🧪 Running the tests

```bash
pytest -v quant_trading_bot/tests
```

The tests use a synthetic price generator so they don't need internet
access.

---

## 🛠️ Extending the bot

* **Add a new indicator** → drop it into `indicators/technical.py` and
  expose it in `add_all_indicators`.
* **Add a new strategy** → subclass `Strategy` in
  `strategies/base.py` and register it in `main.py` →
  `_build_strategies`.
* **Add a new ML model** → keep the public interface
  (`fit(close, **kwargs)` + `predict(close) -> pd.Series`) and wrap
  it in a strategy that lives next to `lstm_strategy.py`.

---

## ⚠️ Disclaimer

This codebase is for **educational and research purposes only**.  No
part of it constitutes financial advice.  Past performance of any
strategy — backtested or otherwise — is **not** indicative of future
results.  Live trading carries the risk of substantial loss; do not
deploy this bot with real money without extensive out-of-sample
testing, slippage / liquidity analysis and proper regulatory review.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
