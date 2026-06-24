# Simplified Quant Trading System

**Quantitative Modelling Program — Final Project**

An end-to-end (but intentionally simple) quantitative trading research
framework in Python. Implements the five layers covered in the course —
market data, analytics, signal engine, execution simulation, evaluation —
and runs the pipeline against real equity data.

## Project Structure

```
.
├── notebook/
│   └── project_notebook.ipynb   # orchestrating notebook (run me)
├── src/
│   ├── data.py                  # OHLCV load + clean (PS1)
│   ├── indicators.py            # returns, MA, vol, Bollinger (PS2)
│   ├── signals.py               # MA crossover + mean reversion (PS2)
│   ├── execution.py             # bid/ask + market/limit orders (PS3)
│   ├── backtest.py              # vectorised backtest engine
│   ├── evaluation.py            # Sharpe, MDD, win rate, etc. (PS4)
│   └── plots.py                 # matplotlib helpers
├── data/
│   ├── SPY.csv                  # raw OHLCV (2018-2024)
│   └── SPY_clean.csv            # cleaned OHLCV
├── plots/                       # generated charts (10)
├── report/
│   └── final_report.md          # write-up — covers all required sections
├── verify.py                    # 49-check invariant test suite
└── README.md
```

## Verifying the bookkeeping

To run the full invariant suite (data integrity, no look-ahead bias,
signal logic, execution rules, backtest formulas, Sharpe by hand):

```bash
.venv/bin/python verify.py
```

Should print `=== SUMMARY: 49 PASS / 0 FAIL ===`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib yfinance jupyter seaborn
```

## Run

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebook/project_notebook.ipynb
```

Or open `notebook/project_notebook.ipynb` in JupyterLab and run all cells.

The notebook auto-downloads SPY from Yahoo Finance on first run and caches
it under `data/SPY.csv`. If the network is unavailable, `load_ohlcv` falls
back to a seeded synthetic dataset.

## Problem-Statement Map

| PS | Where to look |
|---|---|
| 1 — Market data processing | `src/data.py` + notebook §1 |
| 2 — Indicators & signals | `src/indicators.py`, `src/signals.py` + notebook §2 |
| 3 — Execution simulation | `src/execution.py` + notebook §3 |
| 4 — Strategy evaluation | `src/backtest.py`, `src/evaluation.py` + notebook §4 |

## Notes on Backtest Correctness

- All strategy returns use `Position.shift(1) × Return` to avoid look-ahead
  bias (Lecture 5 convention).
- Transaction cost (10 bps) and slippage (5 bps) are deducted on every
  signal flip, not every bar.
- Equity curve uses `(1 + r).cumprod()` and starts at 1.0.
- Annualisation factor is 252 trading days.
- Sharpe ratio assumes `Rf = 0`.

See `report/final_report.md` for the full write-up, results table, and
observations.
