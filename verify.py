"""
verify.py - Invariant test suite for the simplified quant trading system.

Runs 49 sanity checks across all five layers (data, indicators, signals,
execution, backtest, evaluation) and prints a PASS/FAIL line for each. Used
to make sure the bookkeeping is right - in particular that there is no
look-ahead bias and that the execution rules from PS3 hold exactly.

Run from the project root:
    .venv/bin/python verify.py
"""

from __future__ import annotations

import os
import sys

# Make src/ importable regardless of where the script is launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.data import load_ohlcv, clean_ohlcv
from src.indicators import build_indicators
from src.signals import (
    ma_crossover_signal,
    mean_reversion_signal,
    buy_and_hold_signal,
)
from src.execution import (
    add_bid_ask,
    signal_to_orders,
    simulate_orders,
    Order,
    execute_order,
)
from src.backtest import BacktestConfig, run_backtest
from src.evaluation import evaluate


PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    extra = "" if cond else f"  {detail}"
    print(f"  [{mark}] {name}{extra}")


# ---------- load real data (cache hit if already downloaded) ----------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "SPY.csv")
df_raw = load_ohlcv(
    ticker="SPY",
    start="2018-01-01",
    end="2024-12-31",
    cache_path=DATA_PATH,
)
df = clean_ohlcv(df_raw)

# ---------- A. data layer ----------
print("\n=== A. data layer ===")
check("OHLCV columns present",
      set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns))
check("DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
check("no NaNs after clean", df.isna().sum().sum() == 0)
check("all prices > 0", (df[["Open", "High", "Low", "Close"]] > 0).all().all())
check("high >= low on every bar", (df["High"] >= df["Low"]).all())
check("high >= max(open,close)",
      (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-6).all())
check("low  <= min(open,close)",
      (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-6).all())
check("volume non-negative", (df["Volume"] >= 0).all())

# ---------- B. indicators ----------
print("\n=== B. indicators ===")
df_ind = build_indicators(df, 20, 50, 20)
check("Return[0] is NaN (no prior)", pd.isna(df_ind["Return"].iloc[0]))
check("CumReturn matches manual product",
      np.isclose(df_ind["CumReturn"].iloc[-1],
                 (1 + df_ind["Return"].dropna()).prod() - 1))
check("MA20 NaN until row 19, valid at 20",
      pd.isna(df_ind["MA20"].iloc[18]) and not pd.isna(df_ind["MA20"].iloc[19]))
check("MA20 == manual rolling mean",
      np.isclose(df_ind["MA20"].iloc[19], df_ind["Close"].iloc[:20].mean()))
check("Vol20 == manual rolling std",
      np.isclose(df_ind["Vol20"].iloc[20],
                 df_ind["Return"].iloc[1:21].std()))

# ---------- C. signals: no look-ahead bias ----------
print("\n=== C. signals (no look-ahead) ===")
ma = ma_crossover_signal(df_ind, 20, 50)
mr = mean_reversion_signal(df_ind, 20, 0.03)
bh = buy_and_hold_signal(df_ind)

for name, s in [("MA crossover", ma), ("Mean reversion", mr), ("Buy & Hold", bh)]:
    pos = s["Position"]
    sig = s["Signal"]
    aligned = (pos.iloc[1:].fillna(0).values
               == sig.shift(1).iloc[1:].fillna(0).values)
    check(f"{name}: Position = Signal.shift(1)", aligned.all())
    check(f"{name}: Position[0] == 0 (no preposition)", s["Position"].iloc[0] == 0)
    sig_vals = set(sig.dropna().unique())
    check(f"{name}: Signal in {{0,1}}", sig_vals.issubset({0, 1, 0.0, 1.0}))

# ---------- D. signal logic correctness ----------
print("\n=== D. signal logic ===")
mask = ma.dropna(subset=["MA20", "MA50"]).copy()
ok = ((mask["MA20"] > mask["MA50"]).astype(int)
      == mask["Signal"].astype(int)).all()
check("MA crossover: Signal = (MA20 > MA50)", ok)

mask2 = mr.dropna(subset=["MA20"]).copy()
exited_after_above = True
in_pos = False
for c, m, s in zip(mask2["Close"], mask2["MA20"], mask2["Signal"]):
    if in_pos and c > m and s != 0:
        exited_after_above = False
        break
    in_pos = (s == 1)
check("Mean reversion: exits when Close > MA20", exited_after_above)

# ---------- E. execution ----------
print("\n=== E. execution ===")
book = add_bid_ask(ma, spread=0.10)
check("Ask > Bid everywhere", (book["Ask"] > book["Bid"]).all())
check("Spread == 0.10", np.allclose(book["Spread"], 0.10))
check("Bid = Close - spread/2", np.allclose(book["Bid"], book["Close"] - 0.05))
check("Ask = Close + spread/2", np.allclose(book["Ask"], book["Close"] + 0.05))

bar = book.iloc[100]
mb = execute_order(Order("buy", 1, "market"), bar)
check("Market BUY fills at Ask", np.isclose(mb.fill_price, bar["Ask"]))
ms = execute_order(Order("sell", 1, "market"), bar)
check("Market SELL fills at Bid", np.isclose(ms.fill_price, bar["Bid"]))

lb_unfilled = execute_order(
    Order("buy", 1, "limit", limit_price=bar["Ask"] - 0.01), bar)
lb_filled = execute_order(
    Order("buy", 1, "limit", limit_price=bar["Ask"] + 0.01), bar)
check("Limit BUY unfilled when Ask > limit", lb_unfilled is None)
check("Limit BUY filled when Ask <= limit",
      lb_filled is not None and np.isclose(lb_filled.fill_price, bar["Ask"]))

ls_unfilled = execute_order(
    Order("sell", 1, "limit", limit_price=bar["Bid"] + 0.01), bar)
ls_filled = execute_order(
    Order("sell", 1, "limit", limit_price=bar["Bid"] - 0.01), bar)
check("Limit SELL unfilled when Bid < limit", ls_unfilled is None)
check("Limit SELL filled when Bid >= limit",
      ls_filled is not None and np.isclose(ls_filled.fill_price, bar["Bid"]))

orders = signal_to_orders(book)
n_flips = int(ma["Trade"].sum())
check(f"signal_to_orders produces {n_flips} orders == Trade flips",
      len(orders) == n_flips)

fills = simulate_orders(book, orders)
check("All market orders filled", len(fills) == len(orders))

# ---------- F. backtest ----------
print("\n=== F. backtest ===")
cfg = BacktestConfig(0.001, 0.0005)
bt = run_backtest(book, cfg)
check("StrategyReturn = GrossReturn - cost on flips",
      np.allclose(bt["StrategyReturn"],
                  bt["GrossReturn"] - bt["Trade"] * (0.001 + 0.0005)))

manual_gross = bt["Position"].fillna(0) * bt["Return"].fillna(0)
check("GrossReturn uses Position (not Signal)",
      np.allclose(bt["GrossReturn"].dropna(), manual_gross.dropna()))

expected_eq = (1 + bt["StrategyReturn"]).cumprod()
check("Equity = (1+r).cumprod()",
      np.allclose(bt["Equity"].dropna(), expected_eq.dropna()))
check("Equity[0] > 0 (starts near 1.0)", 0.5 < bt["Equity"].iloc[0] < 2.0)
check("Drawdown <= 0 everywhere", (bt["Drawdown"].dropna() <= 1e-9).all())

expected_bh = (1 + bt["Return"].fillna(0)).cumprod()
check("BuyHoldEquity = (1+Return).cumprod()",
      np.allclose(bt["BuyHoldEquity"].dropna(), expected_bh.dropna()))

# ---------- G. evaluation ----------
print("\n=== G. evaluation ===")
m = evaluate(bt)
check("total_trades == winning + losing",
      m.total_trades == m.winning_trades + m.losing_trades)
check("annualized_return finite", np.isfinite(m.annualized_return))
check("sharpe_ratio finite", np.isfinite(m.sharpe_ratio))
check("max_drawdown in [0,1]", 0 <= m.max_drawdown <= 1)
check("total_return == Equity[-1] - 1",
      np.isclose(m.total_return, bt["Equity"].dropna().iloc[-1] - 1))
check("buy_and_hold_return == BuyHoldEquity[-1] - 1",
      np.isclose(m.buy_and_hold_return,
                 bt["BuyHoldEquity"].dropna().iloc[-1] - 1))

sr = bt["StrategyReturn"].fillna(0)
hand_sharpe = (sr.mean() * 252) / (sr.std() * np.sqrt(252))
check("Sharpe matches mean*252 / (std*sqrt(252))",
      np.isclose(m.sharpe_ratio, hand_sharpe))

# ---------- summary ----------
print(f"\n=== SUMMARY: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
if FAIL:
    print("FAILED CHECKS:")
    for f in FAIL:
        print(" -", f)
    sys.exit(1)
