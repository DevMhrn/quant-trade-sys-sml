"""
Backtest Engine — joins signals + execution into per-bar returns and an equity
curve. Vectorized using the `signal.shift(1) * return` convention from
Lecture 5 to avoid look-ahead bias.

Transaction costs and slippage are deducted on every `Trade` flip, mirroring
the staged backtest from Lecture 5 Part 2 (Baseline → +Costs → +Slippage →
+Liquidity filter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    cost_per_trade: float = 0.001       # 10 bps round-trip
    slippage_per_trade: float = 0.0005  # 5 bps
    apply_liquidity_filter: bool = False
    liquidity_window: int = 20


def run_backtest(df: pd.DataFrame, config: BacktestConfig | None = None) -> pd.DataFrame:
    """
    Expects columns: Close, Return (or computes it), Position, Trade.
    Returns the same DataFrame extended with:
        StrategyReturn   — net of costs + slippage + optional liquidity filter
        GrossReturn      — strategy return before costs (for comparison)
        Equity           — equity curve (starts at 1.0)
        BuyHoldEquity    — equity for buy-and-hold benchmark
        Drawdown         — drawdown of the equity curve
    """
    config = config or BacktestConfig()
    df = df.copy()

    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change()

    if "Position" not in df.columns or "Trade" not in df.columns:
        raise ValueError("Position and Trade columns required — run a signal first")

    gross = df["Position"].fillna(0) * df["Return"].fillna(0)

    if config.apply_liquidity_filter:
        avg_vol = df["Volume"].rolling(config.liquidity_window).mean()
        liquid = (df["Volume"] > avg_vol).astype(int).fillna(0)
        gross = gross * liquid

    trade_cost = df["Trade"].fillna(0) * (config.cost_per_trade
                                          + config.slippage_per_trade)

    df["GrossReturn"] = gross
    df["StrategyReturn"] = gross - trade_cost
    df["Equity"] = (1 + df["StrategyReturn"]).cumprod()
    df["BuyHoldEquity"] = (1 + df["Return"].fillna(0)).cumprod()

    running_max = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] / running_max - 1
    return df
