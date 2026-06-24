"""
Evaluation Layer (Problem Statement 4)

Per-bar and per-trade metrics for the backtest output produced by
`backtest.run_backtest`. Formulas follow Lecture 8 / KB lines 4143-4283:

    annual_return   = mean(returns) * 252
    annual_vol      = std(returns)  * sqrt(252)
    sharpe          = annual_return / annual_vol      (r_f = 0)
    max_drawdown    = max(1 - equity / equity.cummax())
    hit_rate        = (strategy_return > 0) / (strategy_return != 0)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_return: float
    total_return: float
    buy_and_hold_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate_per_bar: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(self), index=["value"]).T


def _trade_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compress per-bar StrategyReturn into one return per round-trip trade.

    A "trade" runs from the bar a position opens to the bar before it closes;
    we sum the per-bar strategy returns within each holding period (log-sum
    approximation is fine here since per-bar returns are small).
    """
    pos = df["Position"].fillna(0).astype(int)
    holding_id = (pos.diff().fillna(pos.iloc[0]) != 0).cumsum()
    in_trade = pos != 0
    grouped = df.loc[in_trade, "StrategyReturn"].groupby(holding_id[in_trade]).sum()
    return grouped


def evaluate(df: pd.DataFrame) -> PerformanceMetrics:
    if "StrategyReturn" not in df.columns:
        raise ValueError("run_backtest first — StrategyReturn missing")

    sr = df["StrategyReturn"].fillna(0)
    br = df["Return"].fillna(0)

    per_trade = _trade_returns(df)
    wins = int((per_trade > 0).sum())
    losses = int((per_trade < 0).sum())
    total = int(len(per_trade))

    ann_ret = sr.mean() * TRADING_DAYS
    ann_vol = sr.std() * np.sqrt(TRADING_DAYS)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")

    equity = df["Equity"].dropna()
    if len(equity) == 0:
        total_return = float("nan")
        max_dd = float("nan")
    else:
        total_return = float(equity.iloc[-1] - 1.0)
        max_dd = float((1 - equity / equity.cummax()).max())

    bh_total = float(df["BuyHoldEquity"].dropna().iloc[-1] - 1.0) \
        if "BuyHoldEquity" in df.columns and df["BuyHoldEquity"].notna().any() \
        else float("nan")

    nonzero = sr[sr != 0]
    hit_rate_bar = float((nonzero > 0).mean()) if len(nonzero) else float("nan")

    return PerformanceMetrics(
        total_trades=total,
        winning_trades=wins,
        losing_trades=losses,
        win_rate=float(wins / total) if total else float("nan"),
        avg_trade_return=float(per_trade.mean()) if total else float("nan"),
        total_return=total_return,
        buy_and_hold_return=bh_total,
        annualized_return=float(ann_ret),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        hit_rate_per_bar=hit_rate_bar,
    )


def compare_strategies(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a side-by-side metrics table from a dict of strategy → bt frame."""
    rows = {name: asdict(evaluate(df)) for name, df in results.items()}
    return pd.DataFrame(rows)
