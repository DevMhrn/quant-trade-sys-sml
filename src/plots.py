"""
Reusable plotting helpers — keep matplotlib calls out of the notebook narrative
so the chart code stays consistent across sections.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, save_path: str | None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")


def plot_price(df: pd.DataFrame, title: str = "Close Price",
               save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["Close"], label="Close", color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, save_path)
    return fig


def plot_volume(df: pd.DataFrame, title: str = "Volume",
                save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(df.index, df["Volume"], width=1.0, color="#888")
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Volume")
    ax.grid(alpha=0.3)
    _save(fig, save_path)
    return fig


def plot_price_with_mas(df: pd.DataFrame, ma_cols: Iterable[str],
                        title: str = "Close + Moving Averages",
                        save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["Close"], label="Close", color="#1f77b4", alpha=0.8)
    for col in ma_cols:
        ax.plot(df.index, df[col], label=col)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, save_path)
    return fig


def plot_signals(df: pd.DataFrame, title: str = "Trading Signals",
                 save_path: str | None = None):
    """Overlay BUY/SELL markers on the close-price chart."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["Close"], label="Close", color="#1f77b4", alpha=0.8)

    pos = df["Position"].fillna(0)
    flips = pos.diff().fillna(pos.iloc[0])
    buys = df.index[flips > 0]
    sells = df.index[flips < 0]

    ax.scatter(buys, df.loc[buys, "Close"], marker="^", color="green",
               s=80, label="BUY", zorder=5)
    ax.scatter(sells, df.loc[sells, "Close"], marker="v", color="red",
               s=80, label="SELL", zorder=5)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, save_path)
    return fig


def plot_volatility(df: pd.DataFrame, vol_col: str = "Vol20",
                    title: str = "Rolling Volatility",
                    save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(df.index, df[vol_col], color="#d62728")
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel(vol_col)
    ax.grid(alpha=0.3)
    _save(fig, save_path)
    return fig


def plot_equity_curves(results: dict[str, pd.DataFrame],
                       include_buy_hold: bool = True,
                       title: str = "Equity Curve — Strategy vs Buy & Hold",
                       save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, df in results.items():
        ax.plot(df.index, df["Equity"], label=name)
    if include_buy_hold and results:
        first = next(iter(results.values()))
        ax.plot(first.index, first["BuyHoldEquity"], label="Buy & Hold",
                linestyle="--", color="black", alpha=0.7)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Equity (start = 1)")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, save_path)
    return fig


def plot_drawdown(df: pd.DataFrame, title: str = "Drawdown",
                  save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(df.index, df["Drawdown"], 0, color="#d62728", alpha=0.4)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    _save(fig, save_path)
    return fig


def plot_returns_distribution(df: pd.DataFrame, col: str = "StrategyReturn",
                              title: str = "Return Distribution",
                              save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[col].dropna(), bins=50, color="#1f77b4", alpha=0.75)
    ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    _save(fig, save_path)
    return fig
