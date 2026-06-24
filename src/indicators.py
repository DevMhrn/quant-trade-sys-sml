"""
Analytics Layer (Problem Statement 2 — indicators only)

Computes returns and rolling indicators used by the signal engine and reports.
Vectorized — no per-bar loops, matching the Lecture 3/4 convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Simple percent returns and cumulative (compounded) returns."""
    df = df.copy()
    df["Return"] = df[price_col].pct_change()
    df["CumReturn"] = (1 + df["Return"]).cumprod() - 1
    return df


def add_moving_average(df: pd.DataFrame, window: int,
                       price_col: str = "Close", name: str | None = None) -> pd.DataFrame:
    df = df.copy()
    col = name or f"MA{window}"
    df[col] = df[price_col].rolling(window).mean()
    return df


def add_rolling_volatility(df: pd.DataFrame, window: int = 20,
                           return_col: str = "Return",
                           name: str | None = None,
                           annualize: bool = False) -> pd.DataFrame:
    """Rolling std of returns. Annualized form multiplies by sqrt(252)."""
    df = df.copy()
    col = name or f"Vol{window}"
    vol = df[return_col].rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    df[col] = vol
    return df


def add_volume_average(df: pd.DataFrame, window: int = 20,
                       name: str | None = None) -> pd.DataFrame:
    df = df.copy()
    col = name or f"AvgVolume{window}"
    df[col] = df["Volume"].rolling(window).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0,
                        price_col: str = "Close") -> pd.DataFrame:
    """Bollinger bands — used by the mean-reversion strategy."""
    df = df.copy()
    mid = df[price_col].rolling(window).mean()
    std = df[price_col].rolling(window).std()
    df[f"BB_Mid{window}"] = mid
    df[f"BB_Upper{window}"] = mid + n_std * std
    df[f"BB_Lower{window}"] = mid - n_std * std
    return df


def build_indicators(df: pd.DataFrame, short_window: int = 20,
                     long_window: int = 50, vol_window: int = 20) -> pd.DataFrame:
    """One-shot helper used by the notebook to attach the standard indicator set."""
    out = add_returns(df)
    out = add_moving_average(out, short_window, name=f"MA{short_window}")
    out = add_moving_average(out, long_window, name=f"MA{long_window}")
    out = add_rolling_volatility(out, vol_window)
    out = add_volume_average(out, vol_window)
    out = add_bollinger_bands(out, short_window)
    return out
