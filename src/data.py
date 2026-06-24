"""
Market Data Layer (Problem Statement 1)

Loads OHLCV data either from yfinance (preferred for real assets) or generates
a reproducible synthetic dataset following the convention from Lecture 2.
Cleans missing/invalid rows and exposes summary statistics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DataSummary:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_total: int
    mean_close: float
    std_close: float
    min_close: float
    max_close: float
    mean_volume: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.__dict__, index=["value"]).T


def load_yfinance(ticker: str, start: str, end: str | None = None,
                  interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and normalize columns."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")

    # yfinance may return a MultiIndex on columns when given a list of tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def generate_synthetic(n_days: int = 500, start: str = "2022-01-03",
                       seed: int = 42, start_price: float = 100.0,
                       drift: float = 0.0005, vol: float = 0.015) -> pd.DataFrame:
    """
    Generate a reproducible synthetic OHLCV series.

    Builds a daily geometric random walk for close, then derives open/high/low
    around it and assigns a random volume — mirrors the Lecture 2 / Lecture 5
    notebook construction with a DatetimeIndex on top.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=vol, size=n_days)
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.concatenate([[start_price], close[:-1]]) \
        * (1 + rng.normal(0, vol / 3, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n_days)))
    volume = rng.integers(low=500_000, high=5_000_000, size=n_days)

    idx = pd.bdate_range(start=start, periods=n_days, name="Date")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    return df


def load_ohlcv(ticker: str | None = "AAPL",
               start: str = "2022-01-01",
               end: str | None = None,
               cache_path: str | None = None,
               fall_back_to_synthetic: bool = True) -> pd.DataFrame:
    """
    Convenience loader. Tries cache → yfinance → synthetic in that order so the
    notebook is runnable offline.
    """
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df[OHLCV_COLS]

    if ticker:
        try:
            df = load_yfinance(ticker, start=start, end=end)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path)
            return df
        except Exception as exc:
            if not fall_back_to_synthetic:
                raise
            print(f"[data] yfinance failed ({exc!s}); using synthetic dataset")

    df = generate_synthetic()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
    return df


def clean_ohlcv(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Drop rows that are entirely empty, forward-fill remaining gaps, and drop any
    rows where a non-positive price slipped through.
    """
    df = df.copy()
    df = df.dropna(how="all")

    if method == "ffill":
        df[OHLCV_COLS] = df[OHLCV_COLS].ffill()
    elif method == "drop":
        df = df.dropna(subset=OHLCV_COLS)
    else:
        raise ValueError(f"Unknown clean method: {method}")

    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df[df["Volume"] >= 0]
    return df


def summarize(df: pd.DataFrame) -> DataSummary:
    return DataSummary(
        rows=len(df),
        start=df.index.min(),
        end=df.index.max(),
        missing_total=int(df[OHLCV_COLS].isna().sum().sum()),
        mean_close=float(df["Close"].mean()),
        std_close=float(df["Close"].std()),
        min_close=float(df["Close"].min()),
        max_close=float(df["Close"].max()),
        mean_volume=float(df["Volume"].mean()),
    )
