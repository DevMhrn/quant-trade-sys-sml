"""
Signal Engine (Problem Statement 2 — signal logic)

Two strategies — Moving-Average Crossover and Mean Reversion. Signals are
encoded as 0/1 (long / flat) per the lecture convention; the `position` column
is `signal.shift(1)` to avoid look-ahead bias on the backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma_crossover_signal(df: pd.DataFrame, short_window: int = 20,
                        long_window: int = 50,
                        price_col: str = "Close") -> pd.DataFrame:
    """
    Strategy A — go long when fast MA crosses above slow MA, exit when it
    crosses back below. 0/1 signal, long-only.
    """
    df = df.copy()
    short_ma = df[price_col].rolling(short_window).mean()
    long_ma = df[price_col].rolling(long_window).mean()

    df[f"MA{short_window}"] = short_ma
    df[f"MA{long_window}"] = long_ma

    df["Signal"] = np.where(short_ma > long_ma, 1, 0)
    df["Signal"] = df["Signal"].where(short_ma.notna() & long_ma.notna())
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Trade"] = (df["Position"].diff().fillna(0) != 0).astype(int)
    return df


def mean_reversion_signal(df: pd.DataFrame, window: int = 20,
                          band: float = 0.05,
                          price_col: str = "Close") -> pd.DataFrame:
    """
    Strategy B — buy when price falls more than `band` (default 5%) below its
    rolling mean, exit when it returns to within the band. 0/1 signal.
    """
    df = df.copy()
    ma = df[price_col].rolling(window).mean()
    df[f"MA{window}"] = ma

    z = (df[price_col] - ma) / ma  # relative deviation from mean
    long_entry = z < -band
    long_exit = z > 0

    state = np.zeros(len(df), dtype=int)
    in_pos = False
    for i, (entry, exit_) in enumerate(zip(long_entry.values, long_exit.values)):
        if not in_pos and entry:
            in_pos = True
        elif in_pos and exit_:
            in_pos = False
        state[i] = 1 if in_pos else 0

    df["Signal"] = pd.Series(state, index=df.index).where(ma.notna())
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Trade"] = (df["Position"].diff().fillna(0) != 0).astype(int)
    return df


def buy_and_hold_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Benchmark — long from the first bar onwards."""
    df = df.copy()
    df["Signal"] = 1
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Trade"] = (df["Position"].diff().fillna(0) != 0).astype(int)
    return df


STRATEGIES = {
    "ma_crossover": ma_crossover_signal,
    "mean_reversion": mean_reversion_signal,
    "buy_and_hold": buy_and_hold_signal,
}
