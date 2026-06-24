"""
Execution Layer (Problem Statement 3)

Models a simple market microstructure with a fixed half-spread around the close
and simulates market + limit orders against it.

Spread model (per project spec 5.2):
    Bid = Close − spread / 2
    Ask = Close + spread / 2

Market BUY  fills at Ask.
Market SELL fills at Bid.
Limit  BUY  fills if Ask <= LimitPrice.
Limit  SELL fills if Bid >= LimitPrice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


OrderType = Literal["market", "limit"]
Side = Literal["buy", "sell"]


@dataclass
class Order:
    side: Side
    qty: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    timestamp: pd.Timestamp | None = None


@dataclass
class Fill:
    order: Order
    fill_price: float
    fill_qty: float
    timestamp: pd.Timestamp
    bid: float
    ask: float


def add_bid_ask(df: pd.DataFrame, spread: float = 0.05) -> pd.DataFrame:
    """
    Attach Bid/Ask/Spread columns. `spread` is in price units (e.g. 0.05 = 5
    cents) — use a fraction of Close for proportional spreads.
    """
    df = df.copy()
    half = spread / 2.0
    df["Bid"] = df["Close"] - half
    df["Ask"] = df["Close"] + half
    df["Spread"] = spread
    return df


def add_proportional_bid_ask(df: pd.DataFrame, bps: float = 5.0) -> pd.DataFrame:
    """Bid/Ask with proportional spread expressed in basis points (bps/10000)."""
    df = df.copy()
    half = (bps / 10_000.0) / 2.0
    df["Bid"] = df["Close"] * (1 - half)
    df["Ask"] = df["Close"] * (1 + half)
    df["Spread"] = df["Ask"] - df["Bid"]
    return df


def execute_order(order: Order, bar: pd.Series) -> Fill | None:
    """Try to fill `order` against a single OHLCV+Bid/Ask bar."""
    bid = float(bar["Bid"])
    ask = float(bar["Ask"])

    if order.order_type == "market":
        price = ask if order.side == "buy" else bid
        return Fill(order, price, order.qty, bar.name, bid, ask)

    if order.order_type == "limit":
        if order.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if order.side == "buy" and ask <= order.limit_price:
            return Fill(order, ask, order.qty, bar.name, bid, ask)
        if order.side == "sell" and bid >= order.limit_price:
            return Fill(order, bid, order.qty, bar.name, bid, ask)
        return None

    raise ValueError(f"unknown order_type {order.order_type}")


def simulate_orders(df: pd.DataFrame, orders: list[Order]) -> pd.DataFrame:
    """
    Walk forward through bars, fill each order on the first bar that accepts
    it (limit orders may wait), and return a Fills DataFrame.
    """
    if "Bid" not in df.columns or "Ask" not in df.columns:
        raise ValueError("call add_bid_ask() first")

    pending = list(orders)
    fills: list[Fill] = []

    for ts, bar in df.iterrows():
        remaining: list[Order] = []
        for order in pending:
            if order.timestamp is not None and ts < order.timestamp:
                remaining.append(order)
                continue
            fill = execute_order(order, bar)
            if fill is None:
                remaining.append(order)
            else:
                fills.append(fill)
        pending = remaining

    if not fills:
        return pd.DataFrame(columns=["timestamp", "side", "order_type",
                                     "limit_price", "fill_price", "fill_qty",
                                     "bid", "ask"])

    rows = [{
        "timestamp": f.timestamp,
        "side": f.order.side,
        "order_type": f.order.order_type,
        "limit_price": f.order.limit_price,
        "fill_price": f.fill_price,
        "fill_qty": f.fill_qty,
        "bid": f.bid,
        "ask": f.ask,
    } for f in fills]
    return pd.DataFrame(rows).set_index("timestamp")


def signal_to_orders(df: pd.DataFrame, qty: float = 1.0) -> list[Order]:
    """
    Translate a 0/1 Position series into market buy/sell orders at each flip.
    Used by the notebook to feed `simulate_orders` from the signal engine.
    """
    if "Position" not in df.columns:
        raise ValueError("expected Position column from signals module")

    pos = df["Position"].fillna(0).astype(int)
    flips = pos.diff().fillna(pos.iloc[0])
    orders: list[Order] = []
    for ts, change in flips.items():
        if change > 0:
            orders.append(Order("buy", qty, "market", timestamp=ts))
        elif change < 0:
            orders.append(Order("sell", qty, "market", timestamp=ts))
    return orders
