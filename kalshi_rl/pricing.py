"""Pricing utilities shared across the Kalshi RL pipeline."""

from __future__ import annotations

import math


def price_bucket(price: float, n_buckets: int = 5) -> int:
    price = max(0.0, min(1.0, price))
    bucket = int(price * n_buckets)
    if bucket == n_buckets:
        bucket = n_buckets - 1
    return bucket


def spread_bucket(spread: float, n_buckets: int = 5, max_spread: float = 0.10) -> int:
    spread = max(0.0, min(max_spread, spread))
    bucket = int((spread / max_spread) * n_buckets)
    if bucket == n_buckets:
        bucket = n_buckets - 1
    return bucket


def calculate_kalshi_fee(price_yes_side: float, contracts: int = 1, factor: float = 0.07) -> float:
    price_yes_side = max(0.0, min(1.0, price_yes_side))
    base = factor * contracts * price_yes_side * (1.0 - price_yes_side)
    fee_cents = math.ceil(base * 100.0)
    return fee_cents / 100.0
