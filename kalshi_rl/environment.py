"""Offline two-step environment built from historical Kalshi episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .pricing import calculate_kalshi_fee, price_bucket, spread_bucket


@dataclass
class Episode:
    contract_id: str
    price0: float
    price1: float
    bid0: float
    ask0: float
    bid1: float
    ask1: float
    resolved: int


class KalshiTwoStepEnv:
    """Simple two-step environment with spread + fee-aware execution."""

    def __init__(
        self,
        episodes: Sequence[Dict],
        *,
        n_price_buckets: int = 5,
        n_spread_buckets: int = 5,
    ) -> None:
        self.episodes: Sequence[Dict] = episodes
        self.n_price_buckets = n_price_buckets
        self.n_spread_buckets = n_spread_buckets
        self.ep: Dict | None = None
        self.step_idx = 0
        self.holding = False
        self.side = 0
        self.entry_cost: float | None = None

    def _encode_state(self) -> Tuple[int, int, int, int, int]:
        assert self.ep is not None, "Environment must be reset before stepping."

        if self.step_idx == 0:
            mid = self.ep["price0"]
            bid = self.ep["bid0"]
            ask = self.ep["ask0"]
        else:
            mid = self.ep["price1"]
            bid = self.ep["bid1"]
            ask = self.ep["ask1"]

        pb = price_bucket(mid, self.n_price_buckets)
        spread = ask - bid
        sb = spread_bucket(spread, self.n_spread_buckets)
        return (self.step_idx, pb, sb, int(self.holding), int(self.side))

    def reset(self) -> Tuple[int, int, int, int, int]:
        self.ep = random.choice(self.episodes)
        self.step_idx = 0
        self.holding = False
        self.side = 0
        self.entry_cost = None
        return self._encode_state()

    def valid_actions(self) -> List[int]:
        if self.step_idx == 0:
            return [0, 1, 2]
        if self.holding:
            return [0, 1]
        return [0]

    def step(self, action: int):
        assert self.ep is not None, "Call reset() before step()."

        done = False
        reward = 0.0

        if self.step_idx == 0:
            if action == 1:
                self.holding = True
                self.side = 1
                ask0 = self.ep["ask0"]
                fee_open = calculate_kalshi_fee(ask0)
                self.entry_cost = ask0 + fee_open
            elif action == 2:
                self.holding = True
                self.side = -1
                bid0 = self.ep["bid0"]
                no_price_open = 1.0 - bid0
                fee_open = calculate_kalshi_fee(no_price_open)
                self.entry_cost = no_price_open + fee_open

            self.step_idx = 1
            next_state = self._encode_state()
        else:
            resolved = self.ep["resolved"]
            if self.holding and self.entry_cost is not None:
                if action == 0:
                    payoff = float(resolved if self.side == 1 else 1 - resolved)
                    reward = payoff - self.entry_cost
                elif action == 1:
                    if self.side == 1:
                        bid1 = self.ep["bid1"]
                        fee_close = calculate_kalshi_fee(bid1)
                        exit_value_net = bid1 - fee_close
                    else:
                        ask1 = self.ep["ask1"]
                        no_price_close = 1.0 - ask1
                        fee_close = calculate_kalshi_fee(no_price_close)
                        exit_value_net = no_price_close - fee_close
                    reward = exit_value_net - self.entry_cost

            done = True
            next_state = self._encode_state()

        return next_state, reward, done
