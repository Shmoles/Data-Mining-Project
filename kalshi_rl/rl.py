"""Tabular Q-learning trainer and evaluation helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .environment import KalshiTwoStepEnv
from .pricing import calculate_kalshi_fee, price_bucket, spread_bucket


def q_learning(
    env: KalshiTwoStepEnv,
    *,
    n_episodes: int = 50_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon: float = 0.1,
) -> Dict:
    Q: Dict = defaultdict(lambda: defaultdict(float))

    for _ in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            actions = env.valid_actions()
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_state = Q[state]
                action = max(actions, key=lambda a: q_state[a])

            next_state, reward, done = env.step(action)

            if done:
                target = reward
            else:
                next_actions = env.valid_actions()
                max_next_q = max(Q[next_state][a] for a in next_actions)
                target = reward + gamma * max_next_q

            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state

    return Q


def run_greedy_policy(
    episodes: Sequence[Dict],
    Q: Dict,
    *,
    n_price_buckets: int = 5,
    n_spread_buckets: int = 5,
) -> np.ndarray:
    pnl: List[float] = []

    for ep in episodes:
        resolved = ep["resolved"]
        pb0 = price_bucket(ep["price0"], n_price_buckets)
        spread0 = ep["ask0"] - ep["bid0"]
        sb0 = spread_bucket(spread0, n_spread_buckets)
        s0 = (0, pb0, sb0, 0, 0)

        actions0 = [0, 1, 2]
        q_values_s0 = {a: Q[s0][a] for a in actions0}
        if not q_values_s0 or all(v == 0.0 for v in q_values_s0.values()):
            a0 = random.choice(actions0)
        else:
            a0 = max(actions0, key=lambda a: q_values_s0[a])

        if a0 == 0:
            pnl.append(0.0)
            continue

        if a0 == 1:
            side = 1
            ask0 = ep["ask0"]
            fee_open = calculate_kalshi_fee(ask0)
            entry_cost = ask0 + fee_open
        else:
            side = -1
            bid0 = ep["bid0"]
            no_price_open = 1.0 - bid0
            fee_open = calculate_kalshi_fee(no_price_open)
            entry_cost = no_price_open + fee_open

        pb1 = price_bucket(ep["price1"], n_price_buckets)
        spread1 = ep["ask1"] - ep["bid1"]
        sb1 = spread_bucket(spread1, n_spread_buckets)
        s1 = (1, pb1, sb1, 1, side)

        actions1 = [0, 1]
        q_values_s1 = {a: Q[s1][a] for a in actions1}
        if not q_values_s1 or all(v == 0.0 for v in q_values_s1.values()):
            a1 = random.choice(actions1)
        else:
            a1 = max(actions1, key=lambda a: q_values_s1[a])

        if a1 == 0:
            payoff = float(resolved if side == 1 else 1 - resolved)
            reward = payoff - entry_cost
        else:
            if side == 1:
                bid1 = ep["bid1"]
                fee_close = calculate_kalshi_fee(bid1)
                exit_value_net = bid1 - fee_close
            else:
                ask1 = ep["ask1"]
                no_price_close = 1.0 - ask1
                fee_close = calculate_kalshi_fee(no_price_close)
                exit_value_net = no_price_close - fee_close
            reward = exit_value_net - entry_cost

        pnl.append(reward)

    return np.array(pnl)


def price_threshold_baseline(
    episodes: Sequence[Dict],
    *,
    lower: float = 0.4,
    upper: float = 0.6,
) -> np.ndarray:
    pnl: List[float] = []
    for ep in episodes:
        p0_mid = ep["price0"]
        ask0 = ep["ask0"]
        bid0 = ep["bid0"]
        resolved = ep["resolved"]

        reward = 0.0
        if p0_mid < lower:
            fee_open = calculate_kalshi_fee(ask0)
            entry_cost = ask0 + fee_open
            payoff = float(resolved)
            reward = payoff - entry_cost
        elif p0_mid > upper:
            no_price_open = 1.0 - bid0
            fee_open = calculate_kalshi_fee(no_price_open)
            entry_cost = no_price_open + fee_open
            payoff = float(1 - resolved)
            reward = payoff - entry_cost

        pnl.append(reward)

    return np.array(pnl)
