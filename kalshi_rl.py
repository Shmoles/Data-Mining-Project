#!/usr/bin/env python3
"""
Two-step reinforcement learning study for Kalshi markets.

This script mirrors the original Google Colab workflow while making it easy
to run locally inside this repository. It pulls resolved market data from
BigQuery, constructs two-decision-step episodes, trains a tabular Q-learning
agent, evaluates against a simple price-threshold baseline, and generates a
handful of diagnostic plots.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

DECISION_DAYS = [2.0, 0.25]
DEFAULT_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "thesis-476006")
DEFAULT_TABLE_ID = os.getenv(
    "BIGQUERY_TABLE_ID", "thesis-476006.thesis_ds.all_data_filtered_regression"
)


def authenticate_if_colab():
    """
    When running inside Google Colab we need to call `auth.authenticate_user()`
    once. ImportError is expected when running locally, so we swallow it.
    """
    try:
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
        print("Authenticated with Google Colab user credentials.")
    except ImportError:
        # Running outside Colab – assume ADC or service-account credentials.
        pass


def fetch_dataframe(project_id: str, table_id: str) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{table_id}`"
    df = client.query(query).to_dataframe()
    print(f"Raw DataFrame shape: {df.shape}")
    return df


def prepare_dataframe(df: pd.DataFrame, time_col: str = "days_to_resolution") -> pd.DataFrame:
    df = df[df["resolved"].notna()].copy()
    df["resolved"] = df["resolved"].astype(int)

    if time_col not in df.columns:
        raise ValueError(f"Expected a '{time_col}' column with days to resolution.")
    if "spread" not in df.columns:
        raise ValueError("Expected a 'spread' column for bid/ask reconstruction.")

    df["bid_price"] = (df["price_mid"] - df["spread"] / 2).clip(lower=0.0)
    df["ask_price"] = (df["price_mid"] + df["spread"] / 2).clip(upper=1.0)

    required_cols = [
        "series_ticker",
        "price_mid",
        "resolved",
        time_col,
        "bid_price",
        "ask_price",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Remaining rows after filter: {df.shape[0]}")
    print(df[required_cols].head())
    return df


def build_episodes(
    df: pd.DataFrame,
    decision_points: Sequence[float] = DECISION_DAYS,
    time_col: str = "days_to_resolution",
    contract_col: str = "series_ticker",
) -> List[Dict]:
    episodes: List[Dict] = []
    grouped = df.groupby(contract_col)

    for _, group in grouped:
        g = group.sort_values(time_col, ascending=False)
        resolved_val = int(g["resolved"].iloc[0])

        prices: List[float] = []
        bid_prices: List[float] = []
        ask_prices: List[float] = []

        for decision_time in decision_points:
            idx = (g[time_col] - decision_time).abs().idxmin()
            row = g.loc[idx]
            prices.append(float(row["price_mid"]))
            bid_prices.append(float(row["bid_price"]))
            ask_prices.append(float(row["ask_price"]))

        if len(prices) != 2:
            continue

        episodes.append(
            {
                "contract_id": g[contract_col].iloc[0],
                "price0": prices[0],
                "price1": prices[1],
                "bid0": bid_prices[0],
                "ask0": ask_prices[0],
                "bid1": bid_prices[1],
                "ask1": ask_prices[1],
                "resolved": resolved_val,
            }
        )

    print(f"Number of episodes: {len(episodes)}")
    return episodes


def price_bucket(price: float, n_buckets: int = 5) -> int:
    p = max(0.0, min(1.0, price))
    bucket = int(p * n_buckets)
    if bucket == n_buckets:
        bucket = n_buckets - 1
    return bucket


def spread_bucket(spread: float, n_buckets: int = 5, max_spread: float = 0.10) -> int:
    s = max(0.0, min(max_spread, spread))
    bucket = int((s / max_spread) * n_buckets)
    if bucket == n_buckets:
        bucket = n_buckets - 1
    return bucket


def calculate_kalshi_fee(price_yes_side: float, contracts: int = 1, factor: float = 0.07) -> float:
    price_yes_side = max(0.0, min(1.0, price_yes_side))
    base = factor * contracts * price_yes_side * (1.0 - price_yes_side)
    fee_cents = math.ceil(base * 100.0)
    return fee_cents / 100.0


class KalshiTwoStepEnv:
    def __init__(self, episodes: List[Dict], n_price_buckets: int = 5, n_spread_buckets: int = 5):
        self.episodes = episodes
        self.n_price_buckets = n_price_buckets
        self.n_spread_buckets = n_spread_buckets

        self.ep: Dict | None = None
        self.step_idx = 0
        self.holding = False
        self.side = 0
        self.entry_cost: float | None = None

    def _encode_state(self) -> Tuple[int, int, int, int, int]:
        assert self.ep is not None
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

    def reset(self):
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
        if self.ep is None:
            raise RuntimeError("Environment not reset.")

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
                    payoff = float(resolved) if self.side == 1 else float(1 - resolved)
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


def q_learning(
    env: KalshiTwoStepEnv,
    n_episodes: int = 50_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon: float = 0.1,
) -> Dict:
    Q = defaultdict(lambda: defaultdict(float))
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.valid_actions()
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_state = Q[state]
                action = max(actions, key=lambda act: q_state[act])

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
    episodes: Iterable[Dict],
    Q: Dict,
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
            payoff = float(resolved) if side == 1 else float(1 - resolved)
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
    episodes: Iterable[Dict],
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


def summarize(pnl: np.ndarray, name: str):
    mean = pnl.mean()
    std = pnl.std()
    sharpe = mean / std if std > 0 else float("nan")
    frac_pos = float(np.mean(pnl > 0)) if len(pnl) > 0 else float("nan")
    print(f"{name}:")
    print(f"  Mean P&L: {mean:.4f}")
    print(f"  Std P&L:  {std:.4f}")
    print(f"  Sharpe:   {sharpe:.4f}")
    print(f"  % > 0:    {frac_pos:.3f}")
    print("")


def profitable_trade_characteristics(test_eps: Sequence[Dict], baseline_pnl: np.ndarray):
    profitable_mask = baseline_pnl > 0
    profitable_indices = np.where(profitable_mask)[0]
    print(f"Number of profitable trades (baseline, test): {len(profitable_indices)}")

    if not profitable_indices.size:
        print("No profitable baseline trades on the test set.")
        return

    profitable_eps = [test_eps[i] for i in profitable_indices]
    price0 = np.array([ep["price0"] for ep in profitable_eps])
    spread0 = np.array([ep["ask0"] - ep["bid0"] for ep in profitable_eps])
    resolved = [ep["resolved"] for ep in profitable_eps]

    print("\n--- Characteristics of Profitable Trades (Baseline, Test Only) ---")
    print(f"Profitable Price0 (Mean):   {np.mean(price0):.4f}")
    print(f"Profitable Price0 (Median): {np.median(price0):.4f}")
    print(f"Profitable Price0 (Min):    {np.min(price0):.4f}")
    print(f"Profitable Price0 (Max):    {np.max(price0):.4f}")

    print(f"\nProfitable Spread0 (Mean):   {np.mean(spread0):.4f}")
    print(f"Profitable Spread0 (Median): {np.median(spread0):.4f}")
    print(f"Profitable Spread0 (Min):    {np.min(spread0):.4f}")
    print(f"Profitable Spread0 (Max):    {np.max(spread0):.4f}")

    resolved_counts = pd.Series(resolved).value_counts(normalize=True)
    print("\nProfitable Resolved Status Distribution (Baseline, Test Only):")
    print(resolved_counts)


def save_plots(
    test_eps: Sequence[Dict],
    rl_pnl: np.ndarray,
    baseline_pnl: np.ndarray,
    output_dir: Path,
    show_plots: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    def finalize(path: Path):
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        if show_plots:
            plt.show()
        plt.close()
        print(f"Saved plot: {path}")

    plt.figure()
    plt.hist(rl_pnl, bins=50, alpha=0.5, label="RL (test)")
    plt.hist(baseline_pnl, bins=50, alpha=0.5, label="Baseline (test)")
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("P&L per episode")
    plt.ylabel("Count")
    plt.title("Distribution of P&L (Test Set)")
    plt.legend()
    finalize(output_dir / "pnl_distribution.png")

    plt.figure()
    plt.boxplot([rl_pnl, baseline_pnl, np.zeros_like(baseline_pnl)], labels=["RL", "Baseline", "No-trade"])
    plt.ylabel("P&L per episode")
    plt.title("P&L Boxplot by Strategy (Test Set)")
    finalize(output_dir / "pnl_boxplot.png")

    plt.figure()
    for pnl, label in [(rl_pnl, "RL"), (baseline_pnl, "Baseline")]:
        sorted_pnl = np.sort(pnl)
        cdf = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
        plt.plot(sorted_pnl, cdf, label=label)
    plt.xlabel("P&L per episode")
    plt.ylabel("Cumulative fraction")
    plt.title("P&L CDF – RL vs Baseline (Test Set)")
    plt.legend()
    finalize(output_dir / "pnl_cdf.png")

    plt.figure()
    plt.scatter([ep["price0"] for ep in test_eps], baseline_pnl, s=5)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Initial mid price (price0)")
    plt.ylabel("Baseline P&L per episode")
    plt.title("Baseline P&L vs Initial Mid Price (Test Set)")
    finalize(output_dir / "baseline_pnl_vs_price0.png")

    profitable_mask = baseline_pnl > 0
    price0_prof = np.array([ep["price0"] for ep in test_eps])[profitable_mask]
    price0_unprof = np.array([ep["price0"] for ep in test_eps])[~profitable_mask]

    plt.figure()
    plt.hist(price0_prof, bins=30, alpha=0.5, label="Profitable")
    plt.hist(price0_unprof, bins=30, alpha=0.5, label="Not profitable")
    plt.xlabel("Initial mid price (price0)")
    plt.ylabel("Count")
    plt.title("Initial Mid Price for Profitable vs Non-Profitable Baseline Trades (Test)")
    plt.legend()
    finalize(output_dir / "price0_hist_profitable_vs_not.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kalshi two-step RL experiment.")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="BigQuery project id.")
    parser.add_argument("--table-id", default=DEFAULT_TABLE_ID, help="BigQuery table id.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test-set fraction.")
    parser.add_argument("--episodes", type=int, default=200_000, help="Training episodes for Q-learning.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate.")
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"), help="Directory to save figures.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots in addition to saving.")
    return parser.parse_args()


def main():
    args = parse_args()
    authenticate_if_colab()

    df = fetch_dataframe(args.project_id, args.table_id)
    df = prepare_dataframe(df)
    episodes = build_episodes(df)

    if not episodes:
        raise RuntimeError("No episodes generated; check data inputs.")

    train_eps, test_eps = train_test_split(episodes, test_size=args.test_size, random_state=42)

    env_train = KalshiTwoStepEnv(train_eps, n_price_buckets=5, n_spread_buckets=5)
    Q = q_learning(
        env_train,
        n_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
    )

    rl_pnl = run_greedy_policy(test_eps, Q)
    baseline_pnl = price_threshold_baseline(test_eps)
    no_trade_pnl = np.zeros_like(baseline_pnl)

    summarize(rl_pnl, "RL policy (test set)")
    summarize(baseline_pnl, "Price-threshold baseline (test set)")
    summarize(no_trade_pnl, "No-trade (test set)")

    profitable_trade_characteristics(test_eps, baseline_pnl)

    if not args.no_plots:
        save_plots(test_eps, rl_pnl, baseline_pnl, args.plots_dir, args.show_plots)
    else:
        print("Skipping plot generation (--no-plots).")


if __name__ == "__main__":
    main()
