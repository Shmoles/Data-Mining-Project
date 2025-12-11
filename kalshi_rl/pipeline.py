"""End-to-end pipeline orchestration for the Kalshi RL project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .data import BigQueryConfig, build_episodes, load_market_data, prepare_dataframe
from .environment import KalshiTwoStepEnv
from .reporting import describe_profitable_episodes, generate_plots, summarize_pnl
from .rl import price_threshold_baseline, q_learning, run_greedy_policy


@dataclass
class PipelineConfig:
    decision_days: Sequence[float] = (2.0, 0.25)
    time_col: str = "days_to_resolution"
    spread_col: str = "spread"
    mid_col: str = "price_mid"
    resolved_col: str = "resolved"
    test_size: float = 0.3
    random_state: int = 42
    n_price_buckets: int = 5
    n_spread_buckets: int = 5
    q_learning_episodes: int = 200_000
    alpha: float = 0.1
    gamma: float = 1.0
    epsilon: float = 0.1
    baseline_lower: float = 0.4
    baseline_upper: float = 0.6


@dataclass
class PipelineResult:
    episodes_total: int
    train_episodes: int
    test_episodes: int
    rl_stats: Dict[str, float]
    baseline_stats: Dict[str, float]
    no_trade_stats: Dict[str, float]
    profitable_stats: Dict[str, float]
    plot_paths: Dict[str, Path]


def run_pipeline(
    *,
    csv_path: Optional[str] = None,
    bigquery_config: Optional[BigQueryConfig] = None,
    config: PipelineConfig = PipelineConfig(),
    plots_dir: Optional[Path] = None,
) -> PipelineResult:
    df = load_market_data(csv_path=csv_path, bigquery_config=bigquery_config)
    prepared = prepare_dataframe(
        df,
        time_col=config.time_col,
        spread_col=config.spread_col,
        mid_col=config.mid_col,
        resolved_col=config.resolved_col,
    )

    episodes = build_episodes(
        prepared,
        config.decision_days,
        time_col=config.time_col,
        contract_col="series_ticker",
        price_col=config.mid_col,
        bid_price_col="bid_price",
        ask_price_col="ask_price",
    )

    if not episodes:
        raise ValueError("No episodes constructed. Check decision points or source data.")

    train_eps, test_eps = train_test_split(
        episodes,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    env_train = KalshiTwoStepEnv(
        train_eps,
        n_price_buckets=config.n_price_buckets,
        n_spread_buckets=config.n_spread_buckets,
    )

    Q = q_learning(
        env_train,
        n_episodes=config.q_learning_episodes,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
    )

    rl_pnl = run_greedy_policy(
        test_eps,
        Q,
        n_price_buckets=config.n_price_buckets,
        n_spread_buckets=config.n_spread_buckets,
    )
    baseline_pnl = price_threshold_baseline(
        test_eps,
        lower=config.baseline_lower,
        upper=config.baseline_upper,
    )
    no_trade_pnl = np.zeros_like(baseline_pnl)

    rl_stats = summarize_pnl(rl_pnl)
    baseline_stats = summarize_pnl(baseline_pnl)
    no_trade_stats = summarize_pnl(no_trade_pnl)
    profitable_stats = describe_profitable_episodes(test_eps, baseline_pnl)

    plot_paths: Dict[str, Path] = {}
    if plots_dir is not None:
        plot_paths = generate_plots(test_eps, rl_pnl, baseline_pnl, no_trade_pnl, plots_dir)

    return PipelineResult(
        episodes_total=len(episodes),
        train_episodes=len(train_eps),
        test_episodes=len(test_eps),
        rl_stats=rl_stats,
        baseline_stats=baseline_stats,
        no_trade_stats=no_trade_stats,
        profitable_stats=profitable_stats,
        plot_paths=plot_paths,
    )
