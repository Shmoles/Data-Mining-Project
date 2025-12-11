"""Command-line interface to run the Kalshi RL pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .data import BigQueryConfig
from .pipeline import PipelineConfig, PipelineResult, run_pipeline


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Kalshi RL agent")
    parser.add_argument("--project-id", default=os.environ.get("BIGQUERY_PROJECT_ID"), help="BigQuery project ID")
    parser.add_argument(
        "--table-id",
        default=os.environ.get("BIGQUERY_TABLE_ID"),
        help="Fully-qualified table ID (project.dataset.table)",
    )
    parser.add_argument(
        "--credentials",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to a service-account JSON key (optional with ADC)",
    )
    parser.add_argument("--csv-path", help="Optional CSV file to bypass BigQuery", default=None)
    parser.add_argument(
        "--decision-days",
        type=float,
        nargs="+",
        default=[2.0, 0.25],
        help="Decision points in days before resolution",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="Test fraction for episodes")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--n-price-buckets", type=int, default=5)
    parser.add_argument("--n-spread-buckets", type=int, default=5)
    parser.add_argument("--q-episodes", type=int, default=200_000, help="Number of Q-learning episodes")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--baseline-range",
        type=float,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        default=(0.4, 0.6),
        help="Thresholds for the price-based baseline",
    )
    parser.add_argument(
        "--plots-dir",
        default="reports/plots",
        help="Directory to store plots (omit or use --no-plots to skip)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.csv_path and (not args.project_id or not args.table_id):
        raise SystemExit("Provide --csv-path or both --project-id and --table-id.")

    bigquery_config = None
    if not args.csv_path:
        bigquery_config = BigQueryConfig(
            project_id=args.project_id,
            table_id=args.table_id,
            credentials_path=args.credentials,
        )

    config = PipelineConfig(
        decision_days=args.decision_days,
        test_size=args.test_size,
        random_state=args.random_state,
        n_price_buckets=args.n_price_buckets,
        n_spread_buckets=args.n_spread_buckets,
        q_learning_episodes=args.q_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        baseline_lower=args.baseline_range[0],
        baseline_upper=args.baseline_range[1],
    )

    plots_dir = None
    if not args.no_plots:
        plots_dir = Path(args.plots_dir)

    result = run_pipeline(
        csv_path=args.csv_path,
        bigquery_config=bigquery_config,
        config=config,
        plots_dir=plots_dir,
    )

    _print_summary(result)


def _print_summary(result: PipelineResult) -> None:
    print("=== Kalshi RL Pipeline ===")
    print(f"Total episodes: {result.episodes_total}")
    print(f"Train episodes: {result.train_episodes}")
    print(f"Test episodes:  {result.test_episodes}")
    print("")

    print("RL policy (test set):")
    _print_stats(result.rl_stats)
    print("Price-threshold baseline (test set):")
    _print_stats(result.baseline_stats)
    print("No-trade baseline (test set):")
    _print_stats(result.no_trade_stats)

    print("\nProfitable baseline episodes (test set):")
    for key, value in result.profitable_stats.items():
        print(f"  {key}: {value}")

    if result.plot_paths:
        print("\nSaved plots:")
        for name, path in result.plot_paths.items():
            print(f"  {name}: {path}")


def _print_stats(stats: dict) -> None:
    print(f"  Mean:   {stats['mean']:.4f}")
    print(f"  Std:    {stats['std']:.4f}")
    print(f"  Sharpe: {stats['sharpe']:.4f}")
    print(f"  % > 0:  {stats['frac_positive']:.3f}")


if __name__ == "__main__":
    main()
