"""Data loading and preprocessing helpers for the Kalshi RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class BigQueryConfig:
    """Configuration needed to read a table from BigQuery."""

    project_id: str
    table_id: str
    credentials_path: Optional[str] = None
    query: Optional[str] = None


def _create_bigquery_client(config: BigQueryConfig):
    """Create a BigQuery client, deferring imports until needed."""

    from google.cloud import bigquery  # type: ignore

    if config.credentials_path:
        from google.oauth2 import service_account  # type: ignore

        credentials = service_account.Credentials.from_service_account_file(
            config.credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=config.project_id, credentials=credentials)

    return bigquery.Client(project=config.project_id)


def load_market_data(
    *,
    csv_path: Optional[str] = None,
    bigquery_config: Optional[BigQueryConfig] = None,
) -> pd.DataFrame:
    """Load the full market-level DataFrame from CSV or BigQuery."""

    if csv_path:
        return pd.read_csv(csv_path)

    if bigquery_config is None:
        raise ValueError("Provide either csv_path or bigquery_config to load data.")

    client = _create_bigquery_client(bigquery_config)
    query = bigquery_config.query or f"SELECT * FROM `{bigquery_config.table_id}`"
    job = client.query(query)
    return job.result().to_dataframe()


def prepare_dataframe(
    df: pd.DataFrame,
    *,
    time_col: str = "days_to_resolution",
    spread_col: str = "spread",
    mid_col: str = "price_mid",
    resolved_col: str = "resolved",
) -> pd.DataFrame:
    """Filter to resolved markets and engineer bid/ask columns."""

    missing = [c for c in [time_col, spread_col, mid_col, resolved_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = df[df[resolved_col].notna()].copy()
    working[resolved_col] = working[resolved_col].astype(int)

    working["bid_price"] = (working[mid_col] - working[spread_col] / 2).clip(lower=0.0)
    working["ask_price"] = (working[mid_col] + working[spread_col] / 2).clip(upper=1.0)

    required = ["series_ticker", mid_col, resolved_col, time_col, "bid_price", "ask_price"]
    missing_after = [c for c in required if c not in working.columns]
    if missing_after:
        raise ValueError(f"Missing required columns after preprocessing: {missing_after}")

    return working


def build_episodes(
    df: pd.DataFrame,
    decision_points: Iterable[float],
    *,
    time_col: str = "days_to_resolution",
    contract_col: str = "series_ticker",
    price_col: str = "price_mid",
    bid_price_col: str = "bid_price",
    ask_price_col: str = "ask_price",
) -> List[Dict]:
    """
    Construct two-step offline RL episodes per contract keyed by decision points.
    """

    decision_points = list(decision_points)
    episodes: List[Dict] = []

    grouped = df.groupby(contract_col)
    for contract, group in grouped:
        g = group.sort_values(time_col, ascending=False)
        resolved_val = int(g["resolved"].iloc[0])

        prices: List[float] = []
        bid_prices: List[float] = []
        ask_prices: List[float] = []
        for d in decision_points:
            idx = (g[time_col] - d).abs().idxmin()
            row = g.loc[idx]

            prices.append(float(row[price_col]))
            bid_prices.append(float(row[bid_price_col]))
            ask_prices.append(float(row[ask_price_col]))

        if len(prices) != len(decision_points):
            continue

        episodes.append(
            {
                "contract_id": contract,
                "price0": prices[0],
                "price1": prices[1],
                "bid0": bid_prices[0],
                "ask0": ask_prices[0],
                "bid1": bid_prices[1],
                "ask1": ask_prices[1],
                "resolved": resolved_val,
            }
        )

    return episodes
