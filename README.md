# Kalshi Two-Step RL Pipeline

This repository implements a fully reproducible pipeline for training and evaluating a two-step reinforcement-learning agent on Kalshi market data. The pipeline loads resolved markets from BigQuery (or a local CSV), reconstructs bid/ask prices, builds offline episodes, trains a tabular Q-learning policy, benchmarks it against a price-threshold baseline, and produces summary metrics plus publication-ready plots.

## Features

- BigQuery loader with optional service-account impersonation or local CSV fallback
- Deterministic episode construction for configurable decision horizons
- Fee- and spread-aware Kalshi execution model via `KalshiTwoStepEnv`
- Tabular Q-learning trainer + greedy policy evaluation on a strict test split
- Baseline strategy + profitable-trade diagnostics and saved matplotlib figures
- CLI with rich configuration flags and non-interactive plotting to `reports/plots`

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- Access to the relevant BigQuery table (default expects `price_mid`, `spread`, `days_to_resolution`, `resolved`, `series_ticker`)

## Authenticating with BigQuery

The pipeline uses the standard Google Cloud client libraries. Choose one of the following authentication flows:

1. **Application Default Credentials (ADC)**
   ```bash
   gcloud auth application-default login
   ```
   Then run the pipeline without `--credentials`.

2. **Service Account JSON key**
   - Download a JSON key with BigQuery read permissions.
   - Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` or pass `--credentials /path/to/key.json`.

3. **Workload Identity / Metadata**
   - When running on GCE/GKE/Cloud Run simply ensure the runtime identity has `bigquery.readsessions.create` and `bigquery.read` roles.

## Running the Pipeline

```bash
python -m kalshi_rl.cli \
  --project-id thesis-476006 \
  --table-id thesis-476006.thesis_ds.all_data_filtered_regression \
  --credentials /path/to/service-account.json \
  --decision-days 2.0 0.25 \
  --q-episodes 200000 \
  --plots-dir reports/plots
```

### Common Flags

| Flag | Description |
| --- | --- |
| `--csv-path data.csv` | Skip BigQuery and load from a local CSV. |
| `--decision-days 2 0.25` | Decision horizons (days to resolution). |
| `--test-size 0.3` | Fraction of episodes reserved for evaluation. |
| `--baseline-range 0.4 0.6` | Mid-price thresholds for the baseline strategy. |
| `--n-price-buckets`, `--n-spread-buckets` | State discretisation granularity. |
| `--q-episodes`, `--alpha`, `--gamma`, `--epsilon` | Q-learning hyper-parameters. |
| `--plots-dir reports/plots` | Output directory for figures (use `--no-plots` to skip). |

### Outputs

- Console summary of train/test sizes, Sharpe-style metrics, and profitable-trade stats.
- Saved plots: histogram, boxplot, CDF, scatter, and profitability histogram (all under `reports/plots` by default).
- Reproducible config via CLI arguments or environment variables (`BIGQUERY_PROJECT_ID`, `BIGQUERY_TABLE_ID`).

## Project Structure

```
kalshi_rl/
├── cli.py              # CLI + summary printer
├── data.py             # BigQuery/CSV loaders + preprocessing
├── environment.py      # KalshiTwoStepEnv with fee-aware execution
├── pipeline.py         # End-to-end orchestration helpers
├── pricing.py          # Bucketing + Kalshi fee utilities
├── reporting.py        # Stats + matplotlib outputs
└── rl.py               # Tabular Q-learning + evaluation policies
```

Feel free to adapt the CLI or import `run_pipeline` directly inside notebooks or analysis scripts.
