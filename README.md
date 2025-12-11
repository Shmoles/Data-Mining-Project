# Kalshi Two-Step RL Project

This repository hosts the reinforcement-learning workflow previously run in
Google Colab. It fetches resolved Kalshi market data from BigQuery, constructs
two-step episodes, trains a tabular Q-learning agent, and compares it against a
simple price-threshold baseline (including Kalshi-style fees and spreads).

## Prerequisites

- Python 3.10+
- Access to the BigQuery table `thesis-476006.thesis_ds.all_data_filtered_regression`
- Google Cloud credentials with BigQuery read permissions  
  - Either run `gcloud auth application-default login` or set the environment
    variable `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the experiment

```bash
python kalshi_rl.py \
  --project-id thesis-476006 \
  --table-id thesis-476006.thesis_ds.all_data_filtered_regression \
  --episodes 200000
```

Flags you may want to adjust:

- `--test-size` (default `0.3`): portion of episodes held out for evaluation
- `--alpha`, `--gamma`, `--epsilon`: Q-learning hyperparameters
- `--plots-dir`: directory where diagnostic figures are saved (`plots/` default)
- `--no-plots`: skip plot generation (useful when running on a server)
- `--show-plots`: display plots in addition to saving them

Use `python kalshi_rl.py --help` to view every option.

## Outputs

- Console summary of P&L statistics for RL, baseline, and no-trade strategies
- Breakdown of profitable baseline trades (test set only)
- PNG figures saved under `plots/` unless the `--no-plots` flag is supplied