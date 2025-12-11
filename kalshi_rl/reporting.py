"""Reporting utilities for summaries and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize_pnl(pnl: np.ndarray) -> Dict[str, float]:
    mean = float(pnl.mean())
    std = float(pnl.std())
    sharpe = float(mean / std) if std > 0 else float("nan")
    frac_pos = float(np.mean(pnl > 0))
    return {"mean": mean, "std": std, "sharpe": sharpe, "frac_positive": frac_pos}


def describe_profitable_episodes(episodes: Sequence[Dict], pnl: np.ndarray) -> Dict[str, float]:
    profitable_mask = pnl > 0
    if not profitable_mask.any():
        return {"count": 0}

    subset = [episodes[i] for i, flag in enumerate(profitable_mask) if flag]
    price0 = np.array([ep["price0"] for ep in subset])
    spread0 = np.array([ep["ask0"] - ep["bid0"] for ep in subset])
    resolved = np.array([ep["resolved"] for ep in subset])

    return {
        "count": len(subset),
        "price0_mean": float(price0.mean()),
        "price0_median": float(np.median(price0)),
        "price0_min": float(price0.min()),
        "price0_max": float(price0.max()),
        "spread0_mean": float(spread0.mean()),
        "spread0_median": float(np.median(spread0)),
        "spread0_min": float(spread0.min()),
        "spread0_max": float(spread0.max()),
        "resolved_true_frac": float(np.mean(resolved)),
    }


def generate_plots(
    episodes: Sequence[Dict],
    rl_pnl: np.ndarray,
    baseline_pnl: np.ndarray,
    no_trade_pnl: np.ndarray,
    plots_dir: Path,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    fig, ax = plt.subplots()
    ax.hist(rl_pnl, bins=50, alpha=0.5, label="RL (test)")
    ax.hist(baseline_pnl, bins=50, alpha=0.5, label="Baseline (test)")
    ax.axvline(0.0, linestyle="--", color="black")
    ax.set_xlabel("P&L per episode")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of P&L (Test Set)")
    ax.legend()
    outputs["pnl_hist"] = _save_fig(fig, plots_dir / "pnl_hist.png")

    fig, ax = plt.subplots()
    ax.boxplot([rl_pnl, baseline_pnl, no_trade_pnl], labels=["RL", "Baseline", "No-trade"])
    ax.set_ylabel("P&L per episode")
    ax.set_title("P&L Boxplot by Strategy (Test Set)")
    outputs["pnl_boxplot"] = _save_fig(fig, plots_dir / "pnl_boxplot.png")

    fig, ax = plt.subplots()
    for pnl, label in [(rl_pnl, "RL"), (baseline_pnl, "Baseline")]:
        sorted_pnl = np.sort(pnl)
        cdf = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
        ax.plot(sorted_pnl, cdf, label=label)
    ax.set_xlabel("P&L per episode")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("P&L CDF – RL vs Baseline (Test Set)")
    ax.legend()
    outputs["pnl_cdf"] = _save_fig(fig, plots_dir / "pnl_cdf.png")

    fig, ax = plt.subplots()
    price0 = np.array([ep["price0"] for ep in episodes])
    ax.scatter(price0, baseline_pnl, s=5)
    ax.axhline(0.0, linestyle="--", color="black")
    ax.set_xlabel("Initial mid price (price0)")
    ax.set_ylabel("Baseline P&L per episode")
    ax.set_title("Baseline P&L vs Initial Mid Price (Test Set)")
    outputs["baseline_scatter"] = _save_fig(fig, plots_dir / "baseline_scatter.png")

    fig, ax = plt.subplots()
    profitable_mask = baseline_pnl > 0
    price0_prof = price0[profitable_mask]
    price0_unprof = price0[~profitable_mask]
    ax.hist(price0_prof, bins=30, alpha=0.5, label="Profitable")
    ax.hist(price0_unprof, bins=30, alpha=0.5, label="Not profitable")
    ax.set_xlabel("Initial mid price (price0)")
    ax.set_ylabel("Count")
    ax.set_title("Initial Mid Price for Profitable vs Non-Profitable Baseline Trades (Test)")
    ax.legend()
    outputs["baseline_profit_hist"] = _save_fig(fig, plots_dir / "baseline_profit_hist.png")

    return outputs


def _save_fig(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
