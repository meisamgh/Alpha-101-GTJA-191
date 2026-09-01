"""Reproducible experiment plots from saved daily results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_backtest(daily: pd.DataFrame, output_dir: Path) -> None:
    """Write cumulative return, drawdown, rolling Sharpe, turnover/cost and monthly plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("install the research extra to create plots") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = daily["net_return"]
    wealth = (1 + returns).cumprod()
    series = {
        "cumulative_net_return": wealth - 1,
        "drawdown": wealth / wealth.cummax() - 1,
        "rolling_sharpe_63d": returns.rolling(63).mean() / returns.rolling(63).std() * 252**0.5,
        "daily_cost": daily["cost"],
    }
    for name, values in series.items():
        ax = values.plot(title=name.replace("_", " ").title(), figsize=(9, 4))
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"{name}.png", dpi=140)
        plt.close(ax.figure)
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    table = monthly.to_frame("return")
    table["year"], table["month"] = table.index.year, table.index.month
    pivot = table.pivot(index="year", columns="month", values="return")
    ax = pivot.plot(kind="bar", figsize=(10, 4), title="Monthly returns")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "monthly_returns.png", dpi=140)
    plt.close(ax.figure)
