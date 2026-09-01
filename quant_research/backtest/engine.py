"""Vectorized next-open-to-close portfolio backtest."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.portfolio.costs import transaction_costs


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    metrics: dict[str, float]


def backtest(
    panel: pd.DataFrame, signal_weights: pd.Series, cost_bps: float = 10
) -> BacktestResult:
    p = panel.sort_index()
    # Weight made after close t is applied to open[t+1] -> close[t+1].
    next_intraday = (p.close / p.open - 1).groupby(level="symbol").shift(-1)
    gross = (signal_weights * next_intraday).groupby(level="date").sum(min_count=1).fillna(0)
    costs = transaction_costs(signal_weights, commission_bps=0,
                              spread_slippage_bps=cost_bps).reindex(gross.index).fillna(0)
    daily = pd.DataFrame({"gross_return": gross, "cost": costs, "net_return": gross - costs})
    return BacktestResult(daily=daily, metrics=performance_metrics(daily.net_return, costs))


def performance_metrics(returns: pd.Series, costs: pd.Series | None = None) -> dict[str, float]:
    import numpy as np

    wealth = (1 + returns).cumprod()
    years = max(len(returns) / 252, 1 / 252)
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if len(wealth) else np.nan
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() else 0.0
    downside = returns.where(returns < 0).std() * np.sqrt(252)
    drawdown = wealth / wealth.cummax() - 1
    return {"cagr": float(cagr), "annual_volatility": float(vol), "sharpe": float(sharpe),
            "sortino": float(returns.mean() * 252 / downside) if downside else 0.0,
            "max_drawdown": float(drawdown.min()),
            "cost_drag": float(costs.sum()) if costs is not None else 0.0}
