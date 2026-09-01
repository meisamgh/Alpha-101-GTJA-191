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


def backtest_overlapping_cohorts(
    panel: pd.DataFrame,
    signal_weights: pd.Series,
    horizon: int,
    cost_bps: float = 10,
    borrow_bps_annual: float = 50,
    overlapping: bool = True,
) -> BacktestResult:
    """Hold each signal from next open through close at the stated horizon.

    Overlapping portfolios allocate 1/horizon of gross capital to each daily cohort. The
    non-overlapping comparison admits only signal dates spaced by the holding horizon.
    """
    p = panel.sort_index()
    weights = signal_weights.reindex(p.index).fillna(0.0)
    if not overlapping:
        dates = pd.Index(p.index.get_level_values("date").unique()).sort_values()
        admitted = set(dates[::horizon])
        weights = weights.where(p.index.get_level_values("date").isin(admitted), 0.0)
        cohort_scale = 1.0
    else:
        cohort_scale = 1.0 / horizon
    close_return = p.groupby(level="symbol").close.pct_change(fill_method=None)
    intraday = p.close / p.open - 1
    gross_contributions = pd.Series(0.0, index=p.index)
    active_short = pd.Series(0.0, index=p.index)
    for age in range(1, horizon + 1):
        active = weights.groupby(level="symbol").shift(age) * cohort_scale
        period_return = intraday if age == 1 else close_return
        gross_contributions += active * period_return.fillna(0)
        active_short += (-active.clip(upper=0))
    gross = gross_contributions.groupby(level="date").sum()
    entry_notional = weights.abs().groupby(level="date").sum() * cohort_scale
    exit_notional = weights.groupby(level="symbol").shift(horizon).abs().groupby(level="date").sum(
    ) * cohort_scale
    costs = (entry_notional + exit_notional) * cost_bps / 10_000
    costs += active_short.groupby(level="date").sum() * borrow_bps_annual / 10_000 / 252
    daily = pd.DataFrame({"gross_return": gross, "cost": costs, "net_return": gross - costs})
    metrics = performance_metrics(daily.net_return, costs)
    metrics["turnover"] = float((entry_notional + exit_notional).mean())
    metrics["number_of_trades"] = float((weights != 0).sum())
    metrics["gross_cagr"] = performance_metrics(daily.gross_return)["cagr"]
    return BacktestResult(daily, metrics)
