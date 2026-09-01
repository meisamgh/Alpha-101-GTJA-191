"""Targets aligned to after-close signals and next-open execution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_return_targets(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    volatility_window: int = 20,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Return open[t+1] -> close[t+h], plus residual, vol-adjusted and rank targets."""
    p = panel.sort_index()
    g = p.groupby(level="symbol", group_keys=False)
    out = pd.DataFrame(index=p.index)
    trailing_beta = _trailing_beta(p, window=60)
    vol = g.adjusted_close.pct_change(fill_method=None).groupby(level="symbol").transform(
        lambda x: x.rolling(volatility_window, min_periods=volatility_window).std()
    )
    for h in horizons:
        entry = g.open.shift(-1)
        exit_price = g.close.shift(-h)
        raw = exit_price / entry - 1
        market_future = _future_compound_return(p.get("market_return"), h)
        residual = raw - trailing_beta * market_future if market_future is not None else raw
        if "sector" in p:
            # Cross-sectional sector component on the target date. This uses future labels only to
            # define the supervised target, never a feature; leave-one-out reduces self-influence.
            sector_component = _leave_one_out_group_mean(residual, p["sector"])
            residual = residual - sector_component.fillna(0)
        adjusted = (residual - cost_bps / 10_000) / (vol * np.sqrt(max(h, 1))).replace(0, np.nan)
        out[f"raw_return_{h}d"] = raw
        out[f"residual_return_{h}d"] = residual
        out[f"vol_adjusted_return_{h}d"] = adjusted
        out[f"rank_target_{h}d"] = adjusted.groupby(level="date").rank(pct=True)
        out[f"label_end_{h}d"] = pd.Series(
            p.index.get_level_values("date"), index=p.index
        ).groupby(level="symbol").shift(-h)
    return out


def _trailing_beta(panel: pd.DataFrame, window: int) -> pd.Series:
    stock = panel.groupby(level="symbol").adjusted_close.pct_change(fill_method=None)
    if "market_return" not in panel:
        return pd.Series(0.0, index=panel.index)
    market = panel.market_return
    cov = stock.groupby(level="symbol").transform(
        lambda x: x.rolling(window, min_periods=max(20, window // 2)).cov(market.loc[x.index])
    )
    var = market.groupby(level="symbol").transform(
        lambda x: x.rolling(window, min_periods=max(20, window // 2)).var()
    )
    return cov / var.replace(0, np.nan)


def _future_compound_return(market: pd.Series | None, horizon: int) -> pd.Series | None:
    if market is None:
        return None
    return market.groupby(level="symbol").transform(
        lambda x: (1 + x.shift(-1)).rolling(horizon).apply(np.prod, raw=True).shift(
            -(horizon - 1)
        ) - 1
    )


def _leave_one_out_group_mean(values: pd.Series, sectors: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": values, "sector": sectors}, index=values.index)
    keys = [frame.index.get_level_values("date"), frame.sector]
    total = frame.value.groupby(keys).transform("sum")
    count = frame.value.groupby(keys).transform("count")
    return (total - frame.value) / (count - 1).replace(0, np.nan)
