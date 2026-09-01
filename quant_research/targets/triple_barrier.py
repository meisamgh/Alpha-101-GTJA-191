"""Volatility-scaled triple-barrier labels bounded by each event's vertical barrier."""
from __future__ import annotations

import pandas as pd


def triple_barrier_labels(
    panel: pd.DataFrame,
    events: pd.MultiIndex | None = None,
    horizon: int = 5,
    pt: float = 1.5,
    sl: float = 1.0,
    volatility_window: int = 20,
) -> pd.DataFrame:
    p = panel.sort_index()
    if events is None:
        events = p.index
    daily = p.groupby(level="symbol").adjusted_close.pct_change(fill_method=None)
    vol = daily.groupby(level="symbol").transform(
        lambda x: x.rolling(volatility_window, min_periods=volatility_window).std()
    )
    rows: list[dict[str, object]] = []
    for date, symbol in events:
        asset = p.xs(symbol, level="symbol")
        if date not in asset.index or pd.isna(vol.loc[(date, symbol)]):
            continue
        loc = asset.index.get_loc(date)
        future = asset.iloc[loc + 1 : loc + horizon + 1]
        if future.empty:
            continue
        entry = future.open.iloc[0]
        upper = pt * float(vol.loc[(date, symbol)])
        lower = -sl * float(vol.loc[(date, symbol)])
        label, end, realized = 0, future.index[-1], future.close.iloc[-1] / entry - 1
        for d, row in future.iterrows():
            if row.high / entry - 1 >= upper:
                label, end, realized = 1, d, upper
                break
            if row.low / entry - 1 <= lower:
                label, end, realized = -1, d, lower
                break
        rows.append({"date": date, "symbol": symbol, "label": label,
                     "label_end": end, "realized_return": realized})
    if not rows:
        return pd.DataFrame(columns=["label", "label_end", "realized_return"]).set_index(
            pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
