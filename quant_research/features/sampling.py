"""Symmetric CUSUM event filter using past volatility thresholds."""
import pandas as pd


def cusum_events(returns: pd.Series, threshold: pd.Series) -> pd.MultiIndex:
    events: list[tuple[object, object]] = []
    for symbol, values in returns.groupby(level="symbol"):
        positive = negative = 0.0
        for idx, value in values.items():
            h = threshold.loc[idx]
            if pd.isna(value) or pd.isna(h) or h <= 0:
                continue
            positive = max(0.0, positive + float(value))
            negative = min(0.0, negative + float(value))
            if positive > h or negative < -h:
                events.append((idx[0], symbol))
                positive = negative = 0.0
    return pd.MultiIndex.from_tuples(events, names=["date", "symbol"])
