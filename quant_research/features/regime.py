"""Trailing regime features and deterministic gate."""
import pandas as pd


def hard_regime_gate(
    features: pd.DataFrame,
    trend_threshold: float = 0.10,
    min_vol_quantile: float = 0.10,
) -> pd.Series:
    vol = features["realized_vol_20d"]
    trailing_floor = vol.groupby(level="symbol").transform(
        lambda x: x.rolling(252, min_periods=60).quantile(min_vol_quantile)
    )
    return (features["trend_strength_20"] > trend_threshold) & (vol > trailing_floor)
