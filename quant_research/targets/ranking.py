"""Cross-sectional target helpers."""
import pandas as pd


def cross_sectional_rank(values: pd.Series) -> pd.Series:
    if "date" not in values.index.names:
        raise ValueError("values must have a date index level")
    return values.groupby(level="date").rank(pct=True)
