"""Meta-label construction utilities."""
import pandas as pd


def make_meta_target(primary_side: pd.Series, realized_return: pd.Series) -> pd.Series:
    aligned = primary_side.align(realized_return, join="inner")
    return (aligned[0] * aligned[1] > 0).astype(int)
