"""Linear cost model in basis points of traded notional."""
import pandas as pd


def transaction_costs(
    weights: pd.Series,
    commission_bps: float = 2,
    spread_slippage_bps: float = 5,
    borrow_bps_annual: float = 0,
) -> pd.Series:
    previous = weights.groupby(level="symbol").shift(1).fillna(0)
    turnover = (weights - previous).abs().groupby(level="date").sum()
    trading = turnover * (commission_bps + spread_slippage_bps) / 10_000
    short_gross = (-weights.clip(upper=0)).groupby(level="date").sum()
    return trading + short_gross * borrow_bps_annual / 10_000 / 252
