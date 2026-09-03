"""Validated representative GTJA-191 factors with a provider-independent API."""
import numpy as np
import pandas as pd


def compute_gtja_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_index()
    out = pd.DataFrame(index=p.index)
    # GTJA alpha 2: -delta(((C-L)-(H-C))/(H-L), 1)
    location = ((p.close - p.low) - (p.high - p.close)) / (p.high - p.low).replace(0, np.nan)
    out["gtja_002"] = -location.groupby(level="symbol").diff()
    # GTJA alpha 12: -(open - mean(vwap,10)) * abs(close-vwap), ranked cross-sectionally.
    vwap = p.get("vwap", (p.high + p.low + p.close) / 3)
    mean_vwap = vwap.groupby(level="symbol").transform(lambda x: x.rolling(10).mean())
    lhs = (p.open - mean_vwap).groupby(level="date").rank(pct=True)
    rhs = (p.close - vwap).abs().groupby(level="date").rank(pct=True)
    out["gtja_012"] = -lhs * rhs
    return out.astype(float)
