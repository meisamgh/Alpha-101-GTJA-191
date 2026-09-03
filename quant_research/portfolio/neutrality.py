"""Cross-sectional projection onto market- and sector-neutral exposure space."""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_weights(
    weights: pd.Series,
    beta: pd.Series,
    sector: pd.Series,
    gross_limit: float = 1.0,
    max_weight: float = 0.02,
) -> pd.Series:
    """Project daily weights off intercept, beta, and sector indicator exposures.

    Gross exposure may fall below the limit when the name cap binds. Missing beta values are
    replaced by the same-date median; missing sectors are assigned ``UNKNOWN``.
    """
    result = pd.Series(0.0, index=weights.index)
    for _, daily in weights.groupby(level="date"):
        active = daily[daily.ne(0)]
        if len(active) < 4:
            continue
        daily_beta = beta.reindex(active.index).astype(float)
        daily_beta = daily_beta.fillna(daily_beta.median()).fillna(0.0)
        daily_sector = sector.reindex(active.index).fillna("UNKNOWN").astype(str)
        dummies = pd.get_dummies(daily_sector, dtype=float)
        design = np.column_stack(
            [np.ones(len(active)), daily_beta.to_numpy(), dummies.to_numpy()[:, :-1]]
        )
        projected = active.to_numpy(dtype=float, copy=True)
        projected -= design @ (np.linalg.pinv(design) @ projected)
        gross = np.abs(projected).sum()
        if gross <= 1e-12:
            continue
        scale = min(gross_limit / gross, max_weight / np.abs(projected).max())
        result.loc[active.index] = projected * scale
    return result
