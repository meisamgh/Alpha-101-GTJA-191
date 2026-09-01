"""Cross-sectional portfolio construction with explicit exposure constraints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    long_quantile: float = 0.90
    short_quantile: float = 0.10
    gross_exposure: float = 1.0
    net_exposure: float = 0.0
    max_weight: float = 0.05
    long_only: bool = False


def construct_weights(
    scores: pd.Series,
    constraints: PortfolioConstraints | None = None,
    volatility: pd.Series | None = None,
    confidence: pd.Series | None = None,
) -> pd.Series:
    constraints = constraints or PortfolioConstraints()
    weights = pd.Series(0.0, index=scores.index)
    for _, daily in scores.groupby(level="date"):
        valid = daily.dropna()
        if len(valid) < 2:
            continue
        long_count = max(1, int(np.ceil(len(valid) * (1 - constraints.long_quantile))))
        short_count = max(1, int(np.ceil(len(valid) * constraints.short_quantile)))
        long_names = valid.nlargest(long_count).index
        short_names = valid.nsmallest(short_count).index
        raw = pd.Series(0.0, index=valid.index)
        raw.loc[long_names] = 1.0
        if not constraints.long_only:
            raw.loc[short_names] = -1.0
        if volatility is not None:
            raw *= 1 / volatility.reindex(raw.index).clip(lower=1e-6)
        if confidence is not None:
            raw *= confidence.reindex(raw.index).clip(0, 1).fillna(0)
        weights.loc[raw.index] = _normalize_sides(raw, constraints)
    return weights


def _normalize_sides(raw: pd.Series, c: PortfolioConstraints) -> pd.Series:
    if c.long_only:
        return _capped_allocation(raw.clip(lower=0), c.gross_exposure, c.max_weight)
    long_budget = (c.gross_exposure + c.net_exposure) / 2
    short_budget = (c.gross_exposure - c.net_exposure) / 2
    result = pd.Series(0.0, index=raw.index)
    positive, negative = raw.clip(lower=0), -raw.clip(upper=0)
    result += _capped_allocation(positive, long_budget, c.max_weight)
    result -= _capped_allocation(negative, short_budget, c.max_weight)
    return result.replace([np.inf, -np.inf], 0)


def _capped_allocation(scores: pd.Series, budget: float, cap: float) -> pd.Series:
    """Allocate proportionally with iterative redistribution around a hard name cap."""
    result = pd.Series(0.0, index=scores.index)
    remaining = scores[scores > 0].astype(float)
    residual_budget = min(budget, len(remaining) * cap)
    while len(remaining) and residual_budget > 1e-12:
        proposal = remaining / remaining.sum() * residual_budget
        capped = proposal >= cap
        if not capped.any():
            result.loc[remaining.index] += proposal
            break
        names = proposal.index[capped]
        result.loc[names] = cap
        residual_budget = budget - result.sum()
        remaining = remaining.drop(names)
    return result
