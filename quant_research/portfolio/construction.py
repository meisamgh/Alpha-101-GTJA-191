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
        raw = _normalize_sides(raw, constraints)
        weights.loc[raw.index] = raw.clip(-constraints.max_weight, constraints.max_weight)
        # Re-normalize after caps when feasible, without ever violating the cap.
        weights.loc[raw.index] = _normalize_sides(weights.loc[raw.index], constraints).clip(
            -constraints.max_weight, constraints.max_weight
        )
    return weights


def _normalize_sides(raw: pd.Series, c: PortfolioConstraints) -> pd.Series:
    if c.long_only:
        total = raw.clip(lower=0).sum()
        return raw.clip(lower=0) * (c.gross_exposure / total if total else 0)
    long_budget = (c.gross_exposure + c.net_exposure) / 2
    short_budget = (c.gross_exposure - c.net_exposure) / 2
    result = pd.Series(0.0, index=raw.index)
    positive, negative = raw.clip(lower=0), -raw.clip(upper=0)
    if positive.sum():
        result += positive / positive.sum() * long_budget
    if negative.sum():
        result -= negative / negative.sum() * short_budget
    return result.replace([np.inf, -np.inf], 0)
