"""Fail-fast market-data quality checks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_panel(panel: pd.DataFrame, jump_threshold: float = 0.50) -> DataQualityReport:
    report = DataQualityReport()
    if panel.index.names != ["date", "symbol"]:
        report.errors.append("index must be exactly (date, symbol)")
        return report
    if panel.index.duplicated().any():
        report.errors.append(f"duplicate rows: {int(panel.index.duplicated().sum())}")
    prices = panel[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or not np.isfinite(prices).all().all():
        report.errors.append("prices must be positive and finite")
    bad_ohlc = (panel["high"] < prices[["open", "close", "low"]].max(axis=1)) | (
        panel["low"] > prices[["open", "close", "high"]].min(axis=1)
    )
    if bad_ohlc.any():
        report.errors.append(f"invalid OHLC rows: {int(bad_ohlc.sum())}")
    if (panel["volume"] < 0).any():
        report.errors.append("negative volume")
    returns = panel.groupby(level="symbol")["adjusted_close"].pct_change(fill_method=None)
    if (returns.abs() > jump_threshold).any():
        report.warnings.append("extreme adjusted-price jumps detected; inspect corporate actions")
    zero_volume = panel["volume"].eq(0).groupby(level="symbol").mean()
    if (zero_volume > 0.05).any():
        report.warnings.append("one or more symbols have over 5% zero-volume observations")
    return report
