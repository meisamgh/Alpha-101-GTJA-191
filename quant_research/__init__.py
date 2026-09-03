"""Leakage-aware quantitative research utilities for Alpha-101 / GTJA-191."""

from .pipeline import (
    PurgedWalkForwardSplit,
    add_regime_features,
    build_alpha_target,
    build_meta_labels,
    cross_sectional_positions,
    evaluate_strategy,
    run_long_short_backtest,
)

__all__ = [
    "PurgedWalkForwardSplit",
    "add_regime_features",
    "build_alpha_target",
    "build_meta_labels",
    "cross_sectional_positions",
    "evaluate_strategy",
    "run_long_short_backtest",
]
