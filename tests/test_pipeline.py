import numpy as np
import pandas as pd

from quant_research.pipeline import (
    PurgedWalkForwardSplit,
    build_alpha_target,
    cross_sectional_positions,
    evaluate_strategy,
    run_long_short_backtest,
)


def _panel(n_dates=80, n_assets=6):
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    assets = [f"A{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    x = np.arange(len(idx))
    base = 100 + 0.01 * x
    return pd.DataFrame(
        {
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * (1 + 0.001 * np.sin(x / 5)),
            "volume": 1_000_000 + x,
            "sector": [f"S{i % 2}" for _d in dates for i in range(n_assets)],
        },
        index=idx,
    )


def test_target_is_cross_sectional_and_bounded():
    p = build_alpha_target(_panel(), horizon=5)
    y = p["alpha_target"].dropna()
    assert y.between(-1, 1).all()
    counts = y.groupby(level="date").count()
    assert (counts > 1).any()


def test_purged_split_has_no_label_overlap():
    p = build_alpha_target(_panel(120), horizon=5).dropna(subset=["alpha_target", "label_end_time"])
    X = p[["close"]]
    splitter = PurgedWalkForwardSplit(n_splits=2, test_size=10, min_train_size=50, embargo=2)
    for tr, te in splitter.split(X, p["label_end_time"]):
        test_start = X.index.get_level_values("date")[te].min()
        assert (pd.to_datetime(p["label_end_time"].iloc[tr]) < pd.Timestamp(test_start)).all()


def test_positions_are_market_neutral_per_date():
    p = _panel(10)
    score = pd.Series(np.tile(np.arange(6), 10), index=p.index, dtype=float)
    pos = cross_sectional_positions(score, long_quantile=0.8, short_quantile=0.2)
    net = pos.groupby(level="date").sum()
    assert np.allclose(net.values, 0.0)


def test_backtest_metrics_run():
    p = _panel(30)
    score = pd.Series(np.tile(np.arange(6), 30), index=p.index, dtype=float)
    pos = cross_sectional_positions(score)
    daily = run_long_short_backtest(p, pos)
    metrics = evaluate_strategy(daily)
    assert "sharpe" in metrics.index
