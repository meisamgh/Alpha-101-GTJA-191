"""Minimal example of the modern research path.

Replace ``make_demo_panel`` and the placeholder feature set with your real
multi-asset OHLCV plus Alpha-101 / GTJA-191 features.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from quant_research import (
    PurgedWalkForwardSplit,
    add_regime_features,
    build_alpha_target,
    cross_sectional_positions,
    evaluate_strategy,
    run_long_short_backtest,
)


def make_demo_panel(n_dates=700, n_assets=30, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"S{i:03d}" for i in range(n_assets)]

    market = rng.normal(0.0002, 0.01, n_dates)
    rows = []
    for j, asset in enumerate(assets):
        idio = rng.normal(0.0, 0.012 + j / 20000, n_dates)
        ret = 0.6 * market + idio
        close = 100 * np.cumprod(1 + ret)
        overnight = rng.normal(0.0, 0.002, n_dates)
        open_ = close / (1 + ret) * (1 + overnight)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_dates))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_dates))
        volume = rng.lognormal(14, 0.4, n_dates)
        for i, d in enumerate(dates):
            rows.append((d, asset, open_[i], high[i], low[i], close[i], volume[i], f"sector_{j % 6}"))

    return pd.DataFrame(
        rows,
        columns=["date", "asset", "open", "high", "low", "close", "volume", "sector"],
    ).set_index(["date", "asset"])


def add_demo_alpha_features(df):
    # Placeholders: replace/add actual Alpha-101 and GTJA columns here.
    g = df.groupby(level="asset", group_keys=False)
    df["mom_5"] = g["close"].pct_change(5)
    df["mom_20"] = g["close"].pct_change(20)
    df["vol_20"] = g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    df["volume_z"] = (
        (df["volume"] - g["volume"].rolling(20).mean().reset_index(level=0, drop=True))
        / g["volume"].rolling(20).std().reset_index(level=0, drop=True)
    )
    return df


def main():
    panel = make_demo_panel()
    panel = add_regime_features(panel)
    panel = add_demo_alpha_features(panel)
    panel = build_alpha_target(panel, horizon=5, sector_col="sector", round_trip_cost_bps=10)

    feature_cols = [
        "mom_5",
        "mom_20",
        "vol_20",
        "volume_z",
        "adx",
        "atr_pct",
        "bb_width",
        "trend_strength",
    ]

    research = panel.dropna(subset=feature_cols + ["alpha_target", "label_end_time"]).copy()
    X = research[feature_cols]
    y = research["alpha_target"]

    splitter = PurgedWalkForwardSplit(n_splits=5, test_size=40, min_train_size=300, embargo=5)
    pred = pd.Series(np.nan, index=research.index, name="score")

    for train_idx, test_idx in splitter.split(X, research["label_end_time"]):
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[test_idx] = model.predict(X.iloc[test_idx])

    valid = pred.notna()
    ic_by_date = pd.DataFrame({"pred": pred[valid], "target": y[valid]}).groupby(level="date").apply(
        lambda z: spearmanr(z["pred"], z["target"]).statistic if len(z) > 2 else np.nan
    )
    print(f"Mean daily Rank IC: {ic_by_date.mean():.4f}")

    positions = cross_sectional_positions(
        pred,
        tradable=research["tradable_regime"],
        long_quantile=0.85,
        short_quantile=0.15,
    )
    daily = run_long_short_backtest(panel, positions, cost_bps=5)
    print(evaluate_strategy(daily).round(4))


if __name__ == "__main__":
    main()
