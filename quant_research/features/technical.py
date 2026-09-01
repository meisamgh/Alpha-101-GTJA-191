"""Trailing-only panel features available after close on date t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grolling(s: pd.Series, window: int, method: str) -> pd.Series:
    grouped = s.groupby(level="symbol", group_keys=False)
    return grouped.transform(lambda x: getattr(x.rolling(window, min_periods=window), method)())


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_index()
    by_symbol = p.groupby(level="symbol", group_keys=False)
    close = p["adjusted_close"]
    ret1 = by_symbol["adjusted_close"].pct_change(fill_method=None)
    out = pd.DataFrame(index=p.index)
    for horizon in (1, 2, 5, 10, 20, 60):
        out[f"momentum_{horizon}d"] = by_symbol["adjusted_close"].pct_change(
            horizon, fill_method=None
        )
    for window in (5, 10, 20, 60):
        out[f"realized_vol_{window}d"] = ret1.groupby(level="symbol").transform(
            lambda x, w=window: x.rolling(w, min_periods=w).std()
        ) * np.sqrt(252)
    prev_close = by_symbol["close"].shift(1)
    tr = pd.concat([(p.high - p.low), (p.high - prev_close).abs(),
                    (p.low - prev_close).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.groupby(level="symbol").transform(
        lambda x: x.rolling(14, min_periods=14).mean()
    ) / close
    up_move = p.high.groupby(level="symbol").diff()
    down_move = -p.low.groupby(level="symbol").diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_raw = tr.groupby(level="symbol").transform(lambda x: x.rolling(14).mean())
    plus_di = 100 * plus_dm.groupby(level="symbol").transform(
        lambda x: x.rolling(14).mean()
    ) / atr_raw
    minus_di = 100 * minus_dm.groupby(level="symbol").transform(
        lambda x: x.rolling(14).mean()
    ) / atr_raw
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx_14"] = dx.groupby(level="symbol").transform(lambda x: x.rolling(14).mean())
    log_hl = np.log(p.high / p.low).pow(2)
    out["parkinson_vol_20d"] = np.sqrt(_grolling(log_hl, 20, "mean") / (4 * np.log(2)))
    out["downside_vol_20d"] = ret1.where(ret1 < 0, 0).groupby(level="symbol").transform(
        lambda x: x.rolling(20, min_periods=20).std()
    ) * np.sqrt(252)
    dollar_volume = p.close * p.volume
    out["dollar_volume"] = dollar_volume
    out["adv_20"] = _grolling(dollar_volume, 20, "mean")
    out["volume_surprise"] = p.volume / _grolling(p.volume, 20, "mean") - 1
    out["amihud_20"] = (ret1.abs() / dollar_volume.replace(0, np.nan)).groupby(
        level="symbol"
    ).transform(lambda x: x.rolling(20, min_periods=20).mean())
    out["turnover_proxy_20"] = p.volume / _grolling(p.volume, 20, "mean")
    for window in (10, 20, 60):
        ma = _grolling(close, window, "mean")
        out[f"distance_ma_{window}"] = close / ma - 1
        out[f"ma_slope_{window}"] = ma.groupby(level="symbol").pct_change(5, fill_method=None)
    mean20, std20 = _grolling(close, 20, "mean"), _grolling(close, 20, "std")
    out["bollinger_width_20"] = 4 * std20 / mean20
    high20 = _grolling(p.high, 20, "max")
    low20 = _grolling(p.low, 20, "min")
    out["rolling_range_20"] = high20 / low20 - 1
    out["breakout_strength_20"] = (close - low20) / (high20 - low20).replace(0, np.nan)
    out["trend_strength_20"] = out["ma_slope_20"].abs() / out["realized_vol_20d"]
    out["volatility_percentile_252"] = out["realized_vol_20d"].groupby(
        level="symbol"
    ).transform(lambda x: x.rolling(252, min_periods=60).rank(pct=True))
    if "market_return" in p:
        market = p["market_return"]
        cov = ret1.groupby(level="symbol").transform(
            lambda x: x.rolling(60, min_periods=40).cov(market.loc[x.index])
        )
        var = market.groupby(level="symbol").transform(
            lambda x: x.rolling(60, min_periods=40).var()
        )
        out["beta_60"] = cov / var.replace(0, np.nan)
        out["market_relative_momentum_20"] = out["momentum_20d"] - market.groupby(
            level="symbol"
        ).transform(lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1)
        out["residual_momentum_20"] = out["momentum_20d"] - out["beta_60"] * market.groupby(
            level="symbol"
        ).transform(lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1)
    if "sector" in p:
        sector_momentum = out["momentum_20d"].groupby(
            [p.index.get_level_values("date"), p.sector]
        ).transform("mean")
        out["sector_relative_momentum_20"] = out["momentum_20d"] - sector_momentum
    return out.replace([np.inf, -np.inf], np.nan)
