"""Modern research pipeline for Alpha-101 / GTJA-191.

Design principles
-----------------
1. Build all predictors from information available at or before signal time.
2. Predict a *cross-sectional*, risk-adjusted future residual return rather
   than an absolute future price.
3. Use event/triple-barrier labels as a second-stage (meta-label) filter.
4. Use purged walk-forward validation when labels overlap in time.
5. Backtest with next-session execution and explicit turnover costs.

Expected panel format
---------------------
A pandas DataFrame indexed by ``[date, asset]`` with lowercase OHLCV columns:
``open, high, low, close, volume``. Optional columns include ``sector`` and a
benchmark-return column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must have a two-level MultiIndex: [date, asset]")
    missing = REQUIRED_OHLCV.difference(panel.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    out = panel.sort_index().copy()
    out.index = out.index.set_names(["date", "asset"])
    return out


def _by_asset(series: pd.Series):
    return series.groupby(level="asset", group_keys=False)


def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _asset_adx(group: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = group["high"], group["low"], group["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=group.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=group.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = _wilder_ema(tr, period)
    plus_di = 100.0 * _wilder_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder_ema(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_ema(dx, period)


def add_regime_features(
    panel: pd.DataFrame,
    vol_window: int = 20,
    trend_window: int = 20,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Add strictly trailing regime/range features.

    ``tradable_regime`` is deliberately conservative: a stock is considered
    tradable when trend strength is meaningful *and* volatility is not in the
    lowest part of its own recent history. It is a gate, not a future label.
    """
    out = _validate_panel(panel)

    out["ret_1d"] = _by_asset(out["close"]).pct_change()
    out["realized_vol"] = _by_asset(out["ret_1d"]).rolling(vol_window).std().reset_index(level=0, drop=True)

    prev_close = _by_asset(out["close"]).shift(1)
    tr = pd.concat(
        [(out["high"] - out["low"]).abs(), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.groupby(level="asset", group_keys=False).rolling(adx_period).mean().reset_index(level=0, drop=True)
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)

    mid = _by_asset(out["close"]).rolling(trend_window).mean().reset_index(level=0, drop=True)
    std = _by_asset(out["close"]).rolling(trend_window).std().reset_index(level=0, drop=True)
    out["bb_width"] = (4.0 * std) / mid.replace(0, np.nan)

    lagged = _by_asset(out["close"]).shift(trend_window)
    out["trend_strength"] = (out["close"] / lagged - 1.0).abs()

    adx_parts = []
    for _, grp in out.groupby(level="asset", sort=False):
        adx_parts.append(_asset_adx(grp, adx_period))
    out["adx"] = pd.concat(adx_parts).sort_index()

    def trailing_pct_rank(s: pd.Series, window: int = 126) -> pd.Series:
        return s.rolling(window, min_periods=max(20, window // 4)).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    out["atr_pctile"] = _by_asset(out["atr_pct"]).apply(trailing_pct_rank)
    out["bb_width_pctile"] = _by_asset(out["bb_width"]).apply(trailing_pct_rank)

    trend_vote = out["adx"] >= 20.0
    movement_vote = out["trend_strength"] >= 0.02
    volatility_ok = (out["atr_pctile"] >= 0.25) | (out["bb_width_pctile"] >= 0.25)
    out["tradable_regime"] = (trend_vote | movement_vote) & volatility_ok
    return out


def build_alpha_target(
    panel: pd.DataFrame,
    horizon: int = 5,
    vol_window: int = 20,
    benchmark_return_col: Optional[str] = None,
    sector_col: Optional[str] = "sector",
    round_trip_cost_bps: float = 10.0,
    centered_rank: bool = True,
) -> pd.DataFrame:
    """Create the primary target: net residual risk-adjusted future-return rank.

    Signal is assumed to be produced after close on date ``t``. Entry therefore
    occurs at ``open[t+1]`` and the horizon exit at ``close[t+horizon]``.
    """
    out = _validate_panel(panel)
    by_asset = out.groupby(level="asset", group_keys=False)

    entry = by_asset["open"].shift(-1)
    exit_ = by_asset["close"].shift(-horizon)
    out["future_return"] = exit_ / entry - 1.0

    residual = out["future_return"].copy()
    if benchmark_return_col and benchmark_return_col in out.columns:
        residual = residual - out[benchmark_return_col]

    if sector_col and sector_col in out.columns:
        sector_mean = out.assign(_future=residual).groupby(
            [out.index.get_level_values("date"), out[sector_col]], dropna=False
        )["_future"].transform("mean")
        residual = residual - sector_mean
    else:
        date_mean = residual.groupby(level="date").transform("mean")
        residual = residual - date_mean

    past_ret = by_asset["close"].pct_change()
    ex_ante_vol = past_ret.groupby(level="asset", group_keys=False).rolling(vol_window).std().reset_index(level=0, drop=True)
    ex_ante_vol = ex_ante_vol.clip(lower=1e-6)

    costs = round_trip_cost_bps / 10_000.0
    net_residual = residual - np.sign(residual).replace(0, 1) * costs
    out["future_residual_return"] = residual
    out["risk_adjusted_target_raw"] = net_residual / ex_ante_vol

    rank = out["risk_adjusted_target_raw"].groupby(level="date").rank(pct=True)
    out["alpha_target"] = 2.0 * rank - 1.0 if centered_rank else rank
    out["label_end_time"] = pd.Series(
        out.index.get_level_values("date"), index=out.index
    ).groupby(level="asset", group_keys=False).shift(-horizon)
    return out


def cusum_events(returns: pd.Series, threshold: pd.Series | float) -> pd.Series:
    """Symmetric CUSUM event filter for one asset's return series."""
    ret = returns.fillna(0.0)
    if np.isscalar(threshold):
        thr = pd.Series(float(threshold), index=ret.index)
    else:
        thr = threshold.reindex(ret.index).ffill()

    pos, neg = 0.0, 0.0
    events = pd.Series(False, index=ret.index)
    for idx, value in ret.items():
        h = thr.loc[idx]
        if not np.isfinite(h) or h <= 0:
            continue
        pos = max(0.0, pos + value)
        neg = min(0.0, neg + value)
        if pos > h or neg < -h:
            events.loc[idx] = True
            pos, neg = 0.0, 0.0
    return events


def build_meta_labels(
    panel: pd.DataFrame,
    side: pd.Series,
    horizon: int = 5,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
    vol_col: str = "realized_vol",
    use_cusum: bool = True,
    cusum_mult: float = 0.5,
) -> pd.DataFrame:
    """Triple-barrier meta-labels for an existing primary-model side."""
    out = _validate_panel(panel)
    side = side.reindex(out.index).fillna(0.0).clip(-1, 1)
    rows = []

    for asset, grp in out.groupby(level="asset", sort=False):
        g = grp.droplevel("asset")
        s = side.xs(asset, level="asset").reindex(g.index).fillna(0.0)
        if vol_col not in g.columns:
            vol = g["close"].pct_change().rolling(20).std()
        else:
            vol = g[vol_col]

        candidate = s.ne(0)
        if use_cusum:
            threshold = (vol * cusum_mult).clip(lower=1e-5)
            candidate &= cusum_events(g["close"].pct_change(), threshold)

        event_dates = g.index[candidate.fillna(False)]
        for t0 in event_dates:
            loc = g.index.get_loc(t0)
            entry_loc = loc + 1
            if entry_loc >= len(g):
                continue
            end_loc = min(loc + horizon, len(g) - 1)
            if end_loc < entry_loc:
                continue

            entry_price = float(g["open"].iloc[entry_loc])
            sigma = float(vol.loc[t0]) if np.isfinite(vol.loc[t0]) else np.nan
            if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(sigma) or sigma <= 0:
                continue

            trade_side = int(np.sign(s.loc[t0]))
            path = g["close"].iloc[entry_loc : end_loc + 1] / entry_price - 1.0
            signed_path = trade_side * path
            pt = pt_mult * sigma
            sl = -sl_mult * sigma

            pt_hits = signed_path[signed_path >= pt]
            sl_hits = signed_path[signed_path <= sl]
            pt_time = pt_hits.index[0] if not pt_hits.empty else None
            sl_time = sl_hits.index[0] if not sl_hits.empty else None

            if pt_time is not None and (sl_time is None or pt_time <= sl_time):
                label, end_time, realized = 1, pt_time, float(signed_path.loc[pt_time])
            elif sl_time is not None:
                label, end_time, realized = 0, sl_time, float(signed_path.loc[sl_time])
            else:
                end_time = signed_path.index[-1]
                realized = float(signed_path.iloc[-1])
                label = int(realized > 0)

            rows.append(
                {
                    "date": t0,
                    "asset": asset,
                    "side": trade_side,
                    "meta_label": label,
                    "event_end_time": end_time,
                    "signed_event_return": realized,
                    "target_vol": sigma,
                }
            )

    if not rows:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        return pd.DataFrame(
            columns=["side", "meta_label", "event_end_time", "signed_event_return", "target_vol"],
            index=empty_index,
        )
    return pd.DataFrame(rows).set_index(["date", "asset"]).sort_index()


@dataclass
class PurgedWalkForwardSplit:
    """Expanding walk-forward splitter with overlapping-label purging."""

    n_splits: int = 5
    test_size: int = 20
    min_train_size: int = 252
    embargo: int = 0

    def split(
        self,
        X: pd.DataFrame,
        label_end_times: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must use [date, asset] MultiIndex")
        dates = pd.Index(X.index.get_level_values("date").unique()).sort_values()
        needed = self.min_train_size + self.n_splits * self.test_size
        if len(dates) < needed:
            raise ValueError(f"need at least {needed} unique dates, got {len(dates)}")

        start = len(dates) - self.n_splits * self.test_size
        row_dates = pd.Series(X.index.get_level_values("date"), index=np.arange(len(X)))
        label_end = label_end_times.reindex(X.index) if label_end_times is not None else None

        for fold in range(self.n_splits):
            test_start_i = start + fold * self.test_size
            test_end_i = test_start_i + self.test_size
            test_dates = dates[test_start_i:test_end_i]
            test_start = test_dates[0]

            train_date_limit = test_start_i - self.embargo
            train_dates = dates[: max(0, train_date_limit)]
            if len(train_dates) < self.min_train_size:
                continue

            train_mask = row_dates.isin(train_dates).to_numpy()
            test_mask = row_dates.isin(test_dates).to_numpy()

            if label_end is not None:
                ends = pd.to_datetime(label_end.to_numpy())
                valid_end = pd.notna(ends)
                overlap = valid_end & (ends >= pd.Timestamp(test_start))
                train_mask &= ~overlap

            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def cross_sectional_positions(
    score: pd.Series,
    tradable: Optional[pd.Series] = None,
    long_quantile: float = 0.85,
    short_quantile: float = 0.15,
    gross_exposure: float = 1.0,
) -> pd.Series:
    """Turn model scores into equal-weight market-neutral positions."""
    if not isinstance(score.index, pd.MultiIndex):
        raise ValueError("score must use [date, asset] MultiIndex")
    eligible = pd.Series(True, index=score.index) if tradable is None else tradable.reindex(score.index).fillna(False)

    positions = pd.Series(0.0, index=score.index, name="position")
    for _, s in score.groupby(level="date"):
        s = s[eligible.reindex(s.index).fillna(False)].dropna()
        if len(s) < 4:
            continue
        lo, hi = s.quantile(short_quantile), s.quantile(long_quantile)
        longs = s[s >= hi].index
        shorts = s[s <= lo].index
        if len(longs):
            positions.loc[longs] = (gross_exposure / 2.0) / len(longs)
        if len(shorts):
            positions.loc[shorts] = -(gross_exposure / 2.0) / len(shorts)
    return positions


def run_long_short_backtest(
    panel: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Backtest signals generated at close[t], executed open[t+1]."""
    out = _validate_panel(panel)
    pos = positions.reindex(out.index).fillna(0.0)
    by_asset = out.groupby(level="asset", group_keys=False)
    next_open = by_asset["open"].shift(-1)
    following_open = by_asset["open"].shift(-2)
    exec_return = following_open / next_open - 1.0

    gross_contrib = pos * exec_return
    prev_pos = pos.groupby(level="asset", group_keys=False).shift(1).fillna(0.0)
    turnover = (pos - prev_pos).abs()
    cost = turnover * (cost_bps / 10_000.0)

    daily = pd.DataFrame(
        {
            "gross_return": gross_contrib.groupby(level="date").sum(min_count=1),
            "turnover": turnover.groupby(level="date").sum(),
            "cost": cost.groupby(level="date").sum(),
        }
    )
    daily["net_return"] = daily["gross_return"] - daily["cost"]
    daily["equity"] = (1.0 + daily["net_return"].fillna(0.0)).cumprod()
    return daily


def evaluate_strategy(daily: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Return compact net-of-cost portfolio metrics."""
    r = daily["net_return"].dropna()
    if r.empty:
        return pd.Series(dtype=float)
    ann_return = (1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = (r.mean() / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else np.nan
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.Series(
        {
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": drawdown.min(),
            "avg_daily_turnover": daily.loc[r.index, "turnover"].mean(),
            "total_cost": daily.loc[r.index, "cost"].sum(),
        }
    )
