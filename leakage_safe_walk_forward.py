"""Leakage-safe walk-forward research baseline for OHLCV strategies.

The key design rule is simple: anything that learns parameters is fit on the
training window only. The test observation is never used for early stopping,
feature fitting, or model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


TRADING_DAYS = 252


@dataclass(frozen=True)
class ResearchConfig:
    symbols: tuple[str, ...] = ("AAPL",)
    start: str = "2015-01-01"
    end: str | None = None
    min_train: int = 756          # ~3 years of daily observations
    validation_size: int = 126    # ~6 months
    test_size: int = 21           # ~1 month per walk-forward fold
    step_size: int = 21
    transaction_cost_bps: float = 10.0
    prediction_threshold: float = 0.0
    random_state: int = 42


def download_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily OHLCV data for one symbol."""
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol!r}.")

    # yfinance may return a 2-level column index even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns for {symbol}: {missing}")

    out = df[wanted].copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna()


def build_causal_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Create features available after the current day's close.

    No feature uses negative shifts or centered windows. Therefore a feature at
    date t depends only on observations <= t.
    """
    df = ohlcv.copy()
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].replace(0, np.nan)

    log_close = np.log(close)
    ret_1 = log_close.diff()

    feat = pd.DataFrame(index=df.index)
    feat["ret_1d"] = ret_1
    feat["ret_2d"] = log_close.diff(2)
    feat["ret_5d"] = log_close.diff(5)
    feat["ret_10d"] = log_close.diff(10)
    feat["ret_20d"] = log_close.diff(20)
    feat["intraday_ret"] = np.log(close / open_)
    feat["overnight_ret"] = np.log(open_ / close.shift(1))
    feat["high_low_range"] = np.log(high / low)
    feat["close_open_range"] = (close - open_) / open_

    for window in (5, 10, 20, 60):
        feat[f"vol_{window}d"] = ret_1.rolling(window).std()
        feat[f"momentum_{window}d"] = log_close - log_close.shift(window)
        feat[f"price_z_{window}d"] = (
            (close - close.rolling(window).mean()) / close.rolling(window).std()
        )
        feat[f"volume_z_{window}d"] = (
            (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
        )

    # Causal volatility-regime proxies; unlike the original HMM/KMeans/Fuzzy
    # features these require no fitting on the full sample.
    vol20 = ret_1.rolling(20).std()
    vol60 = ret_1.rolling(60).std()
    feat["vol_regime_ratio"] = vol20 / vol60
    feat["trend_strength"] = feat["momentum_20d"] / (vol20 * np.sqrt(20))

    # Calendar effects are known in advance.
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month

    return feat.replace([np.inf, -np.inf], np.nan)


def make_supervised_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Align date-t features with the next session's open-to-close return.

    Signals are assumed to be formed after close(t), entered at open(t+1), and
    exited at close(t+1). This avoids pretending we can trade on the same close
    used to calculate the signal.
    """
    features = build_causal_features(ohlcv)
    target = np.log(ohlcv["Close"].shift(-1) / ohlcv["Open"].shift(-1))
    next_open = ohlcv["Open"].shift(-1)
    next_close = ohlcv["Close"].shift(-1)

    frame = features.copy()
    frame["target"] = target
    frame["next_open"] = next_open
    frame["next_close"] = next_close
    return frame.dropna().copy()


def walk_forward_slices(
    n_obs: int,
    min_train: int,
    validation_size: int,
    test_size: int,
    step_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield expanding train, validation, and untouched test indices."""
    first_test = min_train + validation_size
    for test_start in range(first_test, n_obs - test_size + 1, step_size):
        val_start = test_start - validation_size
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, test_start)
        test_idx = np.arange(test_start, test_start + test_size)
        yield train_idx, val_idx, test_idx


def make_model(config: ResearchConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
        early_stopping_rounds=100,
    )


def walk_forward_predict(frame: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    feature_cols = [
        c for c in frame.columns if c not in {"target", "next_open", "next_close"}
    ]
    rows: list[pd.DataFrame] = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(
        walk_forward_slices(
            len(frame),
            config.min_train,
            config.validation_size,
            config.test_size,
            config.step_size,
        ),
        start=1,
    ):
        train = frame.iloc[train_idx]
        val = frame.iloc[val_idx]
        test = frame.iloc[test_idx]

        model = make_model(config)
        model.fit(
            train[feature_cols],
            train["target"],
            eval_set=[(val[feature_cols], val["target"])],
            verbose=False,
        )

        pred = model.predict(test[feature_cols])
        fold_result = pd.DataFrame(
            {
                "actual": test["target"],
                "prediction": pred,
                "next_open": test["next_open"],
                "next_close": test["next_close"],
                "fold": fold,
            },
            index=test.index,
        )
        rows.append(fold_result)

    if not rows:
        raise ValueError(
            "Not enough observations for the requested train/validation/test windows."
        )
    return pd.concat(rows).sort_index()


def max_drawdown(returns: pd.Series) -> float:
    equity = np.exp(returns.cumsum())
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def evaluate_predictions(
    predictions: pd.DataFrame, config: ResearchConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    out = predictions.copy()
    out["position"] = np.where(
        out["prediction"] > config.prediction_threshold,
        1.0,
        np.where(out["prediction"] < -config.prediction_threshold, -1.0, 0.0),
    )

    # Each row predicts the next session. Costs are charged when exposure changes.
    turnover = out["position"].diff().abs().fillna(out["position"].abs())
    cost_per_unit = config.transaction_cost_bps / 10_000.0
    out["cost"] = turnover * cost_per_unit
    out["strategy_log_return"] = out["position"] * out["actual"] - out["cost"]
    out["benchmark_log_return"] = out["actual"]

    ann_mean = out["strategy_log_return"].mean() * TRADING_DAYS
    ann_vol = out["strategy_log_return"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan

    metrics = {
        "n_predictions": float(len(out)),
        "rmse": float(np.sqrt(mean_squared_error(out["actual"], out["prediction"]))),
        "mae": float(mean_absolute_error(out["actual"], out["prediction"])),
        "directional_accuracy": float(
            (np.sign(out["actual"]) == np.sign(out["prediction"])).mean()
        ),
        "annualized_return": float(np.exp(ann_mean) - 1.0),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(out["strategy_log_return"]),
        "average_daily_turnover": float(turnover.mean()),
        "total_return": float(np.exp(out["strategy_log_return"].sum()) - 1.0),
    }
    return metrics, out


def run_symbol(symbol: str, config: ResearchConfig) -> tuple[dict[str, float], pd.DataFrame]:
    ohlcv = download_ohlcv(symbol, config.start, config.end)
    frame = make_supervised_frame(ohlcv)
    predictions = walk_forward_predict(frame, config)
    metrics, details = evaluate_predictions(predictions, config)
    metrics["symbol"] = symbol
    return metrics, details


def main() -> None:
    config = ResearchConfig()
    summaries = []
    for symbol in config.symbols:
        metrics, details = run_symbol(symbol, config)
        summaries.append(metrics)
        details.to_csv(f"walk_forward_predictions_{symbol}.csv")

    summary = pd.DataFrame(summaries).set_index("symbol")
    pd.set_option("display.max_columns", None)
    print(summary.round(4))


if __name__ == "__main__":
    main()
