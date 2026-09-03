"""Train a frozen local model and emit shadow decisions without broker connectivity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.backtest.engine import backtest_overlapping_cohorts
from quant_research.backtest.metrics import prediction_metrics
from quant_research.data.public import download_sp500_panel
from quant_research.experiments.local_research import (
    RESEARCH_TEST_YEARS,
    _model_spec,
    candidate_weights,
    expanding_predictions,
)
from quant_research.models.train import build_model
from quant_research.portfolio.neutrality import neutralize_weights
from quant_research.targets.returns import make_return_targets

HORIZON = 20
THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20)


def run(output_dir: Path = Path("artifacts")) -> None:
    panel = download_sp500_panel(Path("data/sp500_current_2014_2026.parquet"))
    features = pd.read_parquet("data/features_trusted_v1.parquet")
    targets = make_return_targets(panel, (HORIZON,))
    target = targets[f"vol_adjusted_return_{HORIZON}d"]
    label_end = targets[f"label_end_{HORIZON}d"]
    research_prediction, _ = expanding_predictions(
        features, target, label_end, "ridge", RESEARCH_TEST_YEARS
    )
    threshold_results = evaluate_thresholds(
        panel, features, research_prediction, target, THRESHOLDS, "2022-01-01", "2025-01-01"
    )
    eligible = [row for row in threshold_results if row["number_of_trades"] >= 500]
    selected = max(eligible, key=lambda row: row["robustness_score"])
    diagnostic_prediction = fit_and_predict_period(
        features, target, label_end, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
    )
    diagnostic = evaluate_prediction(
        panel, features, diagnostic_prediction, target, float(selected["threshold"]),
        "2025-01-01", None,
    )
    enabled = (
        selected["sharpe_10bps"] > 0.5
        and selected["sharpe_20bps"] > 0
        and diagnostic["sharpe"] > 0
        and diagnostic["cagr"] > 0
        and diagnostic["rank_ic"] > 0.01
    )
    signals = latest_signals(panel, features, target, float(selected["threshold"]), enabled)
    signals.to_csv(output_dir / "offline_trader_signals.csv", index=False)
    result = {
        "trading_enabled": enabled,
        "status": "ELIGIBLE_FOR_SHADOW_TRADING" if enabled else "NO_TRADE_RESEARCH_GATE_FAILED",
        "selected_threshold": selected,
        "post_selection_diagnostic": diagnostic,
        "model": "ridge",
        "horizon": HORIZON,
        "execution": "signal after close t; entry open t+1; exit close t+20",
        "portfolio": "market-beta and sector neutral; gross <= 1; max name weight 2%",
        "data_warning": "current S&P 500 membership; survivorship-biased public fallback",
    }
    (output_dir / "offline_trader_results.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(threshold_results).to_csv(output_dir / "threshold_results.csv", index=False)
    print(json.dumps(result, indent=2))


def evaluate_thresholds(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    prediction: pd.Series,
    target: pd.Series,
    thresholds: tuple[float, ...],
    start: str,
    end: str,
) -> list[dict[str, float]]:
    rows = []
    for threshold in thresholds:
        metrics10 = evaluate_prediction(
            panel, features, prediction, target, threshold, start, end, 10
        )
        metrics20 = evaluate_prediction(
            panel, features, prediction, target, threshold, start, end, 20
        )
        rows.append({
            "threshold": threshold,
            "rank_ic": metrics10["rank_ic"],
            "cagr_10bps": metrics10["cagr"],
            "sharpe_10bps": metrics10["sharpe"],
            "sharpe_20bps": metrics20["sharpe"],
            "max_drawdown": metrics10["max_drawdown"],
            "turnover": metrics10["turnover"],
            "number_of_trades": metrics10["number_of_trades"],
            "robustness_score": min(metrics10["sharpe"], metrics20["sharpe"])
            - abs(metrics10["max_drawdown"]) - 0.1 * metrics10["turnover"],
        })
    return rows


def evaluate_prediction(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    prediction: pd.Series,
    target: pd.Series,
    threshold: float,
    start: str,
    end: str | None,
    cost_bps: int | None = 10,
) -> dict[str, float]:
    dates = prediction.index.get_level_values("date")
    mask = dates >= start
    if end is not None:
        mask &= dates < end
    prediction = prediction.loc[mask]
    base = candidate_weights(
        prediction, features.realized_vol_20d.reindex(prediction.index), threshold, "equal"
    )
    weights = neutralize_weights(
        base,
        features.beta_60.reindex(base.index),
        panel.sector.reindex(base.index),
    )
    panel_dates = panel.index.get_level_values("date")
    period_panel = panel.loc[panel_dates >= pd.Timestamp(start)]
    if end is not None:
        period_panel = period_panel.loc[
            period_panel.index.get_level_values("date") < pd.Timestamp(end) + pd.Timedelta(days=40)
        ]
    backtest = backtest_overlapping_cohorts(
        period_panel, weights, HORIZON, cost_bps=cost_bps or 0
    )
    return {
        **prediction_metrics(prediction, target.reindex(prediction.index)),
        **backtest.metrics,
    }


def fit_and_predict_period(
    features: pd.DataFrame,
    target: pd.Series,
    label_end: pd.Series,
    train_cutoff: pd.Timestamp,
    predict_start: pd.Timestamp,
) -> pd.Series:
    dates = features.index.get_level_values("date")
    train = (dates < train_cutoff) & (label_end < train_cutoff)
    predict = dates >= predict_start
    columns = features.columns[features.loc[train].notna().mean() >= 0.70]
    valid = target.loc[train].notna()
    model = build_model(_model_spec("ridge")).fit(
        features.loc[train, columns].loc[valid], target.loc[train].loc[valid]
    )
    return pd.Series(
        model.predict(features.loc[predict, columns]),
        index=features.loc[predict].index,
        name="prediction",
    )


def latest_signals(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
    enabled: bool,
) -> pd.DataFrame:
    known = target.notna()
    columns = features.columns[features.loc[known].notna().mean() >= 0.70]
    model = build_model(_model_spec("ridge")).fit(features.loc[known, columns], target.loc[known])
    latest_date = features.index.get_level_values("date").max()
    latest = features.loc[[latest_date], columns]
    prediction = pd.Series(model.predict(latest), index=latest.index, name="predicted_alpha")
    base = candidate_weights(
        prediction, features.realized_vol_20d.reindex(prediction.index), threshold, "equal"
    )
    weights = neutralize_weights(
        base, features.beta_60.reindex(base.index), panel.sector.reindex(base.index)
    )
    ranks = prediction.rank(pct=True)
    shadow = np.where(weights > 0, "LONG", np.where(weights < 0, "SHORT", "NO_TRADE"))
    return pd.DataFrame({
        "date": latest_date,
        "symbol": prediction.index.get_level_values("symbol"),
        "predicted_alpha": prediction.to_numpy(),
        "prediction_rank": ranks.to_numpy(),
        "shadow_decision": shadow,
        "decision": shadow if enabled else "NO_TRADE",
        "target_weight": weights.to_numpy() if enabled else 0.0,
        "research_gate": "PASS" if enabled else "FAIL",
    }).sort_values("predicted_alpha", ascending=False)


if __name__ == "__main__":
    run()
