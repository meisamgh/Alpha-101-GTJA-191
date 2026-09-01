"""End-to-end local real-data research runner with a locked terminal holdout."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.backtest.engine import backtest_overlapping_cohorts
from quant_research.backtest.metrics import prediction_metrics
from quant_research.data.public import download_sp500_panel
from quant_research.features.alpha101 import compute_alpha_features
from quant_research.features.gtja191 import compute_gtja_features
from quant_research.features.regime import hard_regime_gate
from quant_research.features.registry import TRUSTED_FACTORS
from quant_research.features.technical import compute_features
from quant_research.models.train import ModelSpec, build_model
from quant_research.portfolio.construction import PortfolioConstraints, construct_weights
from quant_research.targets.returns import make_return_targets

HORIZONS = (1, 5, 10, 20)
PRIMARY_MODELS = ("ridge", "xgboost", "lightgbm")
EXTRA_MODELS_5D = ("elastic_net", "random_forest")
RESEARCH_TEST_YEARS = (2022, 2023, 2024)
HOLDOUT_START = pd.Timestamp("2025-01-01")


def build_feature_cache(panel: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    features = pd.concat(
        [compute_features(panel), compute_alpha_features(panel), compute_gtja_features(panel)],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path)
    return features


def expanding_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    label_end: pd.Series,
    model_name: str,
    test_years: tuple[int, ...],
) -> tuple[pd.Series, list[dict[str, object]]]:
    predictions = pd.Series(np.nan, index=features.index, name="prediction")
    folds: list[dict[str, object]] = []
    row_dates = features.index.get_level_values("date")
    for test_year in test_years:
        validation_year = test_year - 1
        validation_start = pd.Timestamp(f"{validation_year}-01-01")
        test_start, test_end = pd.Timestamp(f"{test_year}-01-01"), pd.Timestamp(
            f"{test_year + 1}-01-01"
        )
        train_mask = (row_dates < validation_start) & (label_end < validation_start)
        validation_mask = (row_dates >= validation_start) & (row_dates < test_start) & (
            label_end < test_start
        )
        test_mask = (row_dates >= test_start + pd.Timedelta(days=5)) & (row_dates < test_end)
        usable_columns = features.columns[features.loc[train_mask].notna().mean() >= 0.70]
        x_train, y_train = features.loc[train_mask, usable_columns], target.loc[train_mask]
        x_validation, y_validation = (
            features.loc[validation_mask, usable_columns], target.loc[validation_mask]
        )
        x_test = features.loc[test_mask, usable_columns]
        train_valid = y_train.notna()
        validation_valid = y_validation.notna()
        model = build_model(_model_spec(model_name))
        model.fit(x_train.loc[train_valid], y_train.loc[train_valid])
        # Validation is measured and reserved for threshold/model choices; never replace test.
        validation_prediction = model.predict(x_validation.loc[validation_valid])
        predictions.loc[test_mask] = model.predict(x_test)
        folds.append({
            "train_period": (
                f"{row_dates[train_mask].min().date()}.."
                f"{row_dates[train_mask].max().date()}"
            ),
            "validation_period": str(validation_year),
            "test_period": str(test_year),
            "feature_count": int(len(usable_columns)),
            "validation_rank_ic": prediction_metrics(
                pd.Series(validation_prediction, index=y_validation.loc[validation_valid].index),
                y_validation.loc[validation_valid],
            )["rank_ic"],
        })
    return predictions.dropna(), folds


def _model_spec(name: str) -> ModelSpec:
    params: dict[str, object] = {}
    if name == "xgboost":
        params = {"n_estimators": 160, "max_depth": 4, "subsample": 0.8,
                  "colsample_bytree": 0.8, "reg_lambda": 10.0}
    elif name == "lightgbm":
        params = {"n_estimators": 160, "num_leaves": 31, "subsample": 0.8,
                  "colsample_bytree": 0.8, "reg_lambda": 10.0}
    elif name == "random_forest":
        params = {"n_estimators": 120, "max_depth": 8, "max_features": 0.5}
    return ModelSpec(name=name, params=params)


def candidate_weights(
    prediction: pd.Series,
    volatility: pd.Series,
    minimum_alpha: float,
    sizing: str,
    gate: pd.Series | None = None,
) -> pd.Series:
    ranks = prediction.groupby(level="date").rank(pct=True)
    candidates = ((ranks >= 0.9) | (ranks <= 0.1)) & (prediction.abs() > minimum_alpha)
    if gate is not None:
        candidates &= gate.reindex(prediction.index).fillna(False)
    selected = prediction.where(candidates)
    vol = volatility.reindex(selected.index) if sizing == "inverse_volatility" else None
    confidence = None
    if sizing == "prediction_magnitude":
        confidence = prediction.abs().groupby(level="date").rank(pct=True)
    return construct_weights(
        selected,
        PortfolioConstraints(gross_exposure=1.0, net_exposure=0.0, max_weight=0.02),
        volatility=vol,
        confidence=confidence,
    )


def run(output_dir: Path, quick: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = Path("data/sp500_current_2014_2026.parquet")
    panel = download_sp500_panel(panel_path, "2014-01-01", "2026-09-01", 300)
    features = build_feature_cache(panel, Path("data/features_trusted_v1.parquet"))
    targets = make_return_targets(panel, HORIZONS)
    model_rows: list[dict[str, object]] = []
    strategy_rows: list[dict[str, object]] = []
    prediction_store: dict[tuple[str, int], pd.Series] = {}
    models_by_horizon = {
        h: PRIMARY_MODELS + (EXTRA_MODELS_5D if h == 5 and not quick else ()) for h in HORIZONS
    }
    if quick:
        models_by_horizon = {5: ("ridge", "lightgbm")}
    for horizon, model_names in models_by_horizon.items():
        target_name, end_name = f"vol_adjusted_return_{horizon}d", f"label_end_{horizon}d"
        target, label_end = targets[target_name], targets[end_name]
        for model_name in model_names:
            prediction, folds = expanding_predictions(
                features, target, label_end, model_name, RESEARCH_TEST_YEARS
            )
            prediction_store[(model_name, horizon)] = prediction
            metrics = prediction_metrics(prediction, target.reindex(prediction.index))
            model_rows.append({"model": model_name, "horizon": horizon,
                               "feature_count": max(int(x["feature_count"]) for x in folds),
                               "train_period": folds[-1]["train_period"],
                               "test_period": "2022-2024", **metrics,
                               "fold_rank_ic_std": float(pd.Series(
                                   [x["validation_rank_ic"] for x in folds]).std())})
            vol = features["realized_vol_20d"].reindex(prediction.index)
            gate = hard_regime_gate(features).reindex(prediction.index)
            prediction_dates = prediction.index.get_level_values("date")
            panel_dates = panel.index.get_level_values("date")
            research_panel = panel.loc[
                (panel_dates >= prediction_dates.min())
                & (panel_dates <= prediction_dates.max() + pd.Timedelta(days=40))
            ]
            for regime in ("none", "hard"):
                for sizing in ("equal", "inverse_volatility", "prediction_magnitude"):
                    weights = candidate_weights(prediction, vol, 0.05, sizing,
                                                gate if regime == "hard" else None)
                    for cost in (5, 10, 20, 50):
                        result = backtest_overlapping_cohorts(
                            research_panel, weights, horizon, cost_bps=cost, overlapping=True
                        )
                        strategy_rows.append({"model": model_name, "horizon": horizon,
                            "regime": regime, "meta_model": "none", "position_sizing": sizing,
                            "cost_bps": cost, **result.metrics})
    model_results, strategy_results = pd.DataFrame(model_rows), pd.DataFrame(strategy_rows)
    model_results.to_csv(output_dir / "model_results.csv", index=False)
    strategy_results.to_csv(output_dir / "strategy_results.csv", index=False)
    leaderboard = make_leaderboard(model_results, strategy_results)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    best = leaderboard.iloc[0]
    holdout = evaluate_holdout(panel, features, targets, best)
    final = {
        "best_research_configuration": best.to_dict(),
        "final_holdout": holdout,
        "data_disclosure": (
            "Current S&P 500 members; survivorship and membership look-ahead bias"
        ),
        "trusted_factors": [asdict(x) for x in TRUSTED_FACTORS.values()],
    }
    (output_dir / "final_model.json").write_text(json.dumps(final, indent=2, default=str))
    print_final(best, holdout)


def make_leaderboard(models: pd.DataFrame, strategies: pd.DataFrame) -> pd.DataFrame:
    merged = strategies.merge(models, on=["model", "horizon"], how="left")
    grouped = merged.groupby(["model", "horizon", "regime", "meta_model", "position_sizing"])
    rows = []
    for keys, group in grouped:
        base = group.loc[group.cost_bps == 10].iloc[0]
        cost_sensitivity = group.sharpe.max() - group.sharpe.min()
        trade_penalty = 10.0 if base.number_of_trades < 100 else 0.0
        negative_penalty = 2.0 if base.sharpe <= 0 or base.cagr <= 0 else 0.0
        robustness = (base.rank_ic * 10 + base.icir * 2 + base.sharpe
                      - abs(base.max_drawdown) - 0.1 * base.turnover
                      - 0.25 * cost_sensitivity - base.fold_rank_ic_std
                      - trade_penalty - negative_penalty)
        rows.append({**dict(zip(["model", "horizon", "regime", "meta_model",
                                "position_sizing"], keys, strict=True)),
                     **base.to_dict(), "cost_sensitivity": cost_sensitivity,
                     "robustness_score": robustness})
    return pd.DataFrame(rows).sort_values("robustness_score", ascending=False)


def evaluate_holdout(
    panel: pd.DataFrame, features: pd.DataFrame, targets: pd.DataFrame, best: pd.Series
) -> dict[str, float]:
    horizon, model_name = int(best.horizon), str(best.model)
    row_dates = features.index.get_level_values("date")
    target = targets[f"vol_adjusted_return_{horizon}d"]
    label_end = targets[f"label_end_{horizon}d"]
    train = (row_dates < pd.Timestamp("2024-01-01")) & (label_end < pd.Timestamp("2024-01-01"))
    test = row_dates >= HOLDOUT_START
    columns = features.columns[features.loc[train].notna().mean() >= 0.70]
    valid = target.loc[train].notna()
    model = build_model(_model_spec(model_name)).fit(features.loc[train, columns].loc[valid],
                                                     target.loc[train].loc[valid])
    prediction = pd.Series(model.predict(features.loc[test, columns]),
                           index=features.loc[test].index)
    metrics = prediction_metrics(prediction, target.reindex(prediction.index))
    vol = features["realized_vol_20d"].reindex(prediction.index)
    gate = hard_regime_gate(features).reindex(prediction.index) if best.regime == "hard" else None
    weights = candidate_weights(prediction, vol, 0.05, str(best.position_sizing), gate)
    holdout_panel = panel.loc[panel.index.get_level_values("date") >= HOLDOUT_START]
    result = backtest_overlapping_cohorts(holdout_panel, weights, horizon, cost_bps=10)
    return {**metrics, **{f"strategy_{k}": v for k, v in result.metrics.items()}}


def print_final(best: pd.Series, holdout: dict[str, float]) -> None:
    conclusion = "NO EDGE"
    if holdout["rank_ic"] > 0.01 and holdout["strategy_sharpe"] > 0.5:
        conclusion = "WEAK"
    if holdout["rank_ic"] > 0.02 and holdout["strategy_sharpe"] > 1.0:
        conclusion = "PROMISING"
    print("\nBEST ML TRADING MODEL")
    print("Target: continuous volatility-adjusted market/sector residual return")
    print(f"Horizon: {int(best.horizon)} days\nModel: {best.model}")
    print(f"Number of features: {int(best.feature_count)}")
    print(f"OOS Rank IC: {best.rank_ic:.4f}\nOOS ICIR: {best.icir:.4f}")
    print(f"Direction accuracy: {best.direction_accuracy:.4f}")
    print("Signal: cross-sectional tails plus absolute-alpha filter")
    print("LONG threshold: rank >= 0.90 and alpha > 0.05")
    print("SHORT threshold: rank <= 0.10 and alpha < -0.05")
    print("NO-TRADE rule: otherwise")
    print(f"Regime filter: {best.regime}\nMeta model: {best.meta_model}")
    print(f"Position sizing: {best.position_sizing}")
    print(f"Gross CAGR: {best.gross_cagr:.4f}\nNet CAGR: {best.cagr:.4f}")
    print(f"Net Sharpe: {best.sharpe:.4f}\nSortino: {best.sortino:.4f}")
    print(f"Max Drawdown: {best.max_drawdown:.4f}\nTurnover: {best.turnover:.4f}")
    print(f"Final holdout CAGR: {holdout['strategy_cagr']:.4f}")
    print(f"Final holdout Sharpe: {holdout['strategy_sharpe']:.4f}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    run(**vars(parser.parse_args()))
