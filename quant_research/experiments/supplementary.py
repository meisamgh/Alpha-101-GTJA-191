"""Research-period baselines and meta-label diagnostics; never touches the final holdout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

from quant_research.backtest.engine import backtest_overlapping_cohorts, performance_metrics
from quant_research.data.public import download_sp500_panel
from quant_research.experiments.local_research import (
    RESEARCH_TEST_YEARS,
    candidate_weights,
    expanding_predictions,
)
from quant_research.portfolio.construction import PortfolioConstraints, construct_weights
from quant_research.targets.returns import make_return_targets
from quant_research.targets.triple_barrier import triple_barrier_labels


def run(output_dir: Path = Path("artifacts")) -> None:
    panel = download_sp500_panel(Path("data/sp500_current_2014_2026.parquet"))
    features = pd.read_parquet("data/features_trusted_v1.parquet")
    targets = make_return_targets(panel, (5, 20))
    research_end = pd.Timestamp("2025-01-01")
    panel_research = panel.loc[panel.index.get_level_values("date") < research_end]
    prediction, _ = expanding_predictions(
        features,
        targets["vol_adjusted_return_20d"],
        targets["label_end_20d"],
        "ridge",
        RESEARCH_TEST_YEARS,
    )
    meta_rows = run_meta(panel_research, features, prediction, output_dir)
    baselines = run_baselines(panel_research, features)
    pd.DataFrame(baselines).to_csv(output_dir / "baseline_results.csv", index=False)
    strategy_path = output_dir / "strategy_results.csv"
    strategies = pd.read_csv(strategy_path)
    strategies = strategies[strategies.meta_model == "none"]
    strategies = pd.concat([strategies, pd.DataFrame(meta_rows)], ignore_index=True)
    strategies.to_csv(strategy_path, index=False)


def run_meta(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    prediction: pd.Series,
    output_dir: Path,
) -> list[dict[str, object]]:
    ranks = prediction.groupby(level="date").rank(pct=True)
    candidate = ((ranks >= 0.9) | (ranks <= 0.1)) & (prediction.abs() > 0.05)
    events = prediction.index[candidate]
    barriers = triple_barrier_labels(panel, events, horizon=20, pt=1.5, sl=1.0)
    meta = pd.DataFrame(index=barriers.index)
    meta["primary_prediction"] = prediction.reindex(meta.index)
    meta["prediction_rank"] = ranks.reindex(meta.index)
    meta["prediction_magnitude"] = meta.primary_prediction.abs()
    for column in ("adx_14", "atr_14", "bollinger_width_20", "realized_vol_20d",
                   "volume_surprise", "adv_20", "volatility_percentile_252"):
        meta[column] = features[column].reindex(meta.index)
    side = np.sign(meta.primary_prediction)
    meta["success"] = (barriers.label * side > 0).astype(int)
    meta_probability = pd.Series(np.nan, index=meta.index)
    dates = meta.index.get_level_values("date")
    columns = [column for column in meta.columns if column != "success"]
    for year in (2023, 2024):
        train = dates < pd.Timestamp(f"{year}-01-01")
        test = (dates >= pd.Timestamp(f"{year}-01-01")) & (
            dates < pd.Timestamp(f"{year + 1}-01-01")
        )
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(meta.loc[train, columns])
        x_test = imputer.transform(meta.loc[test, columns])
        classifier = LGBMClassifier(n_estimators=120, learning_rate=0.03, num_leaves=15,
                                   reg_lambda=10, random_state=42, verbosity=-1)
        classifier.fit(x_train, meta.loc[train, "success"])
        meta_probability.loc[test] = classifier.predict_proba(x_test)[:, 1]
    meta.assign(meta_probability=meta_probability).to_parquet(
        output_dir / "meta_predictions.parquet"
    )
    rows = []
    research_prediction = prediction.loc[prediction.index.get_level_values("date") >= "2023-01-01"]
    vol = features.realized_vol_20d.reindex(research_prediction.index)
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        accepted = meta_probability.reindex(research_prediction.index).fillna(0) > threshold
        filtered = research_prediction.where(accepted)
        weights = candidate_weights(filtered, vol, 0.05, "equal")
        for cost in (5, 10, 20, 50):
            result = backtest_overlapping_cohorts(panel, weights, 20, cost_bps=cost)
            rows.append({"model": "ridge", "horizon": 20, "regime": "none",
                         "meta_model": f"triple_barrier_lgbm_p>{threshold:.2f}",
                         "position_sizing": "equal", "cost_bps": cost, **result.metrics})
    return rows


def run_baselines(panel: pd.DataFrame, features: pd.DataFrame) -> list[dict[str, object]]:
    dates = panel.index.get_level_values("date")
    test_panel = panel.loc[(dates >= "2022-01-01") & (dates < "2025-01-01")]
    rows: list[dict[str, object]] = []
    market = test_panel.market_return.groupby(level="date").first().fillna(0)
    rows.append({"baseline": "SPY", **performance_metrics(market)})
    equal = pd.Series(1.0, index=test_panel.index)
    counts = equal.groupby(level="date").transform("count")
    equal /= counts
    rows.append({"baseline": "equal_weight_universe",
                 **backtest_overlapping_cohorts(test_panel, equal, 5, 10).metrics})
    for name, score in (("momentum_20d", features.momentum_20d),
                        ("mean_reversion_5d", -features.momentum_5d)):
        score = score.reindex(test_panel.index)
        weights = construct_weights(
            score,
            PortfolioConstraints(gross_exposure=1, net_exposure=0, max_weight=0.02),
        )
        rows.append({"baseline": name,
                     **backtest_overlapping_cohorts(test_panel, weights, 5, 10).metrics})
    return rows


if __name__ == "__main__":
    run()
