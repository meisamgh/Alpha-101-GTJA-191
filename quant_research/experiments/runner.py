"""Small staged experiment runner used for CI and real cached panel files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from quant_research.backtest.engine import backtest
from quant_research.backtest.metrics import rank_ic
from quant_research.experiments.registry import ExperimentRecord, save_record
from quant_research.experiments.synthetic import make_synthetic_panel
from quant_research.features.alpha101 import compute_alpha_features
from quant_research.features.gtja191 import compute_gtja_features
from quant_research.features.technical import compute_features
from quant_research.models.train import ModelSpec, build_model, fit_fold
from quant_research.portfolio.construction import PortfolioConstraints, construct_weights
from quant_research.targets.returns import make_return_targets
from quant_research.validation.purged_cv import PurgedWalkForwardSplit


def run(config_path: Path, smoke: bool = False) -> dict[str, float]:
    cfg = yaml.safe_load(config_path.read_text())
    panel = make_synthetic_panel(days=260, assets=20) if smoke else _load_cached(cfg)
    features = pd.concat([compute_features(panel), compute_alpha_features(panel),
                          compute_gtja_features(panel)], axis=1)
    horizon = int(cfg["target"]["horizon"])
    targets = make_return_targets(panel, (horizon,), cost_bps=cfg["costs"]["total_bps"])
    target_name = f"rank_target_{horizon}d"
    label_end = targets[f"label_end_{horizon}d"]
    usable = features.join(targets[[target_name]]).dropna()
    scores = pd.Series(index=usable.index, dtype=float)
    splitter = PurgedWalkForwardSplit(**cfg["validation"])
    for train, validation, test in splitter.split(usable.index, label_end.reindex(usable.index)):
        model = build_model(ModelSpec(cfg["model"]["name"], cfg["model"].get("params", {})))
        fit_fold(model, usable.iloc[train][features.columns], usable.iloc[train][target_name],
                 usable.iloc[validation][features.columns])
        scores.iloc[test] = model.predict(usable.iloc[test][features.columns])
    metrics = rank_ic(scores, targets[target_name].reindex(scores.index))
    constraints = PortfolioConstraints(**cfg["portfolio"])
    weights = construct_weights(scores.dropna(), constraints)
    # Backtest one-day execution returns; horizon target is for prediction/selection.
    result = backtest(panel, weights, cfg["costs"]["total_bps"])
    metrics.update(result.metrics)
    record = ExperimentRecord("smoke" if smoke else cfg["experiment_id"],
        "synthetic-ci-only" if smoke else cfg["dataset"]["version"], cfg["dataset"]["universe"],
        target_name, list(features.columns), cfg["model"]["name"], cfg["model"].get("params", {}),
        cfg["costs"], str(cfg["validation"]), metrics)
    save_record(record, Path(cfg["output_dir"]))
    return metrics


def _load_cached(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["dataset"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"No research dataset at {path}; use --smoke or configure a panel")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if not isinstance(frame.index, pd.MultiIndex):
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index(["date", "symbol"])
    return frame.sort_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run(args.config, args.smoke))
