"""Fold-local preprocessing and tabular model factory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str = "ridge"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42


def build_model(spec: ModelSpec) -> Any:
    if spec.name == "ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("model", Ridge(**{"alpha": 10.0, **spec.params}))])
    if spec.name == "random_forest":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", RandomForestRegressor(
                             n_estimators=200, max_depth=6, n_jobs=-1,
                             random_state=spec.seed, **spec.params))])
    if spec.name in {"xgboost", "lightgbm"}:
        module, cls_name = (("xgboost", "XGBRegressor") if spec.name == "xgboost" else
                            ("lightgbm", "LGBMRegressor"))
        try:
            cls = getattr(__import__(module, fromlist=[cls_name]), cls_name)
        except ImportError as exc:
            raise ImportError(f"install the research extra to use {spec.name}") from exc
        defaults = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
                    "random_state": spec.seed, "n_jobs": 1}
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", cls(**{**defaults, **spec.params}))])
    raise ValueError(f"unknown model: {spec.name}")


def fit_fold(model: Any, x_train: Any, y_train: Any, x_validation: Any | None = None) -> Any:
    """Fit on train only. Validation is accepted for future tuning, never substituted by test."""
    del x_validation
    return model.fit(x_train, y_train)
