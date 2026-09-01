"""Predictive cross-sectional metrics."""
import pandas as pd


def rank_ic(scores: pd.Series, target: pd.Series) -> dict[str, float]:
    joined = pd.concat({"score": scores, "target": target}, axis=1).dropna()
    daily = joined.groupby(level="date").apply(
        lambda x: x.score.corr(x.target, method="spearman"), include_groups=False
    ).dropna()
    std = daily.std()
    return {"rank_ic": float(daily.mean()), "icir": float(daily.mean() / std) if std else 0.0,
            "ic_hit_rate": float((daily > 0).mean()), "ic_observations": float(len(daily))}
