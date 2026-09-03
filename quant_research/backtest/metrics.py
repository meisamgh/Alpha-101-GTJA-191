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


def prediction_metrics(scores: pd.Series, target: pd.Series) -> dict[str, float]:
    joined = pd.concat({"score": scores, "target": target}, axis=1).dropna()
    by_date = joined.groupby(level="date")
    spearman = by_date.apply(lambda x: x.score.corr(x.target, method="spearman"))
    pearson = by_date.apply(lambda x: x.score.corr(x.target, method="pearson"))
    top = joined.score.groupby(level="date").rank(pct=True) >= 0.9
    bottom = joined.score.groupby(level="date").rank(pct=True) <= 0.1
    spread = by_date.apply(
        lambda x: x.loc[x.score.rank(pct=True) >= 0.9, "target"].mean()
        - x.loc[x.score.rank(pct=True) <= 0.1, "target"].mean()
    )
    return {
        "rank_ic": float(spearman.mean()),
        "pearson_ic": float(pearson.mean()),
        "icir": float(spearman.mean() / spearman.std()) if spearman.std() else 0.0,
        "direction_accuracy": float((joined.score * joined.target > 0).mean()),
        "prediction_target_correlation": float(joined.score.corr(joined.target)),
        "top_decile_alpha": float(joined.loc[top, "target"].mean()),
        "bottom_decile_alpha": float(joined.loc[bottom, "target"].mean()),
        "top_bottom_spread": float(spread.mean()),
    }
