"""Deterministic synthetic panel for CI mechanics; never evidence of tradable alpha."""
import numpy as np
import pandas as pd


def make_synthetic_panel(days: int = 320, assets: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=days)
    symbols = [f"S{i:03d}" for i in range(assets)]
    market = rng.normal(0, 0.008, days)
    rows = []
    for i, symbol in enumerate(symbols):
        returns = 0.8 * market + rng.normal(0, 0.012, days)
        close = 50 * np.cumprod(1 + returns)
        overnight = rng.normal(0, 0.003, days)
        open_ = close / (1 + rng.normal(0, 0.006, days)) * (1 + overnight)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, days))
        for j, date in enumerate(dates):
            rows.append((date, symbol, open_[j], high[j], low[j], close[j], close[j],
                         int(rng.lognormal(13, 0.5)), f"sector_{i % 5}", market[j]))
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close",
        "adjusted_close", "volume", "sector", "market_return"]).set_index(
            ["date", "symbol"]
        ).sort_index()
