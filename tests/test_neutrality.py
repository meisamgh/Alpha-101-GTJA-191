import numpy as np
import pandas as pd

from quant_research.portfolio.neutrality import neutralize_weights


def test_neutral_projection_controls_net_beta_and_sector_exposure():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], [f"S{i}" for i in range(12)]],
        names=["date", "symbol"],
    )
    weights = pd.Series(np.linspace(-0.04, 0.04, 12), index=index)
    beta = pd.Series(np.linspace(0.5, 1.5, 12), index=index)
    sector = pd.Series(["A"] * 4 + ["B"] * 4 + ["C"] * 4, index=index)
    neutral = neutralize_weights(weights, beta, sector)
    assert neutral.abs().max() <= 0.02 + 1e-12
    assert neutral.abs().sum() <= 1.0 + 1e-12
    assert abs(neutral.sum()) < 1e-12
    assert abs((neutral * beta).sum()) < 1e-12
    sector_exposure = neutral.groupby(sector).sum()
    assert sector_exposure.abs().max() < 1e-12
