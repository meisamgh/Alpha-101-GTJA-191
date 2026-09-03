import numpy as np

from quant_research.targets.ranking import cross_sectional_rank
from quant_research.targets.returns import make_return_targets
from quant_research.targets.triple_barrier import triple_barrier_labels


def test_next_open_target_alignment(panel):
    targets = make_return_targets(panel, (5,))
    symbol = panel.index.get_level_values("symbol")[0]
    asset = panel.xs(symbol, level="symbol")
    date = asset.index[50]
    expected = asset.close.iloc[55] / asset.open.iloc[51] - 1
    assert np.isclose(targets.loc[(date, symbol), "raw_return_5d"], expected)


def test_cross_sectional_rank_range(panel):
    target = make_return_targets(panel, (5,))["vol_adjusted_return_5d"]
    ranks = cross_sectional_rank(target).dropna()
    assert ranks.between(0, 1).all()


def test_continuous_target_is_not_modified_by_cost_argument(panel):
    gross = make_return_targets(panel, (5,), cost_bps=0)["vol_adjusted_return_5d"]
    costly = make_return_targets(panel, (5,), cost_bps=50)["vol_adjusted_return_5d"]
    assert gross.equals(costly)


def test_triple_barrier_is_bounded_by_vertical_window(panel):
    events = panel.index[80:100:5]
    labels = triple_barrier_labels(panel, events, horizon=5, volatility_window=20)
    for (date, symbol), row in labels.iterrows():
        dates = panel.xs(symbol, level="symbol").index
        assert date < row.label_end <= dates[dates.get_loc(date) + 5]
