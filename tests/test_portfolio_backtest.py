import numpy as np
import pandas as pd

from quant_research.backtest.engine import backtest
from quant_research.portfolio.construction import PortfolioConstraints, construct_weights


def test_weights_obey_caps_and_market_neutrality(panel):
    scores = pd.Series(np.arange(len(panel), dtype=float), index=panel.index)
    constraints = PortfolioConstraints(max_weight=0.10)
    weights = construct_weights(scores, constraints)
    assert weights.abs().max() <= 0.10 + 1e-12
    daily_net = weights.groupby(level="date").sum()
    assert daily_net.abs().max() < 1e-12


def test_costs_reduce_pnl(panel):
    scores = pd.Series(np.sin(np.arange(len(panel))), index=panel.index)
    weights = construct_weights(scores, PortfolioConstraints(max_weight=0.10))
    gross = backtest(panel, weights, cost_bps=0).daily.net_return.sum()
    net = backtest(panel, weights, cost_bps=20).daily.net_return.sum()
    assert net <= gross


def test_signal_executes_at_next_open(panel):
    scores = pd.Series(np.arange(len(panel), dtype=float), index=panel.index)
    weights = construct_weights(scores, PortfolioConstraints(max_weight=0.10))
    result = backtest(panel, weights, cost_bps=0)
    next_intraday = (panel.close / panel.open - 1).groupby(level="symbol").shift(-1)
    expected = (weights * next_intraday).groupby(level="date").sum(min_count=1).fillna(0)
    pd.testing.assert_series_equal(result.daily.gross_return, expected, check_names=False)
