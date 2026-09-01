# Local real-data model report

## Result

The complete chain ran locally against 955,198 daily observations for 300 current S&P 500 members
from 2014-01-02 through 2026-08-31. Research selection used expanding 2022, 2023, and 2024 OOS test
folds. Calendar years 2025 through 2026-08-31 were reserved for the terminal holdout.

**Conclusion: NO EDGE.** The selected research configuration had weak positive IC but negligible
net performance, then failed the terminal holdout materially. It should not be traded.

## Best research-period configuration

| Item | Result |
|---|---:|
| Target | Continuous volatility-adjusted market/sector residual return |
| Horizon | 20 trading days |
| Model | Ridge |
| Features | 38 |
| OOS Rank IC | 0.0218 |
| OOS ICIR | 0.1619 |
| Direction accuracy | 50.66% |
| Gross CAGR | 0.76% |
| Net CAGR at 10 bps | 0.13% |
| Net Sharpe | 0.06 |
| Sortino | 0.09 |
| Maximum drawdown | -3.91% |
| Mean two-way cohort turnover | 2.27% daily |
| Trades | 8,806 |

Signal construction was rank at least 0.90 plus predicted alpha above 0.05 for LONG, rank at most
0.10 plus predicted alpha below -0.05 for SHORT, and NO TRADE otherwise. Positions used overlapping
20-day cohorts, a 2% name cap, gross exposure no greater than one, and approximately neutral net
exposure. The hard regime filter was not selected. Inverse-volatility, magnitude, and equal sizing
converged because the 2% cap bound most selected names.

## Model and horizon comparison

The highest predictive IC was XGBoost at 10 days (Rank IC 0.0280, ICIR 0.2110), followed by
LightGBM at 20 days (0.0269, 0.2302). These higher-IC models did not translate into positive
cost-adjusted portfolios. One-day models were distinctly weaker (Rank IC 0.0052–0.0094).

The complete machine-readable comparison is in `artifacts/model_results.csv`; portfolio variants
and 5/10/20/50 bps scenarios are in `artifacts/strategy_results.csv`.

## Meta-labeling

A LightGBM classifier was trained on prior OOS primary predictions and volatility-scaled
triple-barrier outcomes. Research-period probability thresholds 0.50, 0.55, 0.60, and 0.65 all had
negative net Sharpe. Threshold 0.70 admitted no trades and is treated as invalid, not as zero-risk
performance. Meta-labeling did not improve the strategy.

## Baselines

From 2022–2024, SPY achieved an 8.88% CAGR and 0.57 Sharpe. The daily-rebalanced equal-weight
current-member universe had positive gross performance but negative net performance under the same
linear cost convention. Simple momentum and mean reversion were also negative net of costs. These
comparisons reinforce that the ML strategy did not justify its turnover.

## Locked holdout

| Item | Result |
|---|---:|
| Holdout Rank IC | 0.0135 |
| Holdout top-minus-bottom target spread | -0.0173 |
| Holdout gross CAGR | -4.73% |
| Holdout net CAGR | -5.33% |
| Holdout net Sharpe | -1.18 |
| Holdout maximum drawdown | -15.16% |

The negative top-minus-bottom spread despite positive mean daily rank IC is a warning that average
ordering did not survive in the tradable tails. No parameter was retuned from these holdout results.

## Data and inference limitations

The public fallback uses **today's S&P 500 constituents** and static current sector labels. It is not
point-in-time and has survivorship and membership look-ahead bias. Prices are Yahoo adjusted OHLCV;
spread, auction, locate, delisting, borrow, and market-impact histories are unavailable. Costs are a
linear approximation. Only four Alpha-101/GTJA-191 factors passed the trusted registry; the legacy
catalogs remain quarantined. These limitations would normally bias results upward, making the failed
holdout more—not less—important.
