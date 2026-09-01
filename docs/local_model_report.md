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
| OOS Rank IC | 0.0492 |
| OOS ICIR | 0.2225 |
| Direction accuracy | 51.92% |
| Meta model | Triple-barrier LightGBM, probability > 0.60 |
| Net CAGR at 10 bps | 0.09% |
| Net Sharpe | 0.49 |
| Maximum drawdown | -0.24% |
| Mean two-way cohort turnover | 0.04% daily |
| Trades | 502 |

Signal construction was rank at least 0.90 plus predicted alpha above 0.05 for LONG, rank at most
0.10 plus predicted alpha below -0.05 for SHORT, and NO TRADE otherwise. Positions used overlapping
20-day cohorts, a 2% name cap, gross exposure no greater than one, and approximately neutral net
exposure. The hard regime filter was not selected. The primary-only portfolio had -1.16% net CAGR
and -0.33 Sharpe; meta-labeling was therefore selected from the research period.

## Model and horizon comparison

The highest predictive IC was Ridge at 20 days (Rank IC 0.0492, ICIR 0.2225), followed by XGBoost
at 20 days (0.0413, 0.2087) and LightGBM at 20 days (0.0393, 0.1993). One-day models were distinctly
weaker (Rank IC 0.0061–0.0105).

The complete machine-readable comparison is in `artifacts/model_results.csv`; portfolio variants
and 5/10/20/50 bps scenarios are in `artifacts/strategy_results.csv`.

## Meta-labeling

A LightGBM classifier was trained on prior OOS primary predictions and volatility-scaled
triple-barrier outcomes. The 0.60 research-period threshold produced 0.49 net Sharpe but only 0.09%
CAGR and 502 trades. Threshold 0.70 admitted no trades and is treated as invalid, not as zero-risk
performance. The frozen 0.60 threshold failed on the holdout, so meta-labeling did not establish edge.

## Baselines

From 2022–2024, SPY achieved an 8.88% CAGR and 0.57 Sharpe. The daily-rebalanced equal-weight
current-member universe had positive gross performance but negative net performance under the same
linear cost convention. Simple momentum and mean reversion were also negative net of costs. These
comparisons reinforce that the ML strategy did not justify its turnover.

## Locked holdout

| Item | Result |
|---|---:|
| Holdout Rank IC | 0.0487 |
| Primary-model top-minus-bottom target spread | 0.1662 |
| Meta-filtered net CAGR | -0.48% |
| Meta-filtered net Sharpe | -1.80 |
| Meta-filtered maximum drawdown | -0.90% |
| Meta-accepted candidates | 2,783 |

Predictive ordering survived, but it did not convert into positive realized PnL under the frozen
signal, cohort, meta, and cost rules. No parameter was retuned from these holdout results.

## Data and inference limitations

The public fallback uses **today's S&P 500 constituents** and static current sector labels. It is not
point-in-time and has survivorship and membership look-ahead bias. Prices are Yahoo adjusted OHLCV;
spread, auction, locate, delisting, borrow, and market-impact histories are unavailable. Costs are a
linear approximation. Only four Alpha-101/GTJA-191 factors passed the trusted registry; the legacy
catalogs remain quarantined. These limitations would normally bias results upward, making the failed
holdout more—not less—important.
