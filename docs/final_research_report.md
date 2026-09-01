# Quantitative research report

## Executive summary

The legacy notebooks were audited and a clean research framework was built around causal panel
features, next-open targets, purged expanding walk-forward evaluation, fold-local preprocessing,
constrained portfolio construction, and explicit transaction costs. A deterministic synthetic smoke
experiment verifies orchestration only. No real point-in-time equity panel is present, so target,
horizon, model, portfolio, regime, and meta-label comparisons have **not** been empirically selected.

## What is implemented

- Panel loader and validation for duplicates, OHLC integrity, prices, volume, and extreme adjusted jumps.
- Trailing return, volatility, liquidity, trend, range, and market-relative features.
- A quarantined factor-integration path with four representative validated factors. The remaining
  Alpha-101/GTJA-191 formulas are not claimed correct until formula-level validation is completed.
- Raw, market/sector residual, volatility-adjusted, and cross-sectional rank targets for 1/5/10/20 days.
- Volatility-scaled triple barriers, CUSUM events, deterministic regime gating, and meta-target utility.
- Purged expanding walk-forward folds with a separate validation block and configurable embargo.
- Ridge, Random Forest, optional XGBoost/LightGBM factories; preprocessing fits inside each fold.
- Equal/volatility/confidence-weighted long-only or long-short portfolios with name/gross/net caps.
- Next-open execution, turnover costs and a borrow-cost hook; Rank IC and core portfolio metrics.

## Controlled research sequence

The configured first experiment is a five-day residual-rank Ridge baseline. On real data, the runner
should proceed in stages: compare horizons and targets with Ridge; promote stable candidates to tree
models; add residualization; compare long-only/long-short and costs; then test regimes, CUSUM,
triple-barrier meta-labels, and confidence sizing. Every failed run remains in the registry.

## Best target, horizon, model, and portfolio

**Not determined.** Selecting any winner without real data and an untouched holdout would violate the
research protocol. The five-day residual rank, Ridge, and market-neutral long-short portfolio are
starting hypotheses, not findings.

## Regime filtering and meta-labeling

**Not evaluated on market data.** The implementation supports a trailing hard gate and meta-target
construction. Learned HMM/clustering gates remain disabled until they are fold-local estimators.

## Costs

The engine separately records gross return, traded-notional costs, and net return. Default research
sensitivity should cover 5, 10, 20, and 50 bps plus explicit annualized short borrow. Market impact and
capacity require richer volume/spread data.

## Robustness and final untouched out-of-sample result

No final untouched market-data result exists. Therefore CAGR, Sharpe, Sortino, maximum drawdown,
annual volatility, turnover, cost drag, Rank IC, and ICIR are **not available**. The synthetic CI run
must never be quoted as strategy performance.

Before filling this section, freeze a point-in-time dataset version and terminal holdout, complete
factor formula tests, run the staged matrix, bootstrap by date blocks, report fold/year/sector/regime
stability, test cost sensitivity, and account for experiment multiplicity (DSR and preferably CPCV/PBO).

## Limitations

- No point-in-time universe; a current constituent list would introduce survivorship bias.
- No real dataset, delisting-return policy, or auditable corporate-action history is bundled.
- Daily OHLC does not reveal intraday barrier order when both barriers touch; conservative resolution
  or higher-frequency data is required for those cases.
- Linear bps costs omit nonlinear impact, auction uncertainty, locate availability, and time-varying spreads.
- Borrow costs, recalls, dividends, capacity, and sector classification effective dates are unavailable.
- Only representative Alpha-101/GTJA-191 formulas have been promoted into the trusted package.
- Feature selection stability, learned regimes, full meta-model training, DSR, PBO/CPCV, and benchmark
  comparisons require the real staged experiment suite.

## Conclusion

**No evidence of tradable alpha.** This is an evidence boundary, not a negative empirical finding:
the repository now has a validated research skeleton, but no real untouched out-of-sample study has
been run.
