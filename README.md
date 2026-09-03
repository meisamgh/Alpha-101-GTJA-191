# Alpha-101 + GTJA-191: Leakage-Aware Quant Research

This repository started as research around **WorldQuant Alpha-101**, **GTJA-191**, technical indicators, regime detection, time/volume bars, machine-learning regression, and portfolio backtesting.

The original notebooks are preserved for reproducibility and historical context. A new `quant_research/` path adds a more rigorous research framework inspired by modern financial-ML practice.

> **Research only.** Backtests are not evidence of future profitability. Transaction costs, slippage, borrow availability, survivorship bias, corporate actions, and data quality can materially change results.

## Why change the original regression target?

The original notebook predicts an absolute close price. For alpha research, that is usually not the economic quantity we care about. Price levels are strongly autocorrelated and can produce impressive prediction metrics without creating a tradable edge.

The modern pipeline instead targets a **cross-sectional, net, risk-adjusted future residual return rank**:

```text
signal at close[t]
      ↓
entry at open[t+1]
      ↓
future return over horizon
      ↓
remove market / sector component
      ↓
subtract estimated trading costs
      ↓
divide by ex-ante volatility
      ↓
cross-sectional rank
      ↓
primary ML target
```

This asks the model a more useful question:

> Which securities are expected to outperform their peers per unit of risk after costs?

## Recommended architecture

```text
                     Point-in-time OHLCV
                            │
                ┌───────────┴───────────┐
                │                       │
          Alpha-101 / GTJA-191     Regime features
                │              ADX / ATR / width / trend
                └───────────┬───────────┘
                            │
                            ▼
                    Primary alpha model
                            │
              residual-return rank score
                            │
                 ┌──────────┴──────────┐
                 │                     │
               LONG                  SHORT
                 │                     │
                 └──────────┬──────────┘
                            │
                       Range gate
                            │
                     candidate trade
                            │
                 CUSUM / event sampling
                            │
                    Triple barriers
                            │
                      Meta-label model
                            │
                 take / skip probability
                            │
                     Position sizing
                            │
             Next-session execution + costs
                            │
                     Portfolio metrics
```

## What was fixed conceptually

### 1. No test-set early stopping

A held-out test observation must not be passed to the model as an `eval_set` for early stopping. Hyperparameter selection and early stopping belong inside the training period.

### 2. Parameter-learning features must be fit inside training folds

HMMs, clustering, scalers, feature selection and ML models can leak future distribution information if they are fit on the full history before walk-forward testing.

### 3. Purged walk-forward validation

When a label at `t` uses returns through `t+5`, samples immediately before the test period can overlap the test label horizon. `PurgedWalkForwardSplit` removes these observations.

### 4. A range market is a gate, not the target

The pipeline uses trailing ADX, ATR percentile, Bollinger width and trend strength to produce `tradable_regime`. The primary return target remains continuous; the regime layer can keep the strategy in cash when the environment is unattractive.

### 5. Triple barrier is used for meta-labeling

The primary model decides **direction / relative attractiveness**. Triple-barrier labels answer a different question: **should this proposed trade be taken?** This preserves more information than using a binary barrier outcome as the only target.

## New files

```text
quant_research/
├── __init__.py
└── pipeline.py

example_modern_pipeline.py
requirements-modern.txt
tests/test_pipeline.py
```

### `build_alpha_target`

Creates a 5-day (configurable) target using next-session execution, sector/market residualization, ex-ante volatility adjustment and cross-sectional ranking.

### `add_regime_features`

Adds trailing-only regime features:

- realized volatility
- ATR / price
- Bollinger-band width
- trend strength
- ADX
- trailing volatility/width percentiles
- `tradable_regime`

### `build_meta_labels`

Creates CUSUM-filtered triple-barrier meta-labels for an existing long/short side. The meta-label never chooses direction.

### `PurgedWalkForwardSplit`

Expanding time-series validation on unique dates with overlapping-label purging and an optional embargo.

### `cross_sectional_positions`

Converts scores into equal-weight, approximately market-neutral long/short portfolios using cross-sectional tails.

### `run_long_short_backtest`

Signals are assumed known after `close[t]`. Execution begins at `open[t+1]`; costs are charged on turnover.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-modern.txt
python example_modern_pipeline.py
pytest -q
```

The example intentionally uses synthetic data. Replace the placeholder momentum/volume features with the repo's Alpha-101 and GTJA-191 feature outputs after adapting them to a common multi-asset panel schema.

## Suggested research sequence

1. Build a survivorship-aware liquid equity universe.
2. Adjust OHLCV for splits/dividends and enforce point-in-time membership.
3. Compute Alpha-101 / GTJA-191 features without backward filling future information.
4. Compare 1, 5, 10 and 20 trading-day target horizons.
5. Track **daily Rank IC / ICIR** before looking at portfolio PnL.
6. Run purged walk-forward tests.
7. Convert only extreme scores into candidate positions.
8. Add the regime gate and compare `always trade` vs `tradable_regime`.
9. Train a separate meta-label model on triple-barrier outcomes.
10. Evaluate net Sharpe, Sortino, drawdown, turnover, hit rate and capacity after realistic costs.

## Metrics that matter

Do not select a model mainly by `R²`, RMSE or raw direction accuracy.

Prefer:

- Rank IC / ICIR
- net Sharpe and Sortino
- maximum drawdown
- turnover and cost drag
- long-short spread return
- stability by year / sector / volatility regime
- performance decay as assumed costs increase

## Existing research assets

The repository also contains:

- `101Alpha_code_1.py` / `101Alpha_code_2.py` — Alpha-101 implementations
- `GTJA_Alpha191.py` — GTJA factor implementation
- `Regression_Time_bar_.ipynb` — original ML/time-bar experiment
- `RegimeChange.ipynb` — regime research
- `Volatility_and_Value_at_Risk.ipynb` — volatility / VaR research
- source papers for Alpha-101 and GTJA factors

The goal of the new pipeline is not to discard that work, but to place it inside a stricter **point-in-time → target → validation → portfolio → cost** framework.
