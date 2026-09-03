# Legacy research audit

## Scope and status

This audit covers `Regression_Time_bar_.ipynb`, `Copy_of_TA_lib.ipynb`,
`RegimeChange.ipynb`, `Volatility_and_Value_at_Risk.ipynb`, `101Alpha_code_1.py`,
`101Alpha_code_2.py`, and `GTJA_Alpha191.py` at commit `4687c81`. The legacy files are
preserved as references; they are not imported by the new framework.

## Findings

| Issue | Location | Inflation mechanism | New-pipeline control |
|---|---|---|---|
| Test-set early stopping | `Regression_Time_bar_.ipynb` and `Copy_of_TA_lib.ipynb`, XGBoost cells pass `X_test, y_test` in `eval_set` | Training decisions adapt to the reported test set | Models fit on train only; validation and test indices are separate |
| Global HMM fitting | `RegimeChange.ipynb`, `Volatility_and_Value_at_Risk.ipynb`, and `Regression_Time_bar_.ipynb` fit HMMs to the complete series before assigning regimes | Regime parameters and state assignments incorporate future observations | Any learned regime model must be instantiated and fit inside a training fold; current core exposes a trailing deterministic gate only |
| Global clustering | `Regression_Time_bar_.ipynb` fits KMeans outside a walk-forward pipeline | Cluster definitions see future distribution shifts | Fold-local model contract; clustering is not enabled until implemented as a fold transformer |
| Backward fill | `101Alpha_code_1.py:decay_linear` calls forward fill and then backward fill in-place | Early features can contain later observations; caller data is mutated | New factors never backward-fill or mutate input; insufficient history remains `NaN` |
| Incorrect cross-sectional rank | `101Alpha_code_1.py:rank` uses `DataFrame.rank(pct=True)` without a specified cross-sectional axis | Depending on input shape, ranks may be time-series rather than same-date cross-sectional ranks | Panel ranks explicitly group by `date` |
| Deprecated/broken APIs | Alpha scripts use `as_matrix`, old `pd.rolling_*`; GTJA constructor depends on undefined `get_price`/`get_index_stocks`; Alpha code 2 depends on Zipline symbols | Results cannot be reproduced reliably and formula behavior depends on abandoned environments | Provider-independent MultiIndex APIs and formula-level representative tests |
| Formula translation defects | Examples include Python `sum` where rolling sum is intended; GTJA alpha 2 applies `.diff()` only to a denominator due to parentheses; several `maximum/minimum` calls use constants where rolling extrema are described | Features differ from published definitions, making any attribution unreliable | Only Alpha 012/101 and GTJA 002/012 are promoted, each expressed directly and tested; the remaining catalog is quarantined pending validation |
| Same-bar execution ambiguity | Notebook strategies consume predictions alongside the current bar without a documented signal timestamp/fill price | A close-derived feature may be filled at that same close | Signals are defined after close `t`; fills are open `t+1`; alignment is unit-tested |
| Absolute-price objective and MSE selection | Regression notebooks predict price-like values and report MSE/RMSE | Low price error does not establish cross-sectional alpha or economic value | Return, residual, volatility-adjusted, and rank targets; Rank IC and net portfolio metrics are primary |
| Target/return reconstruction ambiguity | `Regression_Time_bar_.ipynb` computes percentage changes of predicted and true levels after prediction and reverses prediction lists | Reordering/alignment errors can manufacture direction or PnL | Targets are indexed at signal date and carry explicit `label_end`; next-open formula is tested |
| Missing purging and embargo | Notebook time-series splits do not remove overlapping forward labels | Adjacent folds share information through overlapping outcome windows | Interval-based purging plus configurable calendar embargo |
| Preprocessing leakage risk | No enforceable fold-local scaler/selector abstraction exists | Global normalization learns future means and variance | Scikit-learn pipeline is fit on train indices only; test verifies scaler mean |
| Corporate actions/adjustment unspecified | yfinance notebook downloads mix provider conventions; OHLC adjustment and dividends are not documented | Splits/dividends can create false returns and inconsistent OHLC | Schema separates close and adjusted close, validator flags extreme adjusted jumps; provider must document adjustment timing |
| Survivorship bias unaddressed | Single current tickers and no point-in-time membership table | Delisted and historically removed securities are absent | Universe metadata must state point-in-time status; static universes are explicitly disclosed as biased |
| Data availability timing unspecified | Technical and regime cells do not record when fields become knowable | Features can be treated as tradable before publication/finalization | Daily market fields are assumed final after close; execution starts next open; fundamental data is not accepted without effective timestamps |
| Transaction costs absent/unrealistic | Legacy backtests do not consistently model spread, slippage, commission, turnover, or borrow | Gross PnL is mistaken for implementable performance | Explicit bps-on-traded-notional costs, short borrow hook, gross/net series, and cost-reduction test |
| No exposure constraints | Legacy signals can become direct positions without portfolio caps | Concentration and leverage inflate returns and tail risk | Gross/net budgets, symmetric tails, max name weights, optional inverse-vol/confidence sizing |
| No experiment multiplicity control | Notebook cells are changed and rerun without an experiment registry | The best-looking result is selected from an unknown number of trials | Append-only records include commit, data version, configuration, status, and metrics; staged matrix is documented |
| No untouched holdout | No immutable final period or dataset fingerprint is identified | Repeated inspection converts the test into training data | Runner emits walk-forward tests only; a real final report additionally requires a frozen holdout and data version |

## Boundaries

The public repository does not contain a multi-asset dataset, point-in-time membership history,
spread observations, borrow rates, or a frozen holdout. Consequently, the legacy results cannot
support a profitability claim and the new framework cannot yet produce a real final strategy result.
