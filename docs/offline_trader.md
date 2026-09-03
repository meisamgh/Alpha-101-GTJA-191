# Offline ML trader

The offline trader trains locally, creates shadow LONG/SHORT candidates, projects positions to
approximately zero net, market-beta, and sector exposure, and applies an evidence gate before any
decision can be marked tradable. It has no broker adapter and cannot submit orders.

Run:

```bash
make trader
```

Outputs:

- `artifacts/offline_trader_results.json`: selected threshold, research and diagnostic results,
  data disclosure, and the global trading gate.
- `artifacts/offline_trader_signals.csv`: latest predicted alpha, percentile, shadow decision,
  gated decision, and target weight.
- `artifacts/threshold_results.csv`: research-period sensitivity at 10 and 20 bps.

## Current gate

The research period selected an absolute-alpha threshold of 0.10. It produced 0.36% CAGR and 0.50
Sharpe at 10 bps and remained positive at 20 bps. The later diagnostic period produced -0.41% CAGR
and -0.48 Sharpe. Consequently `trading_enabled` is `false`, every final decision is `NO_TRADE`,
and candidate directions are retained only in `shadow_decision` for analysis.

The gate requires research Sharpe above 0.5, positive research Sharpe at 20 bps, and positive
diagnostic CAGR, Sharpe, and Rank IC. A failed gate always results in no trade.

## Data warning

The public fallback uses current S&P 500 membership and static sectors. It is survivorship-biased
and unsuitable for a production profitability claim. Replace it with a point-in-time provider before
treating results as investable evidence.
