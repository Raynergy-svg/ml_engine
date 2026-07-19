# OANDA trend lane — runtime-exact backtest verdict (2026-07-18)

Readiness-report step 3: "Backtest the exact OANDA universe, costs, leverage
and risk gates. Resolve the runtime SMA100 versus validated research SMA200
mismatch before interpreting performance."

Runner: `scripts/backtest_oanda_trend_runtime.py` (offline, cached
`market_data/factor/{PAIR}_D.csv`). Artifact:
`trained_data/backtests/oanda_trend_runtime_20260719T015827Z.json`.
Construction is the VERBATIM runtime call (`trend_sleeve_weights(close.ffill(),
sma_window=W, step=1)`, equal-weight on-set, long-or-flat, next-bar
application, gross leverage 3.0). Two pre-specified arms only (N_TRIALS=2):
the runtime window (100) and the research window (200). Costs: fixed
turnover × {0, 1, 2} bps/side grid — stated assumption, not fitted.

## Result (FX-only cached subset: the 10 FX candidates; 2014-05 → 2026-06)

| arm | cost | net Sharpe | maxDD (unlev) | boot p | pos. years |
|---|---|---|---|---|---|
| runtime SMA100 | 0 bps | −0.086 | 23.8% | 0.62 | 6/13 |
| runtime SMA100 | 1 bps | −0.164 | 25.8% | 0.73 | 6/13 |
| runtime SMA100 | 2 bps | −0.242 | 27.8% | 0.82 | 5/13 |
| research SMA200 | 0 bps | −0.101 | 19.3% | 0.64 | — |
| research SMA200 | 1 bps | −0.164 | 21.1% | 0.72 | — |
| research SMA200 | 2 bps | −0.226 | 23.9% | 0.79 | — |

Avg annual turnover: 43× (SMA100) vs 33× (SMA200) — the daily-step
construction re-touches the book constantly, so even 1 bp/side is a
material drag.

## Verdict

1. **The SMA window mismatch is resolved and it is NOT the load-bearing
   problem.** On the FX-only subset, BOTH windows are expectancy-negative
   at every cost point over 12 years. Switching the runtime from 100 to 200
   does not rescue the lane (HIGH confidence for this subset — full-period,
   causal, cost-stressed, both arms).
2. **The lane's positive expectation hypothesis was never about FX alone.**
   The validated research book earns its Sharpe from CROSS-ASSET diversity
   (rates, commodities, metals, crypto, equity indices) + HRP + vol target.
   Half the runner's candidate list (XAU/XAG, index CFDs, energy) is
   uncached, so this verdict is decisive for the FX subset and SILENT on
   the full candidate set (stated limit, not a loophole).
3. **Operator decision required (pre-registration discipline — no in-place
   tweak):** either (a) retire the FX-only practice trend lane as a
   negative result, or (b) pre-register a runtime spec that matches the
   validated construction (full cross-asset candidate set + HRP + 10% vol
   target + monthly cadence — i.e. point the practice account at the
   `multi_asset_trend` lane's frozen spec) and validate THAT runtime-exact
   before interpreting practice P&L. Continuing to run FX-only SMA100 and
   reading its practice P&L as evidence for the validated trend strategy
   is now demonstrably unsupported.

Not modeled (honest limits): order-level risk gates, intraday fills,
financing/swap, uncached candidates. Equity-harvester half of step 3: the
gate artifact's `universe_hash` (896fc636…) matches the committed
`market_data/equity/universe_snapshot_pit.json` exactly (verified
2026-07-18); numeric reproduction needs yfinance (proxy-blocked in the
cloud sandbox) — run `python scripts/run_equity_harvester_shipgate_pit.py`
on the operator machine to re-derive SHIP_GATE.json from scratch.
