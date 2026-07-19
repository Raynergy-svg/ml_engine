# OANDA trend lane — SIGNAL-ONLY backtest verdict (2026-07-18; scope corrected 2026-07-19)

Readiness-report step 3: "Backtest the exact OANDA universe, costs, leverage
and risk gates. Resolve the runtime SMA100 versus validated research SMA200
mismatch before interpreting performance."

**SCOPE (corrected after the 2026-07-19 audit): SIGNAL-ONLY.** The
verbatim-exact part is the TARGET-WEIGHT rule (`trend_sleeve_weights(
close.ffill(), sma_window=W, step=1)` — the precise call
`oanda_trend.trend_targets` makes). The actual runtime then sizes with
ATR-based risk-normalized units (gross leverage is only a ceiling) and
applies drawdown halts, margin limits, the one-position gate,
currency-bucket limits, cost gates and SL/TP brackets — NONE of which are
modeled here. This verdict therefore rejects the equal-weight SMA SIGNAL
specification on this universe; it does NOT prove the risk-gated practice
runtime is expectancy-negative. Runner:
`scripts/backtest_oanda_trend_runtime.py`; artifact:
`trained_data/backtests/oanda_trend_runtime_20260719T024608Z.json`
(supersedes the 015827Z artifact, whose purpose field overclaimed
"runtime-exact"; numbers identical). Two pre-specified arms only
(N_TRIALS=2); costs: fixed turnover × {0,1,2} bps/side grid.

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

1. **For the equal-weight SMA SIGNAL on the FX-only subset, the window
   mismatch is resolved and NOT load-bearing.** Both windows are
   expectancy-negative at every cost point over 12 years (HIGH confidence
   for this signal + subset — full-period, causal, cost-stressed, both
   arms). The SMA200 arm is also NOT the validated 37-asset construction
   (that one is monthly, HRP-combined, vol-targeted, cross-asset) — it
   isolates the window variable only.
2. **The lane's positive expectation hypothesis was never about FX alone.**
   The validated research book earns its Sharpe from CROSS-ASSET diversity
   (rates, commodities, metals, crypto, equity indices) + HRP + vol target.
   Half the runner's candidate list (XAU/XAG, index CFDs, energy) is
   uncached, so this verdict is decisive for the FX subset and SILENT on
   the full candidate set (stated limit, not a loophole).
3. **UPDATE 2026-07-19 — the (b3) fork is now closed.** The MODELED
   risk-gated runtime (`scripts/backtest_oanda_trend_atr_runtime.py`,
   artifact `oanda_trend_atr_runtime_20260719T031332Z.json`) drives the
   runtime's OWN sizing/cap functions (1% NAV risk / 2×ATR stop, 4×ATR TP,
   one ticket per instrument, 2R bucket caps with same-cycle accumulation,
   3× gross ceiling, 50% margin rail, 20% dd halt) on the cached daily
   OHLC, one pre-specified config, cumulative N_TRIALS=3. Result: **also
   decisively negative** — Sharpe −0.27…−0.31 across the cost grid, 1/13
   positive years, bootstrap p 0.86–0.89, DSR 0.02–0.03, final NAV 0.85×,
   maxDD ~22–23%, and the 20% drawdown rail active on ~76% of days (the
   lane breaches early and spends most of its life halted — the halt
   preserves capital but there is no edge to resume into). Named
   approximations in the artifact (daily bars, next-open fills, SL-first
   intra-bar, no swap, no winner-management). **Operator fork simplifies
   to:** retire the FX-only practice trend lane (both the signal spec AND
   the modeled risk-gated runtime are negative), or pre-register a
   cross-asset runtime spec matching the validated 37-asset construction.
   Reading FX-only practice P&L as strategy evidence remains unsupported
   either way.

Not modeled (honest limits): ATR risk-normalized sizing (the runtime's
actual sizer), SL/TP brackets and position management, drawdown halts,
margin/one-position/currency-bucket/cost gates, intraday fills,
financing/swap, uncached candidates. Equity-harvester half of step 3: the
gate artifact's `universe_hash` (896fc636…) matches the committed
`market_data/equity/universe_snapshot_pit.json` exactly (verified
2026-07-18); numeric reproduction needs yfinance (proxy-blocked in the
cloud sandbox) — run `python scripts/run_equity_harvester_shipgate_pit.py`
on the operator machine to re-derive SHIP_GATE.json from scratch.
