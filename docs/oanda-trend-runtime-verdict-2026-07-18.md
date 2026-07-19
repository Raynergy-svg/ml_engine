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
3. **Operator decision required (pre-registration discipline — no in-place
   tweak):** (a) reject the simplified equal-weight FX-only SIGNAL
   specification (this verdict supports that), and either (b1) retire the
   FX-only practice lane, or (b2) pre-register a runtime spec matching the
   validated construction (cross-asset + HRP + 10% vol target + monthly)
   and validate it, or (b3) model/forward-test the ACTUAL ATR-risk-sized,
   gate-constrained runtime before judging it — the risk-gated runtime
   remains UNPROVEN in both directions. Reading FX-only SMA100 practice
   P&L as evidence for the validated trend strategy remains unsupported.

Not modeled (honest limits): ATR risk-normalized sizing (the runtime's
actual sizer), SL/TP brackets and position management, drawdown halts,
margin/one-position/currency-bucket/cost gates, intraday fills,
financing/swap, uncached candidates. Equity-harvester half of step 3: the
gate artifact's `universe_hash` (896fc636…) matches the committed
`market_data/equity/universe_snapshot_pit.json` exactly (verified
2026-07-18); numeric reproduction needs yfinance (proxy-blocked in the
cloud sandbox) — run `python scripts/run_equity_harvester_shipgate_pit.py`
on the operator machine to re-derive SHIP_GATE.json from scratch.
