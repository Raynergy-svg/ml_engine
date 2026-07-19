# AXIOM Strategy Backtest and Forward-Test Readiness Report

Operator-authored review through commit 5fb76436 · 2026-07-18.
Committed verbatim as the standing evidence-accumulation directive; the
execution-order items below are tracked against this document.

## Operating conclusion

Stop adding architecture and begin a controlled evidence-accumulation
program: exact strategy-owned ledgers, frozen configurations, runtime-exact
backtests, forward observations BEFORE residual rewards or portfolio
promotion.

## Key repository facts at review time (re-verified from disk 2026-07-18)

- 5 registered hedge/portfolio lanes; 3 with committed scorecard history;
  1 cycle each; 0 aligned after-cost residual cycles.
- SMA mismatch: runtime `src/equity/oanda_trend.py` DEFAULT_SMA=100 vs the
  validated research constructions `src/equity/multi_asset_trend.py`
  SMA_WINDOW=200 and `src/equity/trend_sleeve.py` DEFAULT_SMA=200.
- Active crypto lane is H4 14-day cross-sectional momentum (ledger meta:
  `signal: xs_momentum_14d`, `gate_verdict: clears_ex_history=FALSE`);
  H5 time-series trend has no lane.
- `trained_data/oanda/account_state.json` absent → fx_trend and oanda_fx
  lanes currently have no book source.
- No `trained_data/hedge/portfolio_allocations.json` — correct until the
  candidate set is frozen; the gate refuses to guess.

## Forward-testing budget (P0 lanes)

1. Equity single-stock harvester — reproduce PIT ship gate on current
   universe hash; cost/slippage stress; automate recurring forward marks.
2. 37-asset multi-asset trend — dedicated shadow lane, parameters frozen
   (SMA200, monthly, HRP, 10% vol target).
3. Crypto H5 time-series trend — NEW `crypto_ts_trend` lane (do not
   overwrite H4); freeze universe/weekly cadence/costs/leverage; ≥52
   weekly observations.
4. Track B at scale — automate PIT filing capture + frozen-score
   provenance; ~405 filings for 80% power (48 committed today).
5. OANDA trend — runtime-exact backtest; resolve SMA100-vs-SMA200 before
   interpreting practice P&L.

## Acceptance rules (summary)

Code minimums (8 gate cycles / 5 residual cycles) are warm-up thresholds,
not proof. Weekly: ≥52 obs + 12 months. Monthly: 24–30 rebalances.
FX scanner: ≥200 closed practice trades + 6 months with pair/regime split.
Promotion: positive after-cost residual expectancy, stable bounds, no
unresolved cost basis, acceptable DD, low duplication, positive marginal
contribution under an operator-authored allocation plan.

## Execution order

1. ✅ 2026-07-18: regenerate scorecard + residual + promotion reports on
   the latest branch — all five lanes now carry explicit committed
   CONTINUE_SHADOW verdicts (this commit).
2. ✅ 2026-07-18: strategy-owned lanes shipped — `src/crypto/ts_trend_shadow.py`
   (H5, bit-for-bit regression-locked to the pre-registered
   `ts_trend_backtest`; separate ledger, H4 untouched) and
   `src/equity/multi_asset_trend_lane.py` (37-asset frozen spec at
   pre-registered target_vol=0.10, net regression-locked to
   `combined_portfolio`, universe SHA-256 stamped per row). Both registered
   in the hedge/portfolio registry (7 lanes covered universally).
3. ✅ 2026-07-18 (with one operator-side remainder):
   - **OANDA trend, runtime-exact** — `scripts/backtest_oanda_trend_runtime.py`
     ran the VERBATIM runtime rule on the cached FX candidates, both
     pre-specified windows. Verdict
     (`docs/oanda-trend-runtime-verdict-2026-07-18.md`): the SMA mismatch
     is resolved and NOT load-bearing — the FX-only subset is
     expectancy-negative under BOTH SMA100 and SMA200 at every cost point
     (Sharpe −0.09…−0.24, 2014→2026). Operator decision queued: retire the
     FX-only practice trend lane, or pre-register a runtime spec matching
     the validated cross-asset construction.
   - **Equity harvester** — SHIP_GATE.json's universe_hash (896fc636…)
     verified equal to the committed `universe_snapshot_pit.json`.
     Numeric re-derivation needs yfinance (proxy-blocked in the sandbox):
     run `python scripts/run_equity_harvester_shipgate_pit.py` on the
     operator machine.
4. ◐ capture automation shipped (`scripts/run_strategy_lanes.py` — one
   command records both new lanes + the hedge cycle for ALL lanes incl.
   equity/Track B + all reports; duplicate asof rows refused). Remaining
   (operator machine): run it once with `--refresh`, then schedule daily.
5. ✅ 2026-07-18: `scripts/research_metrics.py` — one command, one schema
   for every registered strategy (n, cumulative, ann return/vol, Sharpe,
   maxDD, DSR, bootstrap p, turnover, costs, residual fraction phi, beta,
   benchmark) from forward-ledger evidence only; nulls carry reasons;
   significance undefined below 30 observations by design. Artifact:
   `trained_data/research/strategy_metrics_report.json`.
6. Residual rewards stay disabled; no capital allocations until lane
   histories are populated and reviewed.

(Full operator report retained in operator records; this file is the
repository-tracked digest and checklist.)
