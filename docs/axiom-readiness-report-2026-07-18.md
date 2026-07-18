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
2. Add strategy-owned ledgers for `multi_asset_trend` and `crypto_ts_trend`.
3. Runtime-exact backtests for equity harvester and OANDA trend; resolve
   the SMA mismatch.
4. Automate recurring forward captures (equity, trend, crypto H5, Track B)
   with immutable configuration/provenance fields.
5. One consolidated research command emitting the same metric schema for
   every strategy.
6. Residual rewards stay disabled; no capital allocations until lane
   histories are populated and reviewed.

(Full operator report retained in operator records; this file is the
repository-tracked digest and checklist.)
