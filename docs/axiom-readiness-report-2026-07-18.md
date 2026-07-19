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
2. ✅ 2026-07-18, rev 2 evidence-safe 2026-07-19 (operator review of
   f590485, 5 findings fixed): strategy-owned lanes
   `src/crypto/ts_trend_shadow.py` (H5, bit-for-bit locked to
   `ts_trend_backtest`) + `src/equity/multi_asset_trend_lane.py` (37-asset
   frozen spec; manifest now carries the CORRECT Round-3 37-asset results —
   0.665/0.167 full, 0.788/0.162 OOS, DSR-OOS 0.843, clears_gate=FALSE —
   with the 21-asset run demoted to background context). Recording now goes
   through `src/evidence/forward_ledger.py`: activation BASELINE first (no
   already-realized return counted), deterministic backfill of every unseen
   bar (missed scheduler days recovered, never lost), per-row
   applied_book / next-target split with return-period fields (registry
   loads the standing target; performance ties to the applied book), and
   cadence accounting in weeks/rebalances (52 daily bars ≠ 52 weekly
   observations). Hedge ledger appends refuse duplicate snapshot identities
   (strategy, asof, book SHA-256, notional) and every evidence consumer
   (scorecard, residual attribution, promotion gate) dedupes defensively —
   scheduler re-runs can no longer mint extra "cycles". Daily scheduling of
   `run_strategy_lanes.py` is now safe. Registry = 7 lanes.
   Known follow-up: the H4 `crypto_momentum` lane's OWN recorder predates
   this engine (latest-bar semantics, 1 legacy row); migrate it the next
   time that lane is touched.
3. ◐ (scope corrected 2026-07-19 audit — SIGNAL-ONLY, plus two
   operator-side remainders):
   - **OANDA trend, signal layer** — `scripts/backtest_oanda_trend_runtime.py`
     validates the VERBATIM target-weight rule (equal-weight SMA long-or-
     flat) on the cached FX candidates, both pre-specified windows. Verdict
     (`docs/oanda-trend-runtime-verdict-2026-07-18.md`): the equal-weight
     SMA SIGNAL spec is expectancy-negative on the FX-only subset under
     BOTH windows at every cost point (Sharpe −0.09…−0.24, 2014→2026) —
     reject that signal spec. 2026-07-19: the MODELED risk-gated runtime
     (runtime's own sizing/cap functions, daily OHLC, N_TRIALS=3) is ALSO
     decisively negative (Sharpe −0.27…−0.31, 1/13 positive years, dd rail
     active ~76% of days) — `oanda_trend_atr_runtime_*.json` + verdict doc.
     Remaining operator fork: retire the FX-only lane, or pre-register a
     cross-asset runtime spec.
   - **Equity harvester** — SHIP_GATE.json's universe_hash (896fc636…)
     verified equal to the committed `universe_snapshot_pit.json`.
     Numeric re-derivation needs yfinance (proxy-blocked in the sandbox):
     run `python scripts/run_equity_harvester_shipgate_pit.py` on the
     operator machine.
4. ◐ capture automation shipped (`scripts/run_strategy_lanes.py` — one
   command records both new lanes + the hedge cycle for ALL lanes incl.
   equity/Track B + all reports; duplicate asof rows refused). Remaining
   (operator machine): run it once with `--refresh`, then schedule daily.
5. ✅ 2026-07-18, math corrected 2026-07-19 (audit): `scripts/
   research_metrics.py` — one command, one schema for every registered
   strategy, from forward-ledger evidence only. Safety rules now enforced:
   DSR only against REGISTERED campaign trial counts (crypto 15, trend 20;
   n_trials=1 is never computed — its benchmark degenerates to −inf); no
   annualization below 8 observations (one bar is never a 373% annual
   return) and the factor is DERIVED from observed cadence, never
   hardcoded; drawdown measures from initial capital (peak starts at 1.0);
   one asof = one observation (deduped on both ledger kinds); beta/
   benchmark honestly labeled NOT IMPLEMENTED. Artifact:
   `trained_data/research/strategy_metrics_report.json`.
6. Residual rewards stay disabled; no capital allocations until lane
   histories are populated and reviewed.

(Full operator report retained in operator records; this file is the
repository-tracked digest and checklist.)
