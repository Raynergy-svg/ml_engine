# Equity-Beta Harvester ("Track A") — Independent Audit (2026-07-01)

Follow-up to `docs/equity-harvester-verdict-2026-06-18.md`. That doc covers the original 9-sector-ETF
version (net Sharpe 0.596). This audit covers the **later, currently-binding** single-stock PIT
version that's actually cited as "cleared the ship gate" (`trained_data/backtests/SHIP_GATE.json`,
net Sharpe 0.908). Independent re-derivation by a separate Model QA agent, a Data Engineer agent, and
a Software Architect agent — none of which trusted the pipeline's self-report.

## 1. What it is (plain English)

It does not predict which stocks go up. It holds ~20 liquid large-cap US stocks equal-weighted, and
mechanically resizes the total bet based on two trailing signals: realized volatility (vol-target,
12%/yr) and drawdown from peak (soft de-risk at 10%, full de-risk to cash at 20%). No forecasting, no
stock-picking — it captures the equity risk premium (beta) with a defensive overlay, not alpha.
Rebalances monthly (+ a 0.5pp no-trade-band trigger), executes via IBKR (not OANDA — fully separate
broker path from the FX bot), costs modeled as 2bps flat + 5bps/%ADV slippage.

Data: **yfinance** (free, unofficial) for price/volume — the code has an explicit guard rail marking
yfinance "backtest only, not for live" (`src/equity/data_loader.py:184-189`), but the scripts that
produced the actual SHIP_GATE number bypass that guard and use yfinance directly. IBKR is the intended
live-data source but has not itself produced a validated result. A SEC-EDGAR fundamentals scraper and
a Wikipedia-based point-in-time S&P 500 membership reconstruction (`src/equity/sp500_membership.py`)
both exist in the repo and are **more rigorous** than what's live — but neither feeds the ship-gated
strategy; they only feed a separate, still-unproven offline "quality factor" experiment.

## 2. Independent verification verdict: **MIXED** (not a clean REAL, not fabricated/leaked)

Three separate findings, all with receipts:

**(a) The 0.908 number is real and mechanically sound — HIGH confidence.**
An independent agent reran `scripts/run_equity_harvester_shipgate_pit.py` live against yfinance
(fresh network pull, not a re-read of the artifact) twice and got 0.907 and 0.921 — matching the
claim to within data-revision noise. Lookahead was checked structurally: `execution_lag>=1` is
hard-enforced (`src/equity/backtest.py:301-304`), the vol/DD overlay uses `.shift(1)` (trailing-only,
`backtest.py:89-116`), and ADV uses a trailing (not centered) rolling window. Costs were verified to
actually deduct from returns, not just get reported, via a synthetic zero-cost-vs-with-cost check.

**(b) The universe is a curated 20-name mega-cap-tilted pool, not a representative large-cap
universe — HIGH confidence, this is the load-bearing finding.**
The same repo already contains a survivorship-corrected run (`trained_data/backtests/
pit_quality_bakeoff.json`, built from the Wikipedia PIT S&P reconstruction, 624-873 tickers including
108+ since-removed names) using the **identical** EW + vol/DD-overlay + `evaluate_gate` construction,
with **cheaper** costs (flat 2bps only, no ADV slippage). On that broader, more honest universe:
- Full-sample net Sharpe **0.740** (vs 0.908 on the curated pool), maxDD 0.195, gate PASS
- OOS (2019+) net Sharpe **0.355**, gate **FAIL** — sub-period decay 0.863 → 0.974 → 0.353 (2022-26)

This is not a bug or leakage — it's an economically legible concentration effect (top-20-by-ADV skews
hard toward the handful of mega-caps — AAPL/MSFT/GOOGL/AMZN/NVDA — that dominated 2010-2026 returns).
But it means **"0.908 clears the gate" is universe-dependent**, and the more defensible broad-universe
number fails the gate OOS. The PIT script's own docstring already flags a PASS as "necessary, not yet
sufficient" (`scripts/run_equity_harvester_shipgate_pit.py:20-24`) — this audit confirms that caveat
is load-bearing, not boilerplate.

**(c) Parameter provenance gap — MEDIUM confidence, cheap to fix.**
The shipped overlay params (target_vol=12%, dd_soft=10%/dd_hard=20%) don't trace to the one visible
grid-search artifact (`trained_data/backtests/equity_harvester_20260618T202412Z.json` — a 6-cell grid
on the *sector* book, not the single-stock book). That grid's actual winner was a different
config (vol10%/dd15-25), and the shipped config **fails the gate** (maxDD 0.258) on the sector book it
was tested against. `src/equity/ship_gate.py:71-74`'s docstring claim that the params "match the
validated grid" is not supported by the artifact on disk. Not evidence of heavy p-hacking (only 6
cells were ever tried, on a different book) — but a real traceability gap worth closing before
leaning further on these exact constants.

**Not concerns**: yfinance's day-to-day universe-hash instability (data revisions reshuffle the
top-20 ranking at the margin, Sharpe moves <0.02 across reruns) and the `SHIP_GATE_book.json` OOS
"FAIL" (that one fails purely on `total_years=8 < 10`, not on Sharpe — a genuinely different, more
benign failure mode than finding (b)).

**Recommendation**: don't treat 0.908 as *the* number for sizing/expectations. Surface the 0.740
full-sample / 0.355 OOS survivorship-corrected numbers alongside it. Re-run the vol/dd grid on the
actual single-stock PIT universe to close the parameter-provenance gap.

## 3. Staging status: NOT running, DISARMED, correctly wired to halt

Verified fresh from disk 2026-07-01: `.claude/state.json` → `halted: true` (as of
2026-07-01T12:11:23Z). `src/equity/decision_gate.py:_global_halt()` reads this same flag directly
(fail-closed path, not the fail-open `StateEngine.get_halted()`) and every equity gate (`decision_gate`,
`control_loop`, `live_gate`, `kill_switch`) refuses to act while it's true or while `SHIP_GATE.json`
hash/pass doesn't match. No process is currently running (`ps aux` empty); last activity was a one-off
demo run 2026-06-25 (`trained_data/equity/cycle_ledger.jsonl` seq=11, asof 2026-06-24 — 7 days stale).
`LiveGate` state file doesn't exist on disk → defaults to `armed=False`; it has never been armed.
No launchd/daemon wiring exists for the equity harvester (unlike AXIOM/FX). No OANDA coupling — IBKR
is a fully separate execution path; the only OANDA-flavored code near it is an unrelated sibling
strategy (`src/equity/oanda_trend.py`, already practice-pinned, not part of this harvester) and a
defensive (non-functional) `assert oanda_environment=="practice"` in the harvester's own entrypoint.

**Exact remaining steps for the operator to run this on IBKR paper** (none taken, none recommended
without operator sign-off):
1. `pip install ib_async` (not installed — confirmed `ModuleNotFoundError`).
2. Start IB Gateway/TWS on `127.0.0.1:7497` (paper) and log in (port 7496/live is never referenced
   anywhere in `src/equity/` — hardcoded absence).
3. Unhalt `.claude/state.json` (operator decision only — not touched by this audit).
4. Decide which driver: `scripts/run_equity_harvester.py --broker ibkr-paper` (the proven H1 path,
   shadow-fallback by default) vs. the fuller `src/equity/control_loop.py::AutonomousLoop` (built,
   tested, never actually exercised outside unit tests — no script currently constructs it).
5. **Wiring gap worth knowing about**: `--broker ibkr-paper` calls `place_equity_order` directly and
   currently bypasses `LiveGate.arm()` entirely — the typed-confirmation gate exists but isn't wired
   into the H1 script's paper-order path. If the operator wants that extra confirmation step
   enforced (as the PRD implies), that's a small wiring fix, not just "call arm()".
6. SHIP_GATE.json is 7 days old; not expired by any code TTL, but worth an operator eyeball,
   especially given finding (b) above.
7. Build a launcher/daemon if continuous operation (rather than one-off runs) is wanted.

## Sources
Model QA Specialist (independent Sharpe re-derivation, 43 tool calls, live yfinance reruns), Data
Engineer (mechanics/data-source writeup, 65 tool calls), Software Architect (wiring/staging audit, 16
tool calls) — all dispatched 2026-07-01, all disk-verified, no shared context with each other.
