# Code Review — Equity Real-Money Execution Path (2026-06-21)

Independent Code-Reviewer pass on the uncommitted, Ralph-generated equity execution
subsystem (US-007/012/013/014/015 + IBKR equity broker changes). Trigger: operator asked
"who's reviewing the code?" — answer was nobody (self-attestation only). US-005 had already
proven self-attested green tests can hide a total no-op, so this review reads logic, not test
results.

## Verdict
**NOT safe to move toward paper trading as-is.** Multiple CRITICAL real-money defects that
in-order, no-crash, self-written tests cannot catch. Contained for now: none of this stack is
wired to a live consumer yet, and it's all uncommitted. Fix the CRITICALs before any paper money.

Solid (verified): ship-gate guard is genuinely fail-closed and non-bypassable across all
surfaces; no-mock invariant holds; atomic writes everywhere; bare-except discipline clean; risk
layer is a real gate (not a US-005 no-op); depth-walk math correct; market-calendar logic correct;
Apache-2.0 attribution present and not GPL/freqtrade.

## CRITICAL
- **C1 — Restart double-submit.** executors.py: `_submitted`/`_next_slice_index` never persisted,
  no executor `_load_state()`; after a crash the executor re-calls `place_equity_order` (ledger
  dedupe ≠ broker-call dedupe). IBKR gets a second live order. Fix: persist/reload submit progress
  OR reconcile against broker open-orders/positions by client_order_id before resend.
- **C2 — Fill detection reads AGGREGATE position as per-order fill.** executors.py:648-674 +
  :601 clamp. For a single-stock book the symbol IS the whole book; any pre-existing holding or a
  second order makes the position read ≥ this order's qty, so the clamp reports FILLED when nothing
  executed. Fix: per-order fills from execution reports (ib.fills()/execDetails), never net position.
- **C3 — Rebalance books PENDING as FILLED.** rebalance.py:561-574: `send_order==True` →
  `FILLED, fill_weight=target_weight`, but place_equity_order returns PENDING (accepted, unfilled).
  Corrupts the actual-weight ledger that drives next drift calc; skips genuinely-unfilled orders on
  restart. Fix: mark FILLED only on confirmed broker fill; distinct SENT/PENDING state.
- **C4 — Corporate-action check can't detect a MISSING split** (its whole purpose).
  corporate_actions.py:638-653 + 745-746: divergence only fires when caller's two inputs disagree;
  a real 2:1 split with no CorporateAction record passed produces no alert → 2× share count, corrupt
  cost basis. Fix: detect unexplained price jump near a split ratio with no applied action; validate
  cumulative_adjustment delta == product of applied ratios; halt on mismatch.

## HIGH
- H1 — TWAP per-slice fill by subtraction assumes in-order fills; out-of-order → slices stranded/FAILED while held (same root as C2).
- H2 — BrokerTimeoutError / orphaned-slice paths declare FAILED while a live order may exist (silent live exposure).
- H3 — `_default_fill_query` fabricates a zero on broker-query failure → "0 filled" → FAILED while shares held. Must surface, not mask.
- H4 — Risk: degross-to-zero in soft-drawdown band still returns `block_trade=False`; a consumer reading only block_trade trades full size into max DD. Set block_trade=True when composite ≤ epsilon.
- H5 — Risk: `evaluate()` has no try/except; a malformed weight raises out of the hot path instead of cleanly blocking.
- H6 — Risk: vol-targeter warm-up (< vol_lookback) sizes at full leverage with no vol control for ~21 obs.
- H7 — Corporate actions: same-ex-date multi-action ordering is input-order-dependent; post-delist same-date action silently skipped.

## MEDIUM
- M1 ibkr.py:1099-1199 gate/connection check ordering + fragile `[201,202]` substring match on error string ("20201" would match). Match reject codes structurally.
- M2 kill_switch.py:602-608 counts PENDING as "position closed"; reports success before flatten fills, never re-verifies flat. Re-poll positions.
- M3 corporate_actions.py:520-522 dividend mark floored to 1e-6 instead of halting on cps≥price; peak_nav ratcheted up even on halt (:682).
- M4 risk_agents.py:551-557 concentration check skipped when n_names ≤ concentration_top_k (default 3); correlation drops held names absent from matrix.
- M5 market_calendar.py:319-345 HaltRegistry read-modify-write has no fcntl lock; concurrent writers clobber a fresh halt. Project rule mandates locking.
- M6 market_calendar.py:117-118 `_to_utc` mislabels naive timestamps as UTC (4-5h shift; could report open pre-market).
- M7 depth_pricing.py:307-329 insufficient-depth returns over-optimistic price with no partial flag; add `sufficient`/`is_partial` or `require_full=True`.

## LOW
- L1 LICENSE-APACHE-2.0 file referenced by headers doesn't exist at repo root (only NOTICE). Add the license copy.
- L2 kill_switch get_trades(): malformed working order silently dropped from cancel set (could escape the kill); log louder.
- L3 corporate_actions:796 `except Exception` (logs+returns; narrow it); to_dict default=str can stringify stray types.
- L4 dead defensive branches (corporate_actions 492-496, executors TWAP double-break).

## Block order before any paper money
1. **C2 + C3 + H1** — fill/position ledger must reflect broker truth (currently fiction for a single-symbol book). Load-bearing.
2. **C1** — double-submit on crash.
3. **C4 + H7** — corporate-action missing-split detection + same-ex-date ordering.
4. **H2/H3/H4/H5** — silent-live-exposure + risk fail-open trio.

Source: Code Reviewer subagent, 2026-06-21. No files modified by the review.
