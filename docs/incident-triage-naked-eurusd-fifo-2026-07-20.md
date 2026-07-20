# Incident triage — naked EUR_USD position / FIFO-unsafe flatten

**Raised by:** `com.axiom.safety_monitor`, first run after 13 days offline, 2026-07-20T20:27:42Z
**Check:** `no_open_position_without_tp_sl` — **severity CRITICAL**
**Triaged:** 2026-07-20, all evidence read from disk this session
**Checkpoint:** `trained_data/checkpoints/restoration-2026-07-20.json`

---

## 1. Disposition

**REAL ACTIVE RISK — small magnitude, 5 days old, with a recurring root cause.**

Not a false positive, not a stale marker, not a recovered incident. The position is open right
now: `trained_data/oanda/account_state.json` (mtime 2026-07-20 17:40, refreshed by the restored
API) reports `EUR_USD net_units 200.0, stop_loss null, take_profit null, unrealized_pl -0.582`,
`open_trade_count 2`.

Financial exposure is trivial — ~$228 notional on a practice account, 0.0006% of the $102,307 NAV.
**The defect it exposes is not trivial.** Two distinct bugs, detailed in §3.

The monitor was correct to fire, correct to classify CRITICAL, and correct that this predates its
outage. It attempted a halt; the system was already halted, so the result was
`already_halted, changed=False` — idempotent, no side effect.

---

## 2. Evidence — the 2.2 seconds that created it

Reconstructed from `trained_data/oanda/transactions.jsonl` (40 transactions in the window).

| time (2026-07-15) | event |
|---|---|
| 02:12:50.164 | `MARKET_ORDER` EUR_USD `CLIENT_ORDER` → fill +100 → **opens trade 2687, no SL/TP legs** |
| 02:12:50.492 | same → **opens trade 2689, no legs** |
| 02:12:50.740 | same → **opens trade 2691, no legs** |
| 02:12:52.314 | flatten sweep begins — EUR_JPY closes, emits 2× `ORDER_CANCEL LINKED_TRADE_CLOSED` |
| 02:12:52.332 | EUR_USD `TRADE_CLOSE` → **`ORDER_CANCEL … FIFO_VIOLATION`** |
| 02:12:52.334–.419 | GBP_JPY, USD_CAD, USD_JPY, USD_CHF ×2, USD_CAD close cleanly, each cancelling 2 linked orders |
| 02:12:52.336 | EUR_USD `TRADE_CLOSE` → **`ORDER_CANCEL … FIFO_VIOLATION`** (second failure) |
| 02:12:52.337 | EUR_USD `TRADE_CLOSE` → succeeds, closes **only** trade 2687 (`pl -0.0160`) |

Two independent facts fall out of this table:

1. **The three EUR_USD opens carried no brackets.** Every other instrument's close emitted two
   `ORDER_CANCEL … LINKED_TRADE_CLOSED` transactions — those are its SL and TP being torn down, so
   the rest of the book *was* bracketed. The EUR_USD trades emitted none, and the monitor's payload
   independently confirms `legs: []` on all three (ids 2687, 2689, 2691).
2. **The flatten did not flatten.** Two of three close attempts were rejected with
   `FIFO_VIOLATION`. Trades 2689 and 2691 survived and have been open ever since.

A later repair attempt also failed: `2026-07-15T15:55:35 TAKE_PROFIT_ORDER_REJECT` with
`rejectReason: TAKE_PROFIT_ORDER_WOULD_VIOLATE_FIFO_VIOLATION_SAFEGUARD` on trade 2691. So the
system *tried* to bracket the survivor 13 hours later and FIFO blocked that too.

### Why FIFO blocked it

OANDA enforces first-in-first-out on this account type: the oldest open trade in an instrument must
be closed first. Three separate same-direction EUR_USD trades existed simultaneously, so only the
oldest (2687) was closeable. Attempts against 2689 and 2691 were structurally rejected — and would
be rejected again on retry, in the same order, forever, until 2689 is closed before 2691.

**The FIFO trap was created by opening three separate trades in one instrument rather than one
position of 300 units.**

---

## 3. Root causes (two, both recurring)

### 3a. Positions opened without brackets — violates a hard trading invariant

`.claude/rules/trading.md` requires ATR-based SL/TP on every position, and CLAUDE.md lists
"ATR-based SL/TP only — never hardcoded pips" among the trading invariants. These three fills had
neither. Not yet traced to the emitting code path; the orders carry `reason: CLIENT_ORDER`, which
distinguishes them from the `TRADE_CLOSE` sweep that followed.

**Confidence: HIGH** that the opens were unbracketed (three independent sources: absent
`LINKED_TRADE_CLOSED` cancels, `legs: []` in the monitor payload, and `stop_loss/take_profit null`
in live account state). **Confidence: UNKNOWN** on which component issued them — not yet traced.

### 3b. `flatten_all` is FIFO-unsafe and does not verify completion

This is the more dangerous finding. The sweep issued close orders, absorbed two hard rejections,
and left the book non-flat — with no retry, no reordering to satisfy FIFO, and no alarm. A flatten
that silently does not flatten is a safety-critical failure: every downstream consumer, including
the operator, would reasonably read "flatten executed" as "book is flat."

**This is not a one-off.** `FIFO_VIOLATION` cancels appear on **two separate dates**:

- `2026-05-13T16:08:18` — 2 cancels
- `2026-07-15T02:12:52` — 2 cancels

Two occurrences. Under the promotion rule in `.claude/rules/improvement.md` (3+ observations → rule),
this sits at **candidate pattern**, one short of promotion. It should not be argued down to a
one-off; it should be watched for a third, or promoted early on the catastrophic-evidence exception
if the operator judges a non-verifying flatten to meet that bar.

**Confidence: HIGH** — the rejections and the surviving trades are both directly on disk.

---

## 4. Actions

### Not taken, deliberately

**I did not close the position.** Executing a trade — including a risk-reducing close — is outside
what this agent may do, and the system is globally halted. Unwinding it requires the operator, and
because of FIFO it must be done **oldest-first: close 2689 before 2691.** A naive "close all
EUR_USD" will reproduce the same rejection.

**I did not modify `flatten_all` or the order-emitting path.** Both are trading-execution changes
requiring explicit operator decision, and this session's mandate was stabilization, not behavioural
change.

### Recommended, in order

1. **Operator closes trades 2689 then 2691** (oldest first), or attaches brackets in that order.
   Until then the position is unprotected — small, but unprotected.
2. **Trace 3a**: find what issued three unbracketed `CLIENT_ORDER` fills on EUR_USD at 02:12:50 and
   why the bracket attachment was skipped. Until this is known, the same path can do it again.
3. **Fix 3b**: `flatten_all` must (a) sort closes oldest-first per instrument to satisfy FIFO,
   (b) re-read positions after the sweep and assert the book is actually flat, (c) raise loudly if
   it is not. A flatten that cannot verify its own completion should report failure, not success.
4. **Add a FIFO-aware position-accumulation guard** so one instrument does not end up holding
   multiple same-direction trades that can only be unwound in order.

---

## 5. What this says about the monitor

The check earned its place. Thirteen days offline, and inside 60 seconds of restart it surfaced a
real unprotected position and a real recurring execution defect that nothing else had reported.

It also demonstrates the failure shape recorded in §7 of
`docs/axiom-loop-remediation-plan-2026-07-20.md`: the naked position existed continuously from
2026-07-15, but for the entire time the only instrument capable of noticing was dead. **The defect
was never the missing stop-loss — it was that nothing could tell anyone about it.**
