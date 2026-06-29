# Dashboard Data Contract (read-only) — 2026-06-25

The web dashboard is a **separate, READ-ONLY** consumer of the bot's OANDA practice
state. It **must never** place, modify, or cancel orders, and **must never** touch a
live/real-money endpoint. This doc is the stable contract: the bot keeps these file
paths + schemas stable; any breaking change is noted here first.

## HARD boundaries (non-negotiable)
- **Read-only.** No execution. The dashboard never calls `POST /v3/accounts/{id}/orders`,
  `PUT .../trades/{id}/close`, `.../positions/{inst}/close`, or any mutate endpoint.
- **Practice only.** If the dashboard talks to OANDA directly it uses
  `https://api-fxpractice.oanda.com` (REST) / `https://stream-fxpractice.oanda.com`
  (stream) ONLY. Never `api-fxtrade` / `stream-fxtrade` (live).
- **Token.** If the dashboard reads OANDA directly it needs its OWN practice token in
  its OWN gitignored env (`OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID`). Never commit it.
  Prefer reading the bot's persisted files (below) — no token needed for those.
- **Don't write** into `trained_data/oanda/` — the bot owns those files.

## Source A — bot-persisted state files (preferred; no token needed)
All under `trained_data/oanda/` (gitignored; written atomically by the bot).

### `account_state.json`  (refreshed every trend cycle)
Written by `src/brokers/oanda_v20.py:snapshot_account_state`.
```json
{
  "nav": 102103.0,            // float, account NAV (USD)
  "unrealized_pl": -47.6,     // float, open-position P&L
  "realized_pl": 0.0,         // float, account realized P&L (lifetime, OANDA "pl")
  "margin_used": 11503.0,     // float
  "margin_available": 90623.0,// float
  "open_trade_count": 8,      // int
  "currency": "USD",          // str
  "account_id": "101-...",    // str (PRACTICE, prefix 101)
  "positions": [              // array, one per open net position
    {"instrument": "USD_JPY", "net_units": 76597.0, "unrealized_pl": -12.37}
  ]
}
```

### `transactions.jsonl`  (append-only audit ledger)
One raw OANDA v20 transaction object per line (newest appended). Written by
`oanda_v20.py:TransactionLedger`. The dashboard's **trade history + realized P&L**
source. Key types and fields the dashboard can rely on:
- `ORDER_FILL`: `id` (str), `time` (RFC3339), `instrument`, `units` (signed str),
  `price`, `pl` (realized P&L of this fill), `financing`, `accountBalance`, `reason`.
- `DAILY_FINANCING`: `id`, `time`, `financing`, `accountBalance`.
- `MARKET_ORDER` / others: `id`, `time`, `type`.
Aggregate realized P&L = sum of `pl` over `type == "ORDER_FILL"`. Lines are immutable
once written; dedup by `id` if you cache (a rare mid-sync crash could double-append).

### `ledger_state.json`  → `{"last_transaction_id": "1351"}` (str) — ingestion cursor.
### `peak_nav.json`  → `{"peak_nav": 102200.0}` (float) — drawdown reference (ratchets up).
### `sentiment_snapshot.json`  → order/position-book buckets. **DATA-ONLY**, not a
  strategy input; reserved for a future pre-registered contrarian test. Shape:
  `{"note": "...", "books": {"EUR_USD": [ {price, longCountPercent, shortCountPercent}, ... ]}}`.

## Source B — runtime/halt state
- `.claude/state.json` (read-only): `{"halted": bool, "mode": str, ...}`. Display halt
  status; the dashboard does NOT write it (halt is operator/StateEngine-controlled).

### `.claude/tier7_state.json`  (Tier 7 autonomous self-healing loop — READ-ONLY display)
Written by `src/scanner/automation/tier7_state.py:write_tier7_state` (refreshed each
trend cycle; the dashboard's FastAPI may also call `build_tier7_state()` for a live pull).
**Display only — there is NO control path; the dashboard must never write/halt/act on it.**
```json
{
  "generated_at": "2026-06-29T10:30:00+00:00",
  "running": false,                  // bool — honest: heartbeat fresh AND pid alive
  "running_reason": "pid 84578 not alive",   // why running is true/false
  "halted": false, "mode": "live", "status": "running",
  "goal": "...", "improvement_focus": "...",
  "scan_cycle_count": 1,
  "current_action": "scan",          // state.json 'next' else last meta event
  "last_cycle": {                    // heartbeat freshness
    "heartbeat_ts": "2026-06-25T13:36:36+00:00", "age_seconds": 334618.3,
    "cycle_count": 1, "pid": 84578, "pid_alive": false, "scanner_alive_beacon": true
  },
  "self_heal": {"action_budget": {…}, "debounce": {…}},   // self-heal control-plane state
  "meta_last_event": {"change_id","stage","event","kind","deploy_target","updated_at"},
  "note": "READ-ONLY display snapshot for AXIOM. No control path."
}
```
HONESTY: `running` is `true` ONLY when the heartbeat is fresh (≤90s) AND its pid is
alive — a stale beacon reads `running:false` with the reason (never a false "alive").

## Source C — OANDA v20 PRACTICE read-endpoints (if reading OANDA directly)
All GET, under `https://api-fxpractice.oanda.com/v3/accounts/{accountID}/` unless noted:
- `summary` — NAV / P&L / margin / financing.
- `openPositions` / `positions` — open positions.
- `instruments` — tradable-instruments list.
- `transactions`, `transactions/sinceid?id=`, `transactions/idrange` — audit ledger.
- `transactions/stream` (host `stream-fxpractice`) — live transaction stream (fills/financing).
- `pricing?instruments=...` / `pricing/stream` (host `stream-fxpractice`) — quotes.
- `/v3/instruments/{instrument}/candles?granularity=&count=` — OHLC candles.
- `/v3/instruments/{instrument}/orderBook` · `/positionBook` — sentiment (data-only).
Rate limits: 120 req/s REST; ≤20 active streams; ≤2 new connections/s. Back off on 429.

## Stability / change policy
- The field names + paths above are the contract. Additive fields are safe; renames or
  removals are **breaking** and must be recorded in this doc (with a date) before shipping.
- `src/brokers/oanda_v20.py` (`snapshot_account_state`, `TransactionLedger`) is the
  single writer of `account_state.json` / `transactions.jsonl` — change the schema there
  and here together.
