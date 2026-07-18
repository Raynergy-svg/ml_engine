# AXIOM — live terminal for the Buddy trading engine

A modern, read-only web dashboard that visualizes Buddy's live **OANDA fxPractice**
(demo) trading. TradingView-grade aesthetic, original brand. **It visualizes; it
never trades.**

> **READ-ONLY · PRACTICE-ONLY.** AXIOM cannot place, modify, or close an order. The
> only broker access is a read-method allowlist proxy (`server/safety.py`), and the
> OANDA client it wraps is hard-pinned to `api-fxpractice`. SSE is GET-only, so the
> browser has no channel to push anything to the bot.

## Layout

```
dashboard/
  server/   FastAPI read-only data layer (Python) — reuses src.utils.oanda_practice
            + src.equity.oanda_trend (pure helpers only; never the order path)
  web/      Next.js 16 + React 19 + TypeScript + Tailwind v4 + lightweight-charts v5
  brand/    brand-sheet.html + logo.svg (the AXIOM identity)
```

## Run it locally

Two processes. From the repo root (`ml_engine/`):

**1) Data layer (FastAPI, port 8888)** — needs the gitignored practice token in
`.env.local` (`OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID`); file-backed views still work
without it.

```bash
python -m uvicorn dashboard.server.app:app --port 8888
```

**2) Frontend (Next.js, port 3000)**

```bash
cd dashboard/web
npm install        # first time only
npm run dev        # -> http://localhost:3000
```

> **Dev footgun:** don't run `next build` while `next dev` is serving — both use the
> same `.next/` directory, and the production build clobbers the dev server's
> artifacts. Pages keep serving but their client JS silently dies (e.g. the login
> form stops submitting) until the dev server is restarted.

The frontend reads `NEXT_PUBLIC_AXIOM_API_URL` (defaults to `http://localhost:8888`,
set in `web/.env.local`).

## What's wired vs placeholder (all REAL data — never fabricated)

| View | Source | Status |
|---|---|---|
| Account header (NAV, day/total P&L, margin) | `trained_data/oanda/account_state.json` | ✅ real |
| Open positions + live uPL | `account_state.json` | ✅ real |
| Transaction ledger | `trained_data/oanda/transactions.jsonl` (ORDER_FILL) | ✅ real |
| Equity curve | per-fill `accountBalance` in the ledger | ✅ real |
| Streaming price tiles (10 FX majors) | v20 `get_pricing` (read) | ✅ real, needs token |
| Candles + trend SMA(100) overlay + signal | v20 `get_candles` + `trend_targets` recompute | ✅ real, needs token |
| Trend strategy grid (long/flat per pair) | `trend_targets` (the bot's actual rule) | ✅ real, needs token |
| Halt / running / mode | `.claude/state.json` + lane snapshot freshness | ✅ real |
| **Tier 7 autonomous loop** (running/offline + reason, current action, last meta-pipeline event, self-heal actions, last cycle) | `.claude/tier7_state.json` (`tier7_state.py`) | ✅ real |
| **Position TP/SL** (table columns + chart lines + distance) | `account_state.json` positions `take_profit`/`stop_loss` | 🟡 **pending backend** — fields not in the data/contract yet; renders `—` + note |
| Order-book sentiment | `trained_data/oanda/sentiment_snapshot.json` | 🟡 **placeholder** — tagged `DATA-ONLY; not wired` |

When the broker is unreachable (e.g. stale token) or a source file is absent, views show
an explicit **"not connected" / "—" / pending** state — never a fabricated number.

### Tier 7 honesty note
The panel reports the bot's own `running` + `running_reason` (heartbeat-fresh AND pid-alive)
verbatim — it never recomputes a rosier liveness. It also distinguishes **snapshot freshness**
(is the file still being written?) from **loop running** (is the control loop alive?), so a
fresh snapshot can honestly read `loop OFFLINE`. The header's "RUNNING" is the *trend lane*;
the Tier 7 "OFFLINE" is the *meta self-healing loop* — two separate processes, both truthful.

### Pending on the backend contract
- **TP/SL bracket levels** (TASK A): the dashboard already consumes optional
  `take_profit`/`stop_loss` on each position (table columns, chart price-lines, pip distance).
  They render `—` until the bot writes those additive fields into `account_state.json` and
  records them in `docs/dashboard-data-contract.md`.

### Honesty cross-check
The trend signal the dashboard recomputes (`/api/strategy`) matches the bot's actual
open positions — two independent paths that agree.

## Endpoints (all GET, read-only)

`/api/account` · `/api/status` · `/api/trades` · `/api/equity` · `/api/strategy`
· `/api/tier7` · `/api/prices` · `/api/candles/{instrument}` · `/api/sentiment`
· `/api/instruments` · `/api/stream` (SSE) · `/api/health`

## Adding hosting + auth later (designed for, not yet built)

The data layer and frontend are decoupled by a single env var
(`NEXT_PUBLIC_AXIOM_API_URL`). To go remote/phone:
1. Deploy the FastAPI layer behind an auth proxy (it stays read-only).
2. Point the frontend env var at it; build/host the Next app.

No re-plumbing of the data contract is required. **Keep it practice-only and
read-only** — those guarantees live in `server/safety.py`.
