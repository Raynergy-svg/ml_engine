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
| **Position TP/SL** (table columns + chart lines + distance) | `account_state.json` positions `take_profit`/`stop_loss` (`src/brokers/oanda_v20.py:_position_brackets`) | ✅ real |
| **Per-lane halt status** (oanda_fx / equity / brain) | `StateEngine.get_lane_status()` | ✅ real |
| **Equity harvester lane** (armed/live-vs-shadow, ship gate, cycle-ledger, live risk-gate verdict) | `live_gate_state.json` + `SHIP_GATE.json` + `cycle_ledger.jsonl` + `decision_gate.decide_cycle()` | ✅ real |
| **Brain loop** (last cycle, pending promotions, de-risk/halt audit) | `trained_data/brain_loop/*` + `.claude/brain_loop_audit.jsonl` | ✅ real — honest "not yet run" until the loop executes once |
| System health (lane oracle, verify_gate Hard-NO checks, active alerts) | `read_health()` (Risk tab) | ✅ real |
| Order-book sentiment | `trained_data/oanda/sentiment_snapshot.json` | 🟡 **placeholder** — tagged `DATA-ONLY; not wired` |

When the broker is unreachable (e.g. stale token) or a source file is absent, views show
an explicit **"not connected" / "—" / pending** state — never a fabricated number.

### Tier 7 honesty note
The panel reports the bot's own `running` + `running_reason` (heartbeat-fresh AND pid-alive)
verbatim — it never recomputes a rosier liveness. It also distinguishes **snapshot freshness**
(is the file still being written?) from **loop running** (is the control loop alive?), so a
fresh snapshot can honestly read `loop OFFLINE`. The header's "RUNNING" is the *trend lane*;
the Tier 7 "OFFLINE" is the *meta self-healing loop* — two separate processes, both truthful.

### Honesty cross-check
The trend signal the dashboard recomputes (`/api/strategy`) matches the bot's actual
open positions — two independent paths that agree.

## Endpoints (all GET, read-only)

`/api/account` · `/api/status` · `/api/trades` · `/api/equity` · `/api/strategy`
· `/api/tier7` · `/api/prices` · `/api/candles/{instrument}` · `/api/sentiment`
· `/api/instruments` · `/api/stream` (SSE) · `/api/health` · `/api/system_health`
· `/api/equity_sleeve` (harvester lane) · `/api/lanes` (per-lane halt, always-on)
· `/api/brain_loop`

Control routes (`/api/control/*`) are POST, gated by `AXIOM_CONTROL_ENABLED`, and covered
separately in `CONTROL_DESIGN.md` — `state` and `audit` sub-routes are GET/read-only but
live under that same gate today.

## Adding hosting + auth later (designed for, not yet built)

The data layer and frontend are decoupled by a single env var
(`NEXT_PUBLIC_AXIOM_API_URL`). To go remote/phone:
1. Deploy the FastAPI layer behind an auth proxy (it stays read-only).
2. Point the frontend env var at it; build/host the Next app.

No re-plumbing of the data contract is required. **Keep it practice-only and
read-only** — those guarantees live in `server/safety.py`.
