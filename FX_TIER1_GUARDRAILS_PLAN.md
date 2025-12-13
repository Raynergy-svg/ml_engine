# FX Tier‑1 Guardrails Implementation Plan (Paper → Safer Automation)

Last updated: 2025-12-12

## Goal
Implement a **fail-closed FX day-trading workflow** that cannot “go rogue”, starting in **OANDA PRACTICE** with **Tier‑1 human confirmation** and strict session/risk/cost guardrails.

This plan is written so we can resume work quickly without re-deriving constraints.

---

## Non‑negotiables (User Constraints)

### Universe / timeframe
- Instruments (allowlist): `EUR_USD`, `GBP_USD`, `USD_JPY`
- Timeframe: `M5`
- Forecast horizon: `+1 bar` (next 5 minutes)

### Safety tier + execution
- Tier‑1: **human confirmation required** for any broker action (order placement or forced close)
- PRACTICE only (no live)
- Fail-closed defaults: if any required signal/confidence/inputs are missing or invalid → **no trade**

### Position/entry limits
- Max open positions: `1`
- Max entries/day: `1` ("one win/day" simplified as one entry/day for now)

### Session discipline (EST / America/New_York)
- Trade window: `08:00`–`11:30`
- Force-flat cutoff: `11:55` (close positions if any)
- No overnight holding

### Daily circuit breakers
- Daily loss stop: `-10%` (equity/NAV based; includes unrealized)
- Daily profit stop: **realized-only**, depends on confidence band:
  - High confidence: stop at `+30%`
  - Medium confidence: stop at `+20%`
  - Low confidence: **no trades**

### Costs / execution quality
- Spread filter per pair (block if spread too wide)
- Conservative slippage buffer (against us)

---

## Current Implementation Status (Repo)

### Configuration
- `config.yaml`
  - Added a comprehensive `fx:` policy section with:
    - timezone/session times
    - allowlist instruments
    - limits (max positions, entries/day)
    - risk settings (risk per trade, daily stops, ATR stop multiplier, RR take-profit)
    - costs (max spread pips by pair, slippage params, fallback spreads)
    - confidence bands + profit stop mapping
    - execution controls (`require_confirmation`, `practice_only`)

### Guardrails module
- `fx_guardrails.py` (new)
  - `FxPolicy` + sub-dataclasses for session/limits/risk/costs/confidence
  - Session/time helpers: `now_in_tz`, `within_time_window`, `should_force_flat`
  - Trading gate: `can_open_new_trade()`
  - Confidence mapping: `confidence_band()`, `profit_stop_pct_for_band()`
  - Persistent daily state: `FxDailyState` saved under `trained_data/logs/` (or config override)
  - Account summary ingestion: `update_state_from_account_summary()` calculates:
    - `drawdown_pct` from NAV (equity) vs start NAV
    - `realized_pct` from balance vs start balance (normalized to start NAV)
  - Circuit breakers: `check_daily_stops()`

### Broker client
- `oanda_practice.py`
  - Added:
    - `get_open_positions()`
    - `close_position()`

### FX paper helpers
- `fx_paper.py`
  - Candle parsing now supports bid/ask close extraction when OANDA returns `price="MBA"`
  - Added:
    - `spread_pips_from_df()`
    - `conservative_slippage_pips()`
  - Backwards compatible: when candles contain only `mid`, bid/ask columns drop if empty

### CLI wiring
- `main.py` → `fx_paper_trade()`
  - Enforces allowlist + `M5`
  - Loads FX policy + daily state
  - Updates daily state from account summary
  - Enforces daily stops; if stopped → disables trading for the day
  - Enforces force-flat; closes positions with confirmation when `--execute`
  - Enforces session window + entry limits + max open positions
  - Requests candles with `price="MBA"` and blocks on wide spread
  - Applies conservative slippage to stop-distance sizing
  - Requires confirmation before placing any order when `--execute`
  - Records `entries_today` and persists state on order placement

### Tests
- Added `tests/test_fx_guardrails.py` (unit tests for bands/stops/session/state persistence)

---

## How to Run (Local)

### Activate venv
```zsh
cd /Users/mirelacertan/Documents/ml_engine
source .venv/bin/activate
python -V
```

### Install minimal test deps (if missing)
```zsh
python -m pip install -U pip
python -m pip install -U pytest rich
```

### Run tests
```zsh
python -m pytest -q
```

---

## Known Gaps / Remaining Work

### 1) Confidence source (still placeholder)
- Current `main.py` uses a placeholder confidence `0.70` (band “medium”).
- Next step: compute confidence from **model outputs + reasoning**.

**Acceptance criteria**
- A single function (e.g., in `reasoning_enhanced.py`) produces a scalar confidence in `[0,1]` and a band.
- Low confidence prevents entries.
- Profit-stop band reflects actual confidence.

### 2) Profit-stop enforcement behavior
- Guardrails currently **disable trading** once profit stop is hit.
- Optional enhancement: when profit stop triggers and there are open positions, force-flat them (with confirmation if `--execute`).

**Acceptance criteria**
- Profit stop can block new entries and optionally triggers a flatten.

### 3) Entry/day definition vs “one win/day”
- Implemented as “one entry/day”.
- If you want literally “one winning trade/day”, that requires:
  - reading closed trade PnL and counting wins,
  - or tracking realized PnL change after closing.

**Decision needed**
- Keep “one entry/day” (simpler & safer) vs “one win/day” (more bookkeeping).

### 4) Spread/price precision and pip handling
- Current pip-size heuristic: JPY pairs `0.01`, others `0.0001`.
- Optional improvement: use OANDA instrument metadata for pipLocation/displayPrecision.

### 5) Torch dependency in tests
- Many repo tests import `main.py`, which imports heavy deps (`torch`, `rich`).
- Work has started to make `torch` optional for *importing* `main.py` in minimal environments.
- Recommended: keep `.venv` fully aligned with `requirements.txt` if running full suite.

---

## Roadmap (Paper → Safer Automation)

### Phase A — Paper, Tier‑1 only (current phase)
- Only allowlist pairs
- Only M5
- Only one entry/day
- Only within session
- Always confirmation on broker actions
- Circuit breakers enforced

### Phase B — Paper, Tier‑2 (optional later)
- Allow auto-close on force-flat without prompt (still practice)
- Keep human confirmation for opening new trades

### Phase C — Live (not now)
- Requires:
  - production-grade monitoring
  - exhaustive dry runs
  - broker error handling + idempotency keys
  - more robust fill/slippage models
  - explicit user approval

---

## Safety Checklist (Pre-Trade)
- ✅ Instrument in allowlist
- ✅ Time is inside session window
- ✅ Not past force-flat cutoff
- ✅ `entries_today < max_entries_per_day`
- ✅ `open_positions < max_open_positions`
- ✅ Confidence band != low
- ✅ Daily loss stop NOT hit
- ✅ Daily profit stop NOT hit
- ✅ Spread <= max allowed
- ✅ Confirmation granted (Tier‑1)

---

## Next Actions (When Resuming)
1. Replace placeholder confidence with real confidence from model/reasoning.
2. Decide if profit-stop should auto-flatten open positions.
3. Run the full test suite in `.venv` and fix any remaining environment mismatches.
