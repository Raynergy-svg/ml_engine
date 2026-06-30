# PRD: Fully Hands-Off Equity-Beta Harvester Bot

## Introduction / Overview

Two years of research (committed this session) proved that **directional prediction has no exploitable retail edge** — across FX (~52%), crypto, price-equity factors, point-in-time fundamental value/quality, and PEAD, every market-neutral *alpha* test fails. The **one validated, gate-clearing strategy** is the **equity-beta harvester** (`scripts/build_equity_harvester.py`): vol-managed equity exposure + a causal drawdown / vol-spike circuit-breaker (Moreira–Muir 2017). On equal-weight sectors it returned **net Sharpe 0.60, maxDD 22.6%** through dot-com, GFC, COVID and 2022, clearing the ship gate.

This PRD re-points the existing autonomous machinery (control plane, drawdown guardian, self-heal, state persistence, `AgentVerdict`) away from the dead FX-direction target onto the harvester, as a **fully hands-off** bot. **Operator decisions (2026-06-18):** full build incl. a gated live path · deprecate & retire the FX directional path (non-destructively) · **single-stock large-cap universe**.

**This revision incorporates a two-reviewer completeness audit (Software Architect + Plan).** Headline corrections: the reusable autonomy stack is real but **every adapter is OANDA/FX-wired** (reconciler, transport, `flatten_all`, market-hours, alert conditions) — so "reuse, don't rebuild" really means **"port each FX adapter to IBKR-equity."** Single-stock equities also require corporate-action and market-calendar handling that FX never needed.

## Goals

- Run a **fully autonomous, hands-off** equity bot: data → universe → harvester weights → risk gates → execution → rebalance → monitor → self-heal, operator checks in only.
- Harvest the **equity risk premium** with tail control — make **no directional prediction**.
- Pass the **existing ship gate** (`src/factor/ship_gate.evaluate_gate`) on the deployed single-stock universe, recorded as a **machine-readable artifact**, before any capital is risked.
- Keep the runtime **deterministic and Claude-free** in the hot path; confine LLM/agentic work to the Tier-7 planning layer (enforced by a standing canary test).
- Handle the equity-specific realities FX never had: **corporate actions, market calendar/halts, equity contracts, a working kill switch.**
- Reach a **gated live path** only after shadow validation, a passed gate keyed to the deployed universe, and an explicit typed confirmation.

## Cross-Cutting Acceptance Criteria (apply to EVERY story)

- [ ] `pytest` + `flake8 --config=.flake8` pass.
- [ ] **No mocks** — real classes, real disk via `tmp_path`; external brokers/data use `@pytest.mark.integration` or skip, never `unittest.mock`.
- [ ] All mutable state written atomically (tmp + `os.replace`) and validated on load.
- [ ] No bare `except:` / `except Exception: pass` — log with context and surface.
- [ ] **Ship-gate guard (build stories US-006 onward only):** the module refuses to run unless `trained_data/backtests/SHIP_GATE.json` exists with `gate_pass == true` AND its `universe_hash` matches the active universe; a test asserts the guard blocks when the artifact is missing/false/mismatched.

## User Stories

Dependency-ordered. Foundational primitives first (the codebase has the autonomy plumbing but **none** of the equity-portfolio primitives).

### US-001: Equity price data-loader (point-in-time, adjusted, fail-loud)
**Description:** As the bot, I need cached, gap-validated daily equity prices, so every downstream story has a real data source.

**Acceptance Criteria:**
- [ ] Loader fetches daily adjusted OHLCV for a ticker list, mirroring the atomic/cache/fail-loud contract of `src/factor/data_loader.py` (which is FX-only); caches under `market_data/equity/`.
- [ ] Names the **live EOD source explicitly** (IBKR historical `fetch_candles` granularity "D" preferred over yfinance for live; yfinance allowed for backtest only) and records which was used.
- [ ] **Stale/NaN guard:** if the latest row is older than the configured freshness window or contains NaN closes, the loader fails closed (raises) — does not silently forward-fill.
- [ ] Unit test asserts cache round-trip + that a stale/NaN fixture raises.

### US-002: Single-stock universe builder (point-in-time, no survivorship)
**Description:** As a developer, I need a reproducible, look-ahead-safe universe so the bot trades a defensible list. **(Moved before US-005 — the gate must validate on the deployed universe.)**

**Acceptance Criteria:**
- [ ] Produces a date-indexed membership set of liquid US large caps (top-N by dollar volume / market cap) with documented reconstitution rules.
- [ ] Point-in-time membership (a name appears only on dates it qualified); emits a `universe_hash` for the deployed set.
- [ ] Unit test asserts membership on date D contains only tickers whose qualifying metric was known as of D, and **excludes a known later-IPO ticker** (no future leak).

### US-003: Portfolio-weight backtest harness (weights → curve/Sharpe/maxDD/turnover)
**Description:** As a developer, I need a portfolio backtester, so US-005's gate is real and reproducible. **(Extracted from US-001 — this was a hidden multi-session lift.)**

**Acceptance Criteria:**
- [ ] Takes date×ticker target weights + prices → equity curve, net Sharpe, maxDD, CAGR, turnover; reuses `stats()`/`overlay()` math from `build_equity_harvester.py`.
- [ ] **Minimal per-name cost model** here (flat bps + ADV-scaled slippage, like `COST_BPS`); the richer depth-aware book (US-014) is execution-only, NOT required for this harness.
- [ ] Execution lag ≥1 bar (causal); unit test on a synthetic 2-name panel asserts known Sharpe/maxDD/turnover.

### US-004: Strategy/engine interface contract
**Description:** As the control loop, I need a defined strategy protocol, so US-006 and US-009 agree on the interface.

**Acceptance Criteria:**
- [ ] Define a typed protocol: `compute_target_weights(asof) -> dict[ticker, float]` (causal; sums/bounds documented).
- [ ] Document how US-009's loop consumes it; **explicitly reuse `src/factor/portfolio.py`** (vol-targeting + gross/per-name caps already exist) rather than rebuilding.
- [ ] A trivial stub strategy implements the protocol and a test asserts the loop can call it.

### US-005: Re-validate harvester on the single-stock universe → SHIP_GATE.json (HARD GATE)
**Description:** As the operator, I need machine-readable proof the harvester clears the gate on the *deployed* universe before anything is built on it.

**Acceptance Criteria:**
- [ ] Run the harvester overlay on US-002's universe via the US-003 harness, full-cycle (≥2010–2026), with realistic per-name costs; write `trained_data/backtests/equity_harvester_singlestock_*.json`.
- [ ] **Reuse `src/factor/ship_gate.evaluate_gate`** (do not hand-roll); explicitly decide whether its `MIN_POSITIVE_YEARS=6` / `MIN_TOTAL_YEARS=10` criteria apply and record the decision. Resolve the 2-criterion vs 4-criterion inconsistency vs `build_equity_harvester.py`.
- [ ] **Write `trained_data/backtests/SHIP_GATE.json`** atomically: `{gate_pass, net_sharpe, max_dd, positive_years, universe_hash, asof}`.
- [ ] If `gate_pass == false`, document it and recommend reverting to the validated EW-sector universe; **all downstream build stories are guarded by this artifact** (see Cross-Cutting AC), so a failed gate mechanically blocks them.

### US-006: Harvester strategy module (runtime, deterministic)
**Description:** As the autonomous loop, I need the harvester as a strategy module implementing the US-004 protocol.

**Acceptance Criteria:**
- [ ] `HarvesterStrategy` returns target weights (vol-managed exposure × drawdown/vol-spike scalar), causal, **deterministic, no LLM/no network in the compute path** (same inputs → same weights, unit test).
- [ ] Reuses `src/factor/portfolio.py` weight construction/caps; thresholds configurable via `ScannerConfig` (dataclass field → profile dict → consumer getattr, all three).
- [ ] Long-only by default (see FR-11).

### US-007: Rebalance scheduler with idempotent restart
**Description:** As the bot, I need to rebalance on schedule without churn or double-fills on restart.

**Acceptance Criteria:**
- [ ] Triggers monthly (configurable) and/or on weight-drift beyond a no-trade band; emits the order delta current→target; respects the band.
- [ ] **Idempotent rebalance:** orders carry stable client-order-IDs; on restart mid-rebalance the scheduler reconciles already-sent/filled orders against target and sends only the remaining delta (test: kill after 2 of 5 orders → restart sends exactly the missing 3, no double-fill).
- [ ] State (last rebalance, current target, in-flight order IDs) persisted atomically; survives restart.

### US-008: Lean deterministic portfolio-risk agent layer
**Description:** As the bot, I need risk gates suited to a harvester, not a direction bet.

**Acceptance Criteria:**
- [ ] Implement lean agents returning the existing `AgentVerdict` shape (`_team.py:159`): **drawdown guardian, vol-targeter, gross/per-name exposure caps, concentration/correlation check, per-name ADV/participation cap (≤X% of 20-day ADV per rebalance), rebalance gate.**
- [ ] Drawdown guardian runs **every cycle**; breach de-grosses/halts via the circuit-breaker.
- [ ] Unit tests drive each gate with real configs and assert block/allow on edge cases (zero/negative/None).

### US-009: Corporate-action handling (single-stock critical) [NEW]
**Description:** As the bot, I must reconcile splits/dividends/delistings, so price/position math doesn't silently corrupt.

**Acceptance Criteria:**
- [ ] Detect and apply splits & dividends consistently between the data source and live broker positions; a 4:1 split must NOT register as a −75% drawdown.
- [ ] **Daily adjusted-vs-raw price sanity check:** if back-adjusted series and broker-held shares/price diverge beyond a tolerance, the bot **halts and alerts** (do not trade on skew).
- [ ] Delisting / M&A: the universe builder drops the dead ticker and the position manager liquidates/handles the cash-out; test drives a synthetic split, a dividend, and a delisting and asserts correct adjustment + halt-on-divergence.

### US-010: Equity market-calendar / session / halt gate [NEW]
**Description:** As the bot, I must only trade when the equity market is actually open.

**Acceptance Criteria:**
- [ ] Replace FX `is_market_open` (`config.py:1420`) on this path with a real NYSE/Nasdaq calendar (regular session 09:30–16:00 ET, holidays, half-days) via `exchange_calendars`/`pandas_market_calendars`.
- [ ] No rebalance/order outside the regular session; LULD/halt awareness (a rejected/halted order is detected and retried/deferred, not silently dropped).
- [ ] Test asserts: open on a normal weekday 10:00 ET, closed on a holiday and at 03:00 ET; a simulated halt defers the order.

### US-011: IBKR equity contracts + connect + place one order [split from US-006]
**Description:** As the bot, I need IBKR to accept equity orders at all.

**Acceptance Criteria:**
- [ ] Add `"EQUITY"` to the `AssetClass` literal + validators (`src/brokers/instrument.py:58`, `registry.py`) and a `Stock(symbol, "SMART", "USD")` contract path in `ibkr.py:_build_contract` (currently Forex/Future only).
- [ ] Connect to an IBKR **paper** account; place + confirm one whole-share equity order end-to-end (`@pytest.mark.integration`, skips cleanly without IBGateway).

### US-012: IBKR order lifecycle (types, TIF, fills, reconciliation, retries) [split from US-006]
**Description:** As the bot, I need robust equity order execution.

**Acceptance Criteria:**
- [ ] Documented policy: **market vs limit for rebalance, DAY vs GTC TIF, whole-share rounding with a residual-cash rule** (NOT the existing FX bracket-SL/TP order — the circuit-breaker is the stop).
- [ ] Partial-fill handling + reconciliation; explicit order state machine {SUBMITTED, PARTIAL, FILLED, FAILED}; a forced reconcile mismatch → FAILED, logged at ERROR with `reason_code`.
- [ ] On a simulated connection drop during `place_order`, retries up to `MAX_RETRIES` with backoff and raises a named `BrokerTimeoutError`; test asserts retry count + exception type; `grep` confirms zero bare `except` in `ibkr.py`.

### US-013: IBKR-native kill switch (flatten_all) [NEW]
**Description:** As the operator, I need a working hard kill switch on the equity path.

**Acceptance Criteria:**
- [ ] Implement `flatten_all` via the broker abstraction for IBKR (cancel all open orders + market-close all equity positions); the current `execution.py:6272` is hardcoded to OANDA REST and would no-op on equities.
- [ ] Integration-tested on paper: open positions/orders → `flatten_all` → assert flat.

### US-014: Depth-aware execution pricing (hummingbot, Apache-2.0)
**Description:** As the bot, I need realistic fill-price-at-size for execution sizing.

**Acceptance Criteria:**
- [ ] Reimplement `get_price_for_volume`/`get_vwap_for_volume` (from hummingbot `order_book.pyx`) for the equities path; **Apache-2.0 attribution header + NOTICE entry** retained; used for live execution sizing.
- [ ] Unit test vs a synthetic L2 book asserts correct VWAP-for-size / price-for-volume.

### US-015: Order-lifecycle executor framework (hummingbot, Apache-2.0)
**Description:** As the bot, I need a clean decoupled execution layer.

**Acceptance Criteria:**
- [ ] Adapt the `strategy_v2/executors` pattern (executor base + position/TWAP executors) for equity orders via the IBKR broker; **Apache-2.0 attribution** retained; routes through US-012's lifecycle.
- [ ] Integration test (paper IBKR) places + reconciles an order via the executor.

### US-016: Shadow fill simulator
**Description:** As the bot, I need simulated fills so shadow mode and the loop smoke-test can run without real orders.

**Acceptance Criteria:**
- [ ] A simulator takes an order + price/L2 input (US-014) and returns a fill (with modeled slippage); used by both US-017 and US-021.
- [ ] Unit test asserts deterministic fills for a fixed input.

### US-017: Autonomous control loop (fully hands-off)
**Description:** As the operator, I want the bot to run itself.

**Acceptance Criteria:**
- [ ] Loop runs continuously: schedule → weights → risk gates → rebalance/execute (via US-016 in shadow) → drawdown monitor → state persist → self-heal, no human in the loop.
- [ ] **Port the OANDA-only adapters to IBKR-equity:** a `state_reconciler` IBKR fetcher reconciling **shares + cash** (not just NAV) with equity-appropriate drift thresholds; an IBKR transport FSM (CONNECTED→DEGRADED→RECONNECTING) with heartbeat — on disconnect the loop **fails closed** (no blind rebalance).
- [ ] Graceful shutdown persists all module state atomically; restart resumes (freshness-checked). Corrupt state → log `STATE_CORRUPT`, skip execute, stay halted (test: inject corrupt file → assert no order emitted + halted true).
- [ ] **Standing Claude-free canary test:** an AST/grep test asserts no LLM import is reachable from the strategy/execution modules.
- [ ] Smoke test: loop runs **≥10 cycles** with a **null/stub executor**; test greps `logs/buddy_debug.log` for exact expected substrings.

### US-018: Out-of-band alerting + harvester alert conditions
**Description:** As the operator of a hands-off bot, I need to be notified out-of-band when something breaks.

**Acceptance Criteria:**
- [ ] Add a real notification channel (email/Slack/SMS) for HALT / broker-disconnect / DD-breach / rebalance-failed / stale-data / position-drift; `alert_manager` currently only writes files.
- [ ] Replace directional-trade alert conditions (consecutive_losses, win_rate_drop — meaningless for a harvester) with the harvester conditions above.
- [ ] Test asserts a simulated breach emits a notification payload.

### US-019: Outcome-grounded reflection loop (Tier-7 planning only)
**Description:** As the planning layer, I want to learn from realized outcomes — never in the hot path.

**Acceptance Criteria:**
- [ ] Reimplement (~100 lines, not imported) the TradingAgents (Apache-2.0) pattern: log decision `pending` → on maturity fetch realized return vs benchmark (alpha) → write a compact alpha-cited lesson → inject curated lessons into the next **planning** prompt; anchored on `trained_data/trade_journal_rl.json`, atomic append-only, rotation-capped, idempotent.
- [ ] Runs only in the Tier-7 planning/post-mortem path; the US-017 canary test confirms it is not reachable from the hot path.

### US-020: Deprecate & retire the FX directional path (non-destructive)
**Description:** As the operator, I want the dead FX path retired cleanly without losing history.

**Acceptance Criteria:**
- [ ] FX directional engine path + the 15 directional agents disabled from the active runtime (config flags off; removed from the harvester engine path); **code preserved in git history — no hard-delete of artifacts/journals.**
- [ ] Deprecation notes in code + docs pointing to this PRD and the research verdicts; no runtime path still depends on the retired agents; suite green after removal.

### US-021: Shadow / paper mode end-to-end
**Description:** As the operator, I want to watch the whole bot run in simulation before any real money.

**Acceptance Criteria:**
- [ ] Full pipeline runs in shadow (paper IBKR + US-016 simulated fills): real decisions, no real capital; bot stays halted/fail-closed by default.
- [ ] **Divergence check (verifiable):** if realized shadow maxDD exceeds backtest maxDD by >X pp OR turnover differs by >Y% (operator sets X, Y), emit a `DIVERGENCE` warning; test drives synthetic stats past/under the threshold.

### US-022: Gated live path (typed confirmation + universe-keyed gate)
**Description:** As the operator, I want a safe, explicit path to real money.

**Acceptance Criteria:**
- [ ] Live blocked unless: (a) `SHIP_GATE.json` `gate_pass == true` AND its `universe_hash` matches the deployed universe, AND (b) an explicit **typed confirmation** (reuse `src/tui/screens/mode_modal.py` `ModeConfirmModal`).
- [ ] Position sizing capped to a configurable small initial NAV fraction; max portfolio risk ≤15% NAV; drawdown guardian + US-013 kill switch active.
- [ ] Any failed precondition → refuse + stay halted; every live enable/disable logged; test asserts live cannot fire with a missing/false/mismatched gate.

## Functional Requirements

- **FR-1:** Target weights come from the deterministic harvester strategy (vol-managed exposure + drawdown/vol-spike circuit-breaker), causal (≤ t-1).
- **FR-2:** The runtime hot path contains **no LLM call and no network dependency** for the trade decision, enforced by a standing canary test (US-017).
- **FR-3:** The ship gate (`src/factor/ship_gate.evaluate_gate`) must pass on the deployed universe and be recorded in `SHIP_GATE.json` before any live capital; every build story is guarded by that artifact.
- **FR-4:** A drawdown guardian runs every cycle and de-grosses/halts on breach.
- **FR-5:** All mutable state persisted atomically and validated on load.
- **FR-6:** Execution uses depth-aware fill pricing + a robust order-lifecycle executor with partial-fill/retry/timeout handling and documented order-type/TIF/share-rounding policy.
- **FR-7:** A working IBKR-native kill switch (`flatten_all`) must close all equity positions/orders.
- **FR-8:** Live trading requires an explicit typed confirmation, a universe-matched passed gate, and defaults to halted/fail-closed.
- **FR-9:** Corporate actions (splits/dividends/delistings/M&A) reconciled between data and broker; trade halts on adjusted-vs-raw divergence.
- **FR-10:** Only trade during the real equity market session (NYSE/Nasdaq calendar, halts/LULD respected).
- **FR-11:** The harvester is **long-only** unless US-005 shows a long/short variant clears the gate materially better AND a borrow-availability/locate gate is added (US-011/US-012).
- **FR-12:** Cherry-picked hummingbot code retains Apache-2.0 attribution; **no GPLv3 (freqtrade) code may be copied.**
- **FR-13:** Tests use real classes + real disk (`tmp_path`); no `unittest.mock`/`MagicMock`/`patch`.

## Non-Goals (Out of Scope)

- Directional prediction of any kind; ML alpha / stock-picking (proven dead this session).
- FX trading (retired, not extended); crypto trading; market-making / liquidity-mining (separate possible experiment).
- Any LLM in the runtime hot path.
- Going live before shadow validation, a universe-matched passed gate, and typed confirmation.
- **Wash-sale / tax-lot accounting** (explicitly out of scope — do not half-build it).
- Intraday / sub-daily trading (harvester is daily/monthly-rebalanced).

## Technical Considerations

- **Reuse, don't rebuild — but every reusable piece is FX-wired, so "port to IBKR-equity":** `state_reconciler.py` (OANDA fetcher → IBKR shares/cash), `broker_transport.py` (OANDATransport → IBKR transport), `execution.py:flatten_all` (OANDA REST → IBKR), `config.py:is_market_open` (FX 24/5 → NYSE calendar), `alert_manager.py` conditions (win-rate → harvester). Name these in the relevant stories so Ralph doesn't assume drop-in reuse.
- **Reuse outright (not FX-coupled):** `src/factor/portfolio.py` (vol-target + gross/per-name caps), `src/factor/ship_gate.py` (the gate), `AgentVerdict` (`_team.py:159`), `ModeConfirmModal` (typed-live confirm), `src/autonomy/` self-heal + `code-repair` skill.
- **Reference clones (read-only):** `/tmp/hummingbot` (Apache-2.0) for US-014/US-015; `/tmp/tradingagents` (Apache-2.0) for the US-019 pattern only. **Do NOT copy from `/tmp/freqtrade` (GPLv3).**
- **Data:** yfinance OK for backtest; live EOD should prefer IBKR historical (`fetch_candles` "D") with a stale/NaN fail-closed guard (US-001).
- **Single-stock caveat:** validated Sharpe 0.60/22.6%-DD was on EW sectors; single-stock changes the numbers and maximizes corporate-action + liquidity exposure — US-005 is the hard gate guarding against silent degradation.

## Success Metrics

- US-005 single-stock harvester **clears the gate** and writes a valid `SHIP_GATE.json`.
- The bot runs **≥1 full month of shadow** end-to-end with zero human operation; drawdown/turnover match backtest within the US-021 tolerance.
- Hot path verified **Claude-free** by the standing canary test.
- Live provably **cannot** trade without typed confirmation + a universe-matched passed gate (tested).
- Corporate-action and market-calendar tests pass; suite green, no mocks, flake8 clean.

## Open Questions (operator input needed)

- Single-stock universe size/selection rule (top-100 by liquidity? curated list?) — US-002 to propose, operator ratifies.
- US-021 divergence thresholds **X** (maxDD pp) and **Y** (turnover %).
- US-022 initial live NAV fraction + per-name cap values.
- IBGateway availability/credentials for paper testing (US-011/US-012).
- Cash vs margin account (affects settlement/T+1 and any future shorting) — default cash, long-only.
