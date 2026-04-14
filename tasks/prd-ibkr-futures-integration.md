# PRD: IBKR Futures Integration for Buddy

## Introduction

Extend Buddy's autonomous trading system from FX-only (OANDA) to also support futures trading via Interactive Brokers (IBKR) TWS API. This introduces a broker abstraction layer so the entire scan → agents → gates → execute → learn pipeline works with both OANDA FX pairs and IBKR futures contracts (ES, NQ, CL, GC, etc.) without duplicating business logic.

The integration uses `ib_async` (Python 3.10+) connecting to IB Gateway (headless, port 4002 for paper). Futures have fundamentally different instrument mechanics: tick values instead of pip values, margin-based position sizing instead of unit-based, contract expiration/rolling, and bracket orders instead of attached SL/TP.

## Goals

- Create a clean broker abstraction (ABC) so Scanner, Agents, ExecutionManager, and RL loop are broker-agnostic
- Implement IBKRClient with full API surface matching OandaPracticeClient's capabilities
- Add futures-specific instrument definitions (tick values, multipliers, margin, roll calendar)
- Adapt position sizing from pip-based to tick-based for futures
- Support bracket orders (parent + TP + SL as atomic unit) for IBKR execution
- Add contract roll management (front-month tracking, automatic rollover)
- Maintain 100% backward compatibility with existing OANDA FX trading
- All new code follows existing project gates: JSON safety, retry/robustness, state persistence, test coverage

## User Stories

### LAYER 1: Broker Abstraction (Foundation)

### US-001: Create Broker ABC and Instrument Model
**Description:** As a developer, I need an abstract base class defining the broker interface so that Scanner and ExecutionManager can work with any broker without knowing the implementation details.

**Acceptance Criteria:**
- [ ] Create `src/brokers/__init__.py` with lazy imports
- [ ] Create `src/brokers/base.py` with `BrokerClient` ABC defining: `connect()`, `disconnect()`, `fetch_candles()`, `place_order()`, `close_trade()`, `get_nav()`, `get_open_positions()`, `get_price_quote()`, `get_trades()`
- [ ] Create `src/brokers/instrument.py` with `Instrument` dataclass: `symbol`, `broker_symbol`, `asset_class` (FX|FUTURES), `tick_size`, `tick_value`, `multiplier`, `pip_value`, `price_precision`, `margin_requirement`, `exchange`, `currency`, `contract_month` (optional)
- [ ] Create `src/brokers/types.py` with shared types: `OrderResult`, `CandleData`, `AccountSummary`, `PositionInfo`, `TradeInfo`
- [ ] All return types use these shared types (not raw dicts)
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 unit tests for Instrument model)

### US-002: Wrap OandaPracticeClient as BrokerClient Implementation
**Description:** As a developer, I need the existing OANDA client wrapped to implement the new BrokerClient ABC so the system can use it through the abstraction layer.

**Acceptance Criteria:**
- [ ] Create `src/brokers/oanda.py` with `OandaBroker(BrokerClient)` that wraps existing `OandaPracticeClient`
- [ ] `OandaBroker.fetch_candles()` returns `List[CandleData]` (converts from OANDA response format)
- [ ] `OandaBroker.place_order()` accepts `Instrument` + direction + quantity + sl_price + tp_price, returns `OrderResult`
- [ ] `OandaBroker.get_nav()` returns float (account balance)
- [ ] `OandaBroker.get_open_positions()` returns `List[PositionInfo]`
- [ ] `OandaBroker.get_price_quote()` returns bid/ask dict
- [ ] All existing OANDA env vars (`OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID`) still work
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 unit tests verifying wrapper delegates correctly)

### US-003: Create Instrument Registry with FX and Futures Definitions
**Description:** As a developer, I need a central registry of all tradeable instruments (FX pairs + futures contracts) with their specifications so position sizing and order building can look up tick values, multipliers, and margin requirements.

**Acceptance Criteria:**
- [ ] Create `src/brokers/registry.py` with `InstrumentRegistry` class
- [ ] Registry loads FX instruments from existing `PIP_VALUES` dict (EUR_USD, GBP_USD, etc.)
- [ ] Registry includes futures instruments: ES (CME, $50 mult, 0.25 tick, $12.50/tick), NQ (CME, $20 mult, 0.25 tick, $5.00/tick), CL (NYMEX, $1000 mult, 0.01 tick, $10/tick), GC (NYMEX, $100 mult, 0.10 tick, $10/tick)
- [ ] `registry.get(symbol)` returns `Instrument` or raises `KeyError`
- [ ] `registry.get_by_asset_class("FX")` returns all FX instruments
- [ ] `registry.get_by_asset_class("FUTURES")` returns all futures instruments
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 unit tests)

### US-004: Create Broker Factory with Config-Based Selection
**Description:** As a developer, I need a factory function that creates the correct broker client based on config/environment so the system can switch between OANDA and IBKR without code changes.

**Acceptance Criteria:**
- [ ] Create `src/brokers/factory.py` with `create_broker(config) -> BrokerClient`
- [ ] Add `broker_type` field to `ScannerConfig`: "oanda" (default) or "ibkr"
- [ ] Add `ibkr_host`, `ibkr_port`, `ibkr_client_id` fields to `ScannerConfig` with defaults ("127.0.0.1", 4002, 1)
- [ ] Factory returns `OandaBroker` when `broker_type="oanda"`
- [ ] Factory returns placeholder `IBKRBroker` (raises NotImplementedError) when `broker_type="ibkr"` (implemented in later story)
- [ ] Add `--broker` CLI argument to `main.py` (choices: oanda, ibkr)
- [ ] Typecheck passes
- [ ] Tests pass

### LAYER 2: Wire Abstraction Into Existing Pipeline

### US-005: Update ExecutionManager to Use BrokerClient ABC
**Description:** As a developer, I need ExecutionManager to use the BrokerClient abstraction instead of directly calling OandaPracticeClient so it can execute trades on any broker.

**Acceptance Criteria:**
- [ ] `ExecutionManager.__init__` accepts `broker: BrokerClient` instead of `oanda_client`
- [ ] Replace all `self._oanda.create_market_order(...)` calls with `self._broker.place_order(...)`
- [ ] Replace all `self._oanda.close_trade(...)` calls with `self._broker.close_trade(...)`
- [ ] Replace `self._oanda.get_account_summary()` with `self._broker.get_nav()`
- [ ] Replace `self._oanda.get_trades()` with `self._broker.get_trades()`
- [ ] Backward compatible: if no broker passed, lazy-init `OandaBroker` from env (same as today)
- [ ] Typecheck passes
- [ ] Tests pass (mock BrokerClient, verify delegation)

### US-006: Update Scanner Data Fetching to Use BrokerClient
**Description:** As a developer, I need Scanner's `_fetch_pair_data()` to go through BrokerClient so it can fetch candles from OANDA or IBKR depending on config.

**Acceptance Criteria:**
- [ ] Scanner receives `broker: BrokerClient` in `__init__` (or via config)
- [ ] `_fetch_pair_data()` calls `broker.fetch_candles(instrument, granularity, count)` instead of direct OANDA call
- [ ] Return format is identical: DataFrame with columns `open, high, low, close, volume`, `time` as index
- [ ] Local CSV fallback still works when broker fetch fails
- [ ] Feature engineering pipeline (`_compute_features`) receives identical DataFrame format
- [ ] Typecheck passes
- [ ] Tests pass

### US-007: Update Position Sizing for Tick-Based Instruments
**Description:** As a developer, I need the position sizer to handle both pip-based (FX) and tick-based (futures) sizing so it correctly calculates contracts instead of units for futures.

**Acceptance Criteria:**
- [ ] `DynamicPositionSizer.calculate_position_size()` accepts `Instrument` instead of `instrument: str`
- [ ] For FX instruments: existing pip-based calculation unchanged (units = risk_amount / (sl_pips * pip_value_per_unit))
- [ ] For FUTURES instruments: tick-based calculation (contracts = risk_amount / (sl_ticks * tick_value))
- [ ] Add `PositionSize.unit_type` field: "units" for FX, "contracts" for futures
- [ ] Minimum 1 contract for futures (no fractional contracts)
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 tests: FX sizing unchanged, ES sizing, NQ sizing, CL sizing, edge case 0 contracts)

### LAYER 3: IBKR Client Implementation

### US-008: Create IBKRBroker Connection Manager
**Description:** As a developer, I need the IBKR connection layer that manages connecting to IB Gateway, handling reconnection, and tracking connection state.

**Acceptance Criteria:**
- [ ] Create `src/brokers/ibkr.py` with `IBKRBroker(BrokerClient)`
- [ ] `connect()` calls `ib.connectAsync(host, port, clientId)` and waits for `nextValidId`
- [ ] `disconnect()` calls `ib.disconnect()` gracefully
- [ ] Connection state tracked: `is_connected` property
- [ ] Auto-reconnect on disconnect with exponential backoff (base 1s, max 30s, jitter)
- [ ] Error 502 (Gateway not running) logged with clear message
- [ ] Timeout on connect: 10 seconds max
- [ ] Typecheck passes
- [ ] Tests pass (mock ib_async, verify connect/disconnect/reconnect flows)

### US-009: Implement IBKRBroker Candle Fetching
**Description:** As a developer, I need IBKRBroker to fetch historical candles from IB Gateway so the Scanner can analyze futures data.

**Acceptance Criteria:**
- [ ] `IBKRBroker.fetch_candles(instrument, granularity, count)` calls `ib.reqHistoricalData()`
- [ ] Maps Buddy granularity strings to IB format: "H1" -> barSize="1 hour", durationStr="30 D"; "M5" -> barSize="5 mins", durationStr="5 D"
- [ ] Constructs `Future(symbol, exchange, currency, lastTradeDateOrContractMonth)` from `Instrument`
- [ ] Returns `List[CandleData]` with open, high, low, close, volume, time
- [ ] Respects IB pacing rules: no identical request within 15s (cache last request timestamp)
- [ ] `whatToShow='TRADES'`, `useRTH=0` (include extended hours)
- [ ] Typecheck passes
- [ ] Tests pass (mock ib_async.reqHistoricalData, verify contract construction and data mapping)

### US-010: Implement IBKRBroker Bracket Order Execution
**Description:** As a developer, I need IBKRBroker to place bracket orders (parent + TP + SL) atomically so Buddy can execute futures trades with proper risk management.

**Acceptance Criteria:**
- [ ] `IBKRBroker.place_order(instrument, direction, quantity, entry_price, sl_price, tp_price)` creates bracket order
- [ ] Parent order: `orderType="MKT"` for market entry (or "LMT" if entry_price provided)
- [ ] Take profit child: `orderType="LMT"`, `lmtPrice=tp_price`, `parentId=parent.orderId`
- [ ] Stop loss child: `orderType="STP"`, `auxPrice=sl_price`, `parentId=parent.orderId`
- [ ] `transmit=False` on parent and TP, `transmit=True` on SL (atomic submission)
- [ ] Returns `OrderResult` with trade_id, fill_price, status
- [ ] Direction mapping: "LONG" -> action="BUY", "SHORT" -> action="SELL"
- [ ] Typecheck passes
- [ ] Tests pass (mock ib_async.placeOrder, verify bracket construction and transmit flags)

### US-011: Implement IBKRBroker Account and Position Methods
**Description:** As a developer, I need IBKRBroker to fetch account NAV, open positions, price quotes, and trade details for monitoring and risk management.

**Acceptance Criteria:**
- [ ] `get_nav()` calls `ib.accountSummary()` and extracts NetLiquidation value
- [ ] `get_open_positions()` calls `ib.positions()` and returns `List[PositionInfo]`
- [ ] `get_price_quote(instrument)` calls `ib.reqMktData()` for bid/ask
- [ ] `get_trades()` calls `ib.openTrades()` and returns `List[TradeInfo]`
- [ ] `close_trade(trade_id)` places a closing order (reverse direction, same quantity)
- [ ] All methods handle connection-not-ready gracefully (return None or empty list)
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 unit tests)

### LAYER 4: Futures-Specific Features

### US-012: Implement Contract Roll Calendar
**Description:** As a developer, I need a contract roll calendar so Buddy knows which contract month to trade and when to roll to the next month.

**Acceptance Criteria:**
- [ ] Create `src/brokers/roll_calendar.py` with `RollCalendar` class
- [ ] Quarterly roll schedule for ES/NQ: March (H), June (M), September (U), December (Z) — roll 8 days before expiration (3rd Friday)
- [ ] Monthly roll schedule for CL/GC: roll 3 business days before expiration
- [ ] `get_active_contract(symbol, date) -> str` returns YYYYMM format (e.g., "202406")
- [ ] `days_until_roll(symbol, date) -> int` returns days until next roll
- [ ] `should_roll(symbol, date) -> bool` returns True if within roll window
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 tests: ES quarterly, CL monthly, boundary dates, roll-day exact, future dates)

### US-013: Add Futures Margin Validation to Risk Manager
**Description:** As a developer, I need the risk manager to check margin requirements before placing futures trades so Buddy doesn't attempt trades that exceed available margin.

**Acceptance Criteria:**
- [ ] Add `check_margin(instrument, contracts, account_nav) -> bool` to risk validation path
- [ ] Intraday margin estimates: ES=$500, NQ=$500, CL=$1000, GC=$1000 per contract (configurable)
- [ ] Overnight margin estimates: ES=$12000, NQ=$16000, CL=$6000, GC=$10000 per contract (configurable)
- [ ] Margin check runs before `execute_trade()` for futures instruments
- [ ] Margin exceeded → trade rejected with clear log message
- [ ] Total margin utilization tracked across all open futures positions
- [ ] Typecheck passes
- [ ] Tests pass

### US-014: Update Trade Journal for Multi-Broker Support
**Description:** As a developer, I need the trade journal to capture broker-specific fields so RL learning works across both OANDA FX and IBKR futures.

**Acceptance Criteria:**
- [ ] Add `broker` field to trade journal entries: "oanda" or "ibkr"
- [ ] Add `asset_class` field: "FX" or "FUTURES"
- [ ] Add `contract_month` field (nullable, only for futures)
- [ ] Add `tick_value` and `multiplier` fields (for futures P&L calculation)
- [ ] Futures P&L calculation: `realized_pl = (exit_price - entry_price) * multiplier * contracts * direction_sign`
- [ ] FX P&L calculation unchanged
- [ ] RL sync handles both asset classes (agent weights update from both FX and futures outcomes)
- [ ] Typecheck passes
- [ ] Tests pass (at least 5 tests: FX journal unchanged, futures journal with new fields, P&L calc for ES, P&L calc for CL, RL sync with mixed asset classes)

### LAYER 5: Config, CLI, and Integration

### US-015: Add Futures Config Profile
**Description:** As a developer, I need a futures-specific config profile so Buddy can run with appropriate settings for futures trading (different risk params, different SL/TP logic).

**Acceptance Criteria:**
- [ ] Add `futures_paper` profile to ScannerConfig profiles dict
- [ ] Profile settings: broker_type="ibkr", risk_per_trade_pct=0.02 (2%), max_open_risk_pct=0.10 (10%), atr_sl_multiplier=1.5, atr_tp_multiplier=2.0
- [ ] Profile pairs list: ["ES", "NQ", "CL", "GC"]
- [ ] Profile ibkr_port=4002 (paper trading)
- [ ] Add `futures_live` profile placeholder (ibkr_port=4001, risk_per_trade_pct=0.01)
- [ ] Existing FX profiles (balanced, aggressive, smart, conservative) unchanged
- [ ] `--profile futures_paper` works from CLI
- [ ] Typecheck passes
- [ ] Tests pass

### US-016: Add --futures and --instruments CLI Arguments
**Description:** As a developer, I need CLI arguments to control futures mode and instrument selection so the user can start Buddy in futures mode from the command line.

**Acceptance Criteria:**
- [ ] Add `--futures` flag to main.py argparse (shorthand for `--broker ibkr --profile futures_paper`)
- [ ] Add `--instruments` argument (comma-separated, e.g., `--instruments ES,NQ,CL`)
- [ ] `--instruments` overrides profile's default pairs list
- [ ] `--futures --dry-run` works (scan futures without executing)
- [ ] `--futures --watch` works (continuous mode for futures)
- [ ] Help text clearly describes futures mode
- [ ] Typecheck passes
- [ ] Tests pass

### US-017: Integration Smoke Test — IBKR Paper Trading End-to-End
**Description:** As a developer, I need an integration test that verifies the full pipeline works with IBKR paper trading: connect → fetch candles → run agents → evaluate gates → place bracket order → monitor → close.

**Acceptance Criteria:**
- [ ] Create `tests/integration/test_ibkr_e2e.py`
- [ ] Test 1: Connect to IB Gateway (skipped if Gateway not running)
- [ ] Test 2: Fetch 30 days of ES H1 candles, verify DataFrame shape and columns
- [ ] Test 3: Run feature engineering on futures candles, verify all features computed
- [ ] Test 4: Run agent team evaluation on futures data, verify verdicts returned
- [ ] Test 5: Place a bracket order on ES paper account, verify fill
- [ ] Test 6: Close the position, verify P&L recorded in journal
- [ ] Test 7: Verify RL sync processes the futures trade outcome
- [ ] Tests marked `@pytest.mark.integration` (skippable in CI without Gateway)
- [ ] Typecheck passes

### LAYER 6: Verification and Hardening

### US-018: Audit All PIP_VALUES References and Replace with Instrument Registry
**Description:** As a developer, I need to replace every hardcoded PIP_VALUES reference with Instrument Registry lookups so the system is fully instrument-agnostic.

**Acceptance Criteria:**
- [ ] Grep all files for `PIP_VALUES` — replace each with `registry.get(symbol).pip_value` or `registry.get(symbol).tick_value`
- [ ] Remove `PIP_VALUES` dicts from config.py, execution.py, and trading_metrics.py
- [ ] All pip/tick value lookups go through InstrumentRegistry
- [ ] FX behavior unchanged (same pip values as before)
- [ ] Futures instruments return tick_value instead of pip_value
- [ ] Typecheck passes
- [ ] Tests pass (run full existing test suite — zero regressions)

### US-019: Add IBKR-Specific Retry and Error Handling
**Description:** As a developer, I need IBKR API calls to have the same retry and robustness guarantees as OANDA calls (exponential backoff, specific exception catching, timeout enforcement).

**Acceptance Criteria:**
- [ ] All `ib.req*` calls wrapped with timeout (30s default)
- [ ] Connection errors trigger reconnect with exponential backoff
- [ ] Order rejection errors logged with IB error code and message
- [ ] Pacing violations (error 162) trigger automatic cooldown (15s wait + retry)
- [ ] Market data farm connection errors (error 2104/2106) handled gracefully
- [ ] All retries logged with attempt number, delay, and error context
- [ ] Typecheck passes
- [ ] Tests pass

### US-020: Full Regression Test — Verify OANDA Pipeline Unchanged
**Description:** As a developer, I need to verify that the entire OANDA FX pipeline still works identically after the broker abstraction refactor.

**Acceptance Criteria:**
- [ ] Run `python main.py --dry-run --pairs EUR_USD,GBP_USD` — same output as before refactor
- [ ] Run full existing test suite — zero failures
- [ ] OANDA env vars still work without any new config
- [ ] Default broker is "oanda" when no `--broker` flag specified
- [ ] Trade journal entries for FX trades have identical format (plus new `broker` and `asset_class` fields)
- [ ] RL sync still processes FX trade outcomes correctly
- [ ] Typecheck passes
- [ ] All existing tests pass

## Functional Requirements

- FR-1: The system must support two broker backends: OANDA (FX) and IBKR (Futures)
- FR-2: Broker selection must be config-driven (env var, CLI flag, or profile)
- FR-3: All market data must flow through a unified BrokerClient interface returning standardized types
- FR-4: Position sizing must automatically detect instrument type and use pip-based or tick-based calculation
- FR-5: IBKR orders must use bracket order pattern (parent + TP + SL, atomic submission)
- FR-6: Contract roll management must prevent trading expired contracts
- FR-7: Margin validation must prevent exceeding available margin on futures
- FR-8: Trade journal must capture broker and asset class metadata for RL learning
- FR-9: All existing OANDA functionality must work identically after refactor (zero regression)
- FR-10: IB Gateway connection must auto-reconnect on disconnect
- FR-11: All new modules must follow existing project gates (JSON safety, retry, state persistence, test coverage)

## Non-Goals (Out of Scope)

- No options trading support (futures only, not options on futures)
- No multi-account support (single IB account per instance)
- No GUI or web dashboard for futures monitoring (CLI only, same as FX)
- No live trading in first release (paper trading only via port 4002)
- No continuous futures (CONTFUT) for live data — only for historical backfill
- No cross-broker portfolio management (FX and futures tracked separately)
- No Docker/IBC setup automation (user provisions IB Gateway separately)
- No options chain analysis or Greeks calculation

## Technical Considerations

- `ib_async` requires Python 3.10+ — verify project Python version
- IB Gateway must be running (headless via Xvfb + IBC on Linux, or Docker gnzsnz/ib-gateway-docker)
- IB Gateway paper trading uses port 4002 (live uses 4001)
- Max 32 simultaneous client connections per Gateway instance
- Historical data pacing: max 60 requests per 10 minutes for small bars
- Futures P&L is in points * multiplier (not pips) — all display and logging must adapt
- Contract months use YYYYMM format (e.g., "202406")
- Bracket orders require sequential orderId management via `nextValidId` callback
- Margin requirements are broker-dependent and can change — use configurable defaults

## Success Metrics

- IBKR paper trading runs a full scan cycle on ES, NQ, CL, GC without errors
- Bracket orders placed and filled correctly on paper account
- Position sizing produces correct contract quantities (validated against manual calculation)
- Trade journal captures futures trades with correct P&L
- RL learning processes futures outcomes and updates agent weights
- Zero regression in existing OANDA FX pipeline (all existing tests pass)
- All new code has minimum 5 unit tests per story

## Open Questions

- Should futures and FX share agent weights, or should they have separate weight files?
- Should the roll calendar data come from a static table or be fetched from IB API?
- What is the minimum account size for futures paper trading to be meaningful?
- Should we add a "mixed" mode that scans both FX and futures in the same watch loop?
