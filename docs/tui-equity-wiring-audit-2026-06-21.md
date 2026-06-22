# TUI Audit — Equity Harvester Wiring Gaps (2026-06-21)

Headline: **`grep -r equity src/tui/` returns ZERO.** The TUI is structurally sound (handlers
wired, file paths valid, plumbing healthy) but pointed entirely at the retired FX stack. The
equity-beta harvester — now the bot's whole purpose — is invisible. The work is **repurposing
existing FX tabs to read the already-defined equity state-file contract**, not new infrastructure.
Most items are S/M because the data + atomic-write contract already exist; they just have no reader.

NOTE: `trained_data/equity/` does not exist yet (loop hasn't run live), so every reader must
degrade gracefully — show "pending / not running" when the state file is absent.

## P0 — Dead or broken
- **P0-1 · Diagnostics shows fake FX demo data by default** — diagnostics_screen.py:489-491. `on_mount`
  seeds hardcoded FX lines ("EUR/GBP spread elevated: 3.2 pips", "OANDA connection established",
  :815-823) because screen defaults live=False; 5s tick keeps adding fake entries. Flip to render
  real diagnostics / honest empty state. **S**
- **P0-2 · OANDA Health panel (top-right of Diagnostics) is dead FX infra** — diagnostics_screen.py:279-376,
  419-421, 1239-1256 (`_test_oanda_connection`→OandaBroker). Replace with equity broker/loop-health. **L**
- **P0-3 · Agents screen loads real weights then discards them** — agents_screen.py:538,585. `self._agents`
  assigned, never read; weight matrix renders from static `_BASE_WEIGHTS`. **M**
- **P0-4 · Agents roster wrong** — agents_screen.py:45-69: 14 short-names, 12 weights, canonical is 15
  (_team.py). `order_flow` absent; :455 filter silently drops unknown agents. **S** (but really repurpose
  to the 6 equity risk_agents — see P1-3)
- **P0-5 · ctrl+F asset-class cycle has no equity mode** — app.py:922,1716-1759. `_ASSET_CLASSES=
  ["fx","futures","hybrid"]`; sets config.asset_class on FX EmbeddedScanner which never references
  src/equity/. Add "equity". **M**
- **P0-6 · Reconciler thread/task leaked on shutdown** — embedded_scanner.py:211-212,564-571,1480.
  shutdown() never joins thread / cancels task. **M**

## P1 — Missing equity wiring (the new core is invisible)
- **P1-1 · Control-loop status has no surface** → make it the Overview hero panel (replace dead FX
  LIVE TRADES/MTF at app.py:963-970). control_loop.py to_dict (:535,544): cycle_count, halted, nav,
  peak_nav, transport_state, risk_decision, reconcile_report, loop phases. **M**
- **P1-2 · Ship-gate status invisible** → trained_data/backtests/SHIP_GATE.json (gate_pass, net_sharpe,
  max_dd, positive_years vs thresholds). Add StateStrip badge (app.py:959) + Diagnostics card.
  **S — highest value-per-effort.**
- **P1-3 · Risk-agent block state invisible** → risk_agents.py 6 gates + RiskDecision(block_trade,
  degross_factor, halt, reason). Repurpose F4 Agents screen. **M**
- **P1-4 · Rebalance schedule/plan invisible** → rebalance_state.json (last_rebalance_asof, active_plan
  per-order PENDING/SENT/FILLED/FAILED, target weights). Repurpose F3 Trades screen. **M**
- **P1-5 · Kill-switch state invisible + `k` binding still calls FX flatten_all** — app.py:895,1820.
  kill_switch.py writes kill_switch_state.json. Point Kill at equity kill switch; show on Diagnostics. **M**
- **P1-6 · Live/shadow gate status invisible** → live_gate.py (live_gate_state.json: armed, mode
  shadow/live, nav_fraction) + shadow_pipeline.py (realized vs backtest DD divergence). Prominent
  header indicator — the "are we risking real money?" bit. **M**
- **P1-7 · Alerts invisible** → alerts.py (alerts_state.json + alerts_audit/*.json: HALT, DD_BREACH...).
  Home: F2 Inbox. **M**

## P2 — Upgrades
- P2-1 ship-gate badge (subset of P1-2). P2-2 unified Harvester tab. P2-3 demo-data badge on Agents
  (agents_screen.py:718-738,849 fake EUR_USD voting, no marker). P2-4 stale "Live OANDA spread" comment
  trades_screen.py:566. P2-5 Diagnostics OANDA panel never auto-refreshes (:521-527).

## Clean (verified): Inbox, Config, Journal, Rules, Jobs, gate_trace_modal — all bindings wired to real
handlers, all read real existing paths. data_provider paths all exist. Plumbing healthy; just FX-pointed.

## If you fix 5: P1-2/P2-1 ship-gate badge (S) · P1-1 control-loop Overview panel (M) · P0-1 stop fake
FX demo (S) · P1-6 shadow-vs-live header indicator (M) · P0-3/P1-3 kill static agent data / repurpose
to equity risk_agents (M).

Source: Frontend Developer audit, 2026-06-21. Implementation pass was interrupted by session limit
(wrote nothing) — to be retried.
