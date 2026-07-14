// API response types — mirror the FastAPI data layer (dashboard/server).

export interface Position {
  instrument: string;
  net_units: number;
  unrealized_pl: number;
  // Optional bracket levels — present once the bot writes TP/SL into account_state.json
  // (additive, contract-safe). Absent today → rendered as "—" / no chart line.
  take_profit?: number | null;
  stop_loss?: number | null;
  // Derived server-side from replaying the real fill ledger (never fabricated) —
  // absent/null when the lot can't be reconstructed -> render an honest "—".
  entry_price?: number | null;
  opened_at?: string | null;
  side?: "LONG" | "SHORT" | "FLAT";
  strategy?: string;
}

export interface Account {
  connected: boolean;
  source: string;
  account_id: string | null;
  currency: string | null;
  nav: number;
  unrealized_pl: number;
  realized_pl: number;
  margin_used: number;
  margin_available: number;
  open_trade_count: number;
  positions: Position[];
  peak_nav: number | null;
  drawdown_pct: number | null;
  // Freshness of the on-disk snapshot (the trend lane rewrites it each cycle).
  // stale=true => a stopped/lagging lane; the UI must not present it as live truth.
  snapshot_age_s: number | null;
  stale: boolean;
}

export interface Status {
  halted: boolean;
  mode: string | null;
  status: string | null;
  scan_cycle_count: number | null;
  environment: string;
  running: boolean;
  lane_running: boolean;
  running_source?: string; // "oracle" | "snapshot_age_fallback" — which path decided `running`
  account_snapshot_age_s: number | null;
  last_fill_time: string | null;
  scanner_heartbeat_alive: boolean;
  scanner_heartbeat_age_s: number | null;
  scanner_pid: number | null;
  last_updated: string | null;
}

export interface Price {
  bid: number;
  ask: number;
  mid: number;
  spread_pips: number;
  time: string;
  status: string;
  tradeable: boolean | null; // null = broker didn't report it (never fabricated true)
}
export interface Prices {
  connected: boolean;
  prices: Record<string, Price>;
  error?: string;
}

export interface Trade {
  id: string;
  time: string;
  type: string;
  status: "FILLED" | "ACTIVE" | "CANCELLED" | "REJECTED" | "POSTED" | "RECORDED";
  instrument: string | null;
  units: number;
  side: "BUY" | "SELL";
  price: number | null;
  pl: number;
  financing: number;
  half_spread_cost: number;
  reason: string | null;
  balance: number | null;
  tag: string | null;
  order_id: string | null;
  linked_fill_id: string | null;
  linked_fill_price: number | null;
  linked_fill_pl: number | null;
  fill_kind: string | null;
  trade_ids: string[];
  reject_reason: string | null;
}
export interface Trades {
  connected: boolean;
  source: string;
  count: number;
  fill_count: number;
  order_count: number;
  trades: Trade[];
}

export interface EquityPoint { time: string; balance: number; }
export interface Equity {
  connected: boolean;
  source: string;
  points: EquityPoint[];
  n: number;
  // P&L over THIS ledger window only (not since-inception — see Account.realized_pl).
  ledger_realized_pl: number;
}

export interface Signal {
  on: boolean;
  state: "LONG" | "FLAT";
  price: number;
  sma: number;
  sma_window: number;
  distance_pct: number;
}
export interface Candle {
  time: number; open: number; high: number; low: number; close: number;
  volume?: number | null; // real OANDA tick-count volume when reported; null if absent
}
export interface CandleResponse {
  connected: boolean;
  instrument: string;
  granularity?: string;
  candles: Candle[];
  sma: { time: number; value: number }[];
  signal: Signal | null;
  error?: string;
}

export interface Strategy {
  connected: boolean;
  universe: string[];
  sma_window?: number;
  granularity?: string;
  targets: Record<string, number>;
  on: string[];
  flat: string[];
  missing?: string[];
  partial?: boolean;
  status?: Status;
  error?: string;
}

export interface Sentiment {
  connected: boolean;
  wired_into_strategy: boolean;
  strategy_agent?: string;
  agent_weight?: number;
  note: string;
  books: Record<string, { price: string; longCountPercent: string; shortCountPercent: string }[]>;
  features?: Record<string, Record<string, number>>;
  source: string;
}

export interface Tier7LastCycle {
  heartbeat_ts?: string;
  age_seconds?: number;
  cycle_count?: number;
  pid?: number;
  pid_alive?: boolean;
  scanner_alive_beacon?: boolean;
}
export interface Tier7MetaEvent {
  change_id?: string;
  stage?: string;
  event?: string;
  kind?: string;
  deploy_target?: string;
  updated_at?: string;
}
export interface Tier7SelfHealEvent {
  ts?: string;
  cycle?: number;
  status?: string;        // "degraded" | "ok" | ... (bounded-autonomy state)
  degraded?: boolean;
  halted?: boolean;
  n_actions?: number;
  actions?: string[];
}
export interface Tier7SelfHeal {
  action_budget?: {
    actions_today?: string[];
    last_action_at?: string;
    last_action_label?: string;
  };
  debounce?: Record<string, string>;
  recent_events?: Tier7SelfHealEvent[];
}
export interface Tier7 {
  connected: boolean;
  source: string;
  pending_contract?: boolean;
  error?: string;
  snapshot_age_s?: number | null;
  snapshot_stale?: boolean;
  running?: boolean;
  running_reason?: string | null;
  halted?: boolean | null;
  mode?: string | null;
  status?: string | null;
  goal?: string | null;
  improvement_focus?: string | null;
  scan_cycle_count?: number | null;
  current_action?: string | null;
  autonomy_level?: string | number | null; // e.g. "L5" — optional; honest empty if absent
  max_autonomy?: number | null;
  bounded?: boolean | null;
  runtime?: {
    updated_at?: string;
    pid?: number;
    max_autonomy?: number;
    autonomy_level?: string | number;
    bounded?: boolean;
  } | null;
  last_cycle?: Tier7LastCycle;
  self_heal?: Tier7SelfHeal;
  meta_last_event?: Tier7MetaEvent;
  generated_at?: string | null;
  note?: string | null;
}

export interface LaneOracle {
  available: boolean;
  running?: boolean;
  oanda_trend_proc?: boolean;
  account_state_fresh?: boolean;
  account_state_age_s?: number;
  tier7_heartbeat_fresh?: boolean;
  tier7_heartbeat_age_s?: number;
  tier7_pid_alive?: boolean;
  error?: string;
}
export interface GateCheck { name: string; ok: boolean; hard_no?: boolean; detail?: string }
export interface Gates {
  available: boolean;
  all_ok: boolean;
  status: string; // GREEN | RED | UNKNOWN
  checks: GateCheck[];
  hard_no_count: number;
  verdict_age_s: number | null;
}
export interface AlertItem {
  alert_type: string;
  severity: string;
  message: string;
  timestamp: string;
  value?: number;
  threshold?: number;
  pair?: string;
  acknowledged?: boolean;
}
export interface Alerts {
  available: boolean;
  active: AlertItem[];
  count: number;
  max_severity: string | null;
  last_updated: string | null;
}
export interface SystemHealth { lanes: LaneOracle; gates: Gates; alerts: Alerts }

export interface DiagnosticCheck { label: string; ok: boolean | null; metric: string | null }
export interface Tier7Diagnostics {
  checks: DiagnosticCheck[];
  score: number | null;
  trend_pid: number | null;
  tier7_pid: number | null;
  tier7_process_stat?: { cpu_pct: number; rss_mb: number } | null;
}

export interface ApiHealth {
  ok: boolean;
  oanda_connected: boolean;
  environment: string;
  control_enabled: boolean;
}

export interface ControlLoopState {
  running: boolean;
  pids: number[];
}

export interface ControlState {
  ok: boolean;
  environment: string;
  halted: boolean;
  /** Per-lane halted status (oanda_fx / equity / brain), fed by StateEngine.get_lane_status().
   * The legacy global `halted` flag above still ORs over every lane — a lane can never
   * show running while `halted` is true. */
  lanes?: Record<string, boolean>;
  gross_leverage: number | null;
  override_updated_at: string | null;
  leverage_cap: number;
  loops: Record<string, ControlLoopState>;
  /** ARM lockdown (2026-07-01): flatten/set_gross_leverage/start_loop/unhalt are
   * refused server-side unless armed — this mirrors that state, it doesn't enforce it. */
  armed: boolean;
  arm_expires_at: string | null;
  armed_by: string | null;
}

// GET /api/lanes — always-on per-lane halt read (visible even with AXIOM_CONTROL_ENABLED off).
export interface LaneStatus {
  readable: boolean;
  global_halted: boolean;
  lanes: Record<string, boolean>;
  known_lanes: string[];
}

// GET /api/equity_sleeve — equity-harvester lane status.
export interface EquityShipGate {
  available: boolean;
  gate_pass?: boolean;
  net_sharpe?: number | null;
  max_dd?: number | null;
  asof?: string | null;
  recommendation?: string | null;
  universe_hash?: string | null;
}
export interface EquityLiveGate {
  available: boolean;
  armed: boolean;
  universe_hash?: string | null;
  initial_nav_fraction?: number | null;
  max_portfolio_risk_fraction?: number | null;
  armed_at_ts?: number | null;
  disarmed_at_ts?: number | null;
  last_event?: string | null;
  last_event_reason?: string | null;
  last_event_ts?: number | null;
}
export interface EquityCycleRecord {
  seq: number;
  asof: string;
  decision: string;
  actionable: boolean;
  gross: number;
  n_orders: number;
  reasons: string[];
}
export interface EquityGateDecision {
  available: boolean;
  decision?: "refuse" | "halt" | "no_act" | "abstain" | "continue";
  reasons?: string[];
  actionable?: boolean;
  error?: string;
}
export interface EquitySleeve {
  connected: boolean;
  dormant: boolean;
  mode: "live" | "shadow";
  asof: string | null;
  rebalance_id: string | null;
  target_weights: Record<string, number>;
  actual_weights: Record<string, number>;
  source: string;
  live_gate: EquityLiveGate;
  ship_gate: EquityShipGate;
  last_cycles: EquityCycleRecord[];
  gate_decision: EquityGateDecision;
}

// GET /api/brain_loop — Sonnet brain-loop status (has not run in production yet; every
// field below is an honest empty until it does).
export interface BrainLoopPromotionRequest {
  hypothesis_id: string;
  target: "shadow" | "live";
  status: string; // "APPROVED_SHADOW" | "PENDING_OPERATOR"
  requires_operator_arm: boolean;
  decision: Record<string, unknown>;
  created_at: string;
}
export interface BrainLoopLedgerEvent {
  seq?: number;
  ts?: string;
  kind?: string; // "register" | "result" | "gate_verdict"
  hypothesis_id?: string;
  name?: string;
  status?: string;
  decision?: string;
  reasons?: string[];
}
export interface BrainLoopAuditEvent {
  ts: string;
  actor: string;
  action: string; // "halt" | "tighten_risk_config"
  reason?: string;
  risk_pct?: number;
  prior_risk_pct?: number;
}
export interface BrainLoop {
  has_run: boolean;
  last_event: BrainLoopLedgerEvent | null;
  recent_ledger: BrainLoopLedgerEvent[];
  promotion_requests: BrainLoopPromotionRequest[];
  pending_operator_count: number;
  derisk_audit: BrainLoopAuditEvent[];
  source: Record<string, string>;
}

// GET /api/learning_loop — market-closed continual-learning readout. Backing
// job scripts/offline_learning_cycle.py drains Tier-7 retrain_requests
// markers and walk-forward-gates a risk/calibration model update
// (never directional alpha — see src/training/incremental/risk_calibration_learner.py).
export interface LearningLoopCycle {
  timestamp?: string;
  markers_drained?: number;
  new_outcomes?: number;
  decision: string; // "accepted" | "rejected" | "no_new_data"
  incumbent_brier?: number | null;
  candidate_brier?: number | null;
  max_size_multiplier?: number | null;
  summary?: string;
}
export interface LearningLoop {
  has_run: boolean;
  pending_retrain_markers: number;
  last_cycle: LearningLoopCycle | null;
  recent_cycles: LearningLoopCycle[];
  scope: string;
  // SHADOW/advisory: no live sizing path consumes the learned model yet.
  consumed_by_live_sizing: boolean;
  source: Record<string, string>;
}

export interface ActivityEvent {
  timestamp?: string;
  level?: string;
  message?: string;
  details?: Record<string, unknown>;
}
export interface ActivityItem {
  id: string;
  kind: string;
  title: string;
  source: string;
  status: string;
  started_at?: string | null;
  updated_at?: string | null;
  ended_at?: string | null;
  summary?: string | null;
  error?: string | null;
  metadata?: Record<string, unknown>;
  events?: ActivityEvent[];
}
export interface BackgroundActivity {
  updated_at?: string | null;
  active_count: number;
  history_count: number;
  active: ActivityItem[];
  history: ActivityItem[];
  error?: string;
}
export interface ActivityLogFeed {
  id: string;
  title: string;
  path: string;
  exists: boolean;
  updated_at?: string | null;
  age_s?: number | null;
  size_bytes?: number | null;
  lines: string[];
}
export interface ActivitySnapshot {
  updated_at: string;
  background: BackgroundActivity;
  log_feeds: ActivityLogFeed[];
}

// GET /api/axiom_operator — subscription-backed Claude Code operator session.
export interface AxiomOperatorSession {
  session_id?: string;
  status: "idle" | "running" | "acted" | "held" | "waiting" | "handoff" | "unavailable" | string;
  provider: string;
  epoch: number;
  context_budget_chars?: number;
  rollover_threshold?: number;
  context_used_chars?: number;
  context_used_pct?: number;
  handoff_summary?: string | null;
  open_incidents: string[];
  last_tools: string[];
  blocked_reasons: string[];
  next_action?: string | null;
  last_action?: string | null;
  last_reasoning?: string | null;
  last_error?: string | null;
  last_effect?: Record<string, unknown> | null;
  last_started_at?: string | null;
  last_completed_at?: string | null;
  updated_at?: string | null;
  api_key_refused?: boolean;
  cli_available?: boolean | null;
  mcp_config?: string | null;
  mode?: string;
}
export interface AxiomOperatorDecision {
  ts?: string;
  epoch?: number;
  status?: string;
  action?: string;
  params?: Record<string, unknown>;
  policy?: {
    allowed?: boolean;
    tier?: string;
    requires_human?: boolean;
    reason?: string;
  };
  tools_called?: string[];
  blocked_reasons?: string[];
  context_used_chars?: number;
  context_used_pct?: number;
  rollover?: boolean;
  duration_seconds?: number;
  returncode?: number;
  handoff_summary?: string;
  next_action?: string;
}
export interface AxiomOperator {
  has_run: boolean;
  session: AxiomOperatorSession;
  last_decision: AxiomOperatorDecision | null;
  recent_decisions: AxiomOperatorDecision[];
  source: Record<string, string>;
}

export interface AxiomOperatorRunResult {
  ok: boolean;
  status: string;
  epoch: number;
  action: string;
  error?: string | null;
  decision?: AxiomOperatorDecision;
  session?: AxiomOperatorSession;
}

// GET /api/crypto_momentum — crypto XS-momentum SHADOW lane (no live path exists;
// this signal FAILED the ship gate on significance — see docs/experiment-crypto-
// edge-hunt-round2-2026-06-29.md — the panel exists to show the accumulating
// live-forward-OOS record, not a verified edge).
export interface CryptoMomentumBook {
  longs: Record<string, number>;
  shorts: Record<string, number>;
}
export interface CryptoMomentumCycle {
  cycle_ts: string;
  asof_date: string;
  n_longs: number;
  n_shorts: number;
  gross_leverage: number;
  today_net_return: number;
  today_price_return: number;
  today_carry_return: number;
  today_cost: number;
  today_turnover: number;
  cumulative_shadow_return: number;
  forward_cycle_seq: number;
  orders_placed: number;
}
export interface CryptoMomentumConstruction {
  signal: string;
  direction: number;
  quintile: number;
  vol_target_ann: number;
  vol_window_d: number;
  max_leverage: number;
  rebalance_days: number;
  cost_bps: number;
  source_doc: string;
  pre_registered_oos_sharpe: number;
  gate_verdict: string;
}
export interface CryptoMomentumLiveGate {
  available: boolean;
  armed: boolean;
  last_event?: string | null;
  last_event_reason?: string | null;
}
export interface CryptoMomentum {
  has_run: boolean;
  n_forward_cycles: number;
  current_book: CryptoMomentumBook | null;
  current_asof: string | null;
  current_gross_leverage: number | null;
  cumulative_shadow_return: number;
  forward_sharpe_annualized: number | null;
  first_asof_date: string | null;
  last_asof_date: string | null;
  recent_cycles: CryptoMomentumCycle[];
  construction: CryptoMomentumConstruction | null;
  live_gate: CryptoMomentumLiveGate;
  mode: "live" | "shadow";
  source: Record<string, string>;
}

// GET /api/crypto_carry — crypto cash-and-carry SHADOW lane (long spot proxy /
// short perpetual, positive-funding-only, delta-neutral by construction; no
// live path exists). This signal FAILED the ship gate on turnover cost — see
// docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md — the panel exists to
// show the accumulating live-forward-OOS record, not a verified edge. This is
// a risk premium with a real exchange-solvency/liquidation tail, not free
// money — see `risk_premium_note`.
export interface CryptoCarryBook {
  longs: Record<string, number>;
  shorts: Record<string, number>;
}
export interface CryptoCarryCycle {
  cycle_ts: string;
  asof_date: string;
  n_longs: number;
  n_shorts: number;
  gross_leverage: number;
  today_net_return: number;
  today_price_return: number;
  today_carry_return: number;
  today_cost: number;
  today_turnover: number;
  cumulative_shadow_return: number;
  forward_cycle_seq: number;
  orders_placed: number;
}
export interface CryptoCarryConstruction {
  signal: string;
  lookback_d: number;
  cost_bps_roundtrip: number;
  vol_target_ann: number;
  vol_window_d: number;
  max_leverage: number;
  rebalance_days: number;
  source_doc: string;
  gate_verdict: string;
  risk_premium_note: string;
}
export interface CryptoCarryLiveGate {
  available: boolean;
  armed: boolean;
  last_event?: string | null;
  last_event_reason?: string | null;
}
export interface CryptoCarry {
  has_run: boolean;
  n_forward_cycles: number;
  current_book: CryptoCarryBook | null;
  current_asof: string | null;
  current_gross_leverage: number | null;
  cumulative_shadow_return: number;
  forward_sharpe_annualized: number | null;
  first_asof_date: string | null;
  last_asof_date: string | null;
  recent_cycles: CryptoCarryCycle[];
  construction: CryptoCarryConstruction | null;
  risk_premium_note: string | null;
  live_gate: CryptoCarryLiveGate;
  mode: "live" | "shadow";
  source: Record<string, string>;
}

// GET /api/track_b — SEC filing-text research-alpha SHADOW lane (no live path
// exists; this signal is self-labeled overall_verdict=INSUFFICIENT — UNDERPOWERED,
// not a measured NO EDGE, see docs/adversarial-review-no-edge-verdicts-2026-07-02.md
// — the panel exists to show the accumulating live-forward-OOS record against the
// frozen construction, not a verified edge).
export interface TrackBBook {
  longs: Record<string, number>;
  shorts: Record<string, number>;
}
export interface TrackBCycle {
  cycle_ts: string;
  asof_date: string;
  n_scored_filings: number;
  n_longs: number;
  n_shorts: number;
  gross_leverage: number;
  today_net_return: number;
  today_gross_return: number;
  today_cost: number;
  today_turnover: number;
  cumulative_shadow_return: number;
  forward_cycle_seq: number;
  orders_placed: number;
}
export interface TrackBConstruction {
  signal: string;
  composite_weights: Record<string, number>;
  book: string;
  quintile_divisor: number;
  min_names_for_quintile: number;
  rebalance_step_days: number;
  vol_target_ann: number;
  dd_soft: number;
  dd_hard: number;
  max_leverage: number;
  cost_bps: number;
  model_cutoff: string;
  n_scored_filings: number;
  power_required_n_filings: number;
  source_prereg_doc: string;
  source_scaleup_doc: string;
  source_adversarial_review_doc: string;
  gate_verdict: string;
  coverage_note: string;
}
export interface TrackBLiveGate {
  available: boolean;
  armed: boolean;
  last_event?: string | null;
  last_event_reason?: string | null;
}
export interface TrackB {
  has_run: boolean;
  n_forward_cycles: number;
  n_scored_filings: number | null;
  current_book: TrackBBook | null;
  current_asof: string | null;
  current_gross_leverage: number | null;
  cumulative_shadow_return: number;
  forward_sharpe_annualized: number | null;
  first_asof_date: string | null;
  last_asof_date: string | null;
  recent_cycles: TrackBCycle[];
  construction: TrackBConstruction | null;
  live_gate: TrackBLiveGate;
  mode: "live" | "shadow";
  source: Record<string, string>;
}

// GET /api/hedge — AXIOM Hedge Layer readout, SHADOW / ANALYSIS-ONLY (places no
// orders, unhalts nothing, mutates no broker; every row carries paper_only /
// runtime_allowed flags from its writer). Backed by data_sources.read_hedge():
//   - live_fx_exposure: computed live each request from the open OANDA trend
//     book (persist=False) — the "one USD bet wearing three outfits" detector;
//     null when the book is flat / stale / uncomputable (honest empty state).
//   - exposure_history: the training bridge counter (0 until the capture loop
//     starts — an explicit "not started" state, not an error).
//   - recent_cycles: raw-vs-hedged ledger rows; FX hedged return_basis is often
//     "unresolved" (no FX forward-price source yet — surfaced, not hidden).
// Every nested value below that is an object/array is rendered field-extracted
// or .map()ed, NEVER dropped into JSX directly ("Objects are not valid as a
// React child" has crashed this dashboard before — see HedgePanel guards).
export interface HedgeLeg {
  instrument: string;
  direction: string; // "long" | "short"
  risk_home: number;
}
export interface HedgeProposal {
  style: string; // "direct_offset" | "relative_value" | "market" | "sector"
  legs: HedgeLeg[];
  neutralizes: string; // which exposure bucket this targets, e.g. "currency:USD"
  isolates: string; // one-line narrative: what view survives the hedge
  exposure_reduction: number; // dollar risk_home removed from the flagged bucket
  exposure_reduction_pct: number | null;
  correlation_to_target: number | null;
  cost_known: boolean;
  cost_estimate_dollars: number | null;
  cost_to_reduction_ratio: number | null;
  cost_source: string;
  cost_confidence: number;
  fail_closed: boolean;
  reasons: string[];
}
export interface HedgeLiveFxExposure {
  asof_date: string | null;
  nav: number | null;
  net_currency_exposure: Record<string, number>; // signed home-$ notional per currency
  net_correlation_bucket_exposure: Record<string, number>;
  concentration_warnings: string[];
  narrative: string[];
  position_count: number | null;
  resolved_position_count: number | null;
  raw_net_return: number | null;
  raw_return_basis: string | null;
  hedge_status: string | null; // "applied" | "no_valid_hedge" | "fail_closed" | "unsupported_asset_class"
  hedge_decision: string | null;
  applied_hedge: HedgeProposal | null; // OBJECT — never render raw; extract named fields
  hedged_return_basis: string; // often "unresolved" for FX (no forward-price source yet)
}
export interface HedgeExposureHistory {
  n_snapshots: number;
  first_captured_at: string | null;
  last_captured_at: string | null;
  last_nav: number | null;
  last_net_currency: Record<string, number> | null; // OBJECT — never render raw
  note: string;
}
// One raw-vs-hedged ledger row. raw/hedge/hedged/exposure are OBJECTS — only the
// named scalar fields below get rendered; the rest stay opaque on purpose.
export interface HedgeLedgerCycle {
  cycle_ts: string;
  strategy: string;
  asset_class: string;
  asof_date: string;
  notional: number;
  raw: { net_return: number | null; gross_return?: number | null };
  hedge: { status: string | null; decision: string | null };
  hedged: { return_basis: string; net_return: number | null; gross_return: number | null };
  exposure: Record<string, unknown>;
  paper_only?: boolean;
  runtime_allowed?: boolean;
  human_review_required?: boolean;
}
export interface Hedge {
  live_fx_exposure: HedgeLiveFxExposure | null;
  scorecards: Record<string, unknown>; // per-strategy objects — not rendered raw
  strategies_with_history: string[];
  recent_cycles: HedgeLedgerCycle[];
  n_ledger_cycles: number;
  exposure_history: HedgeExposureHistory;
  paper_only: boolean;
  runtime_allowed: boolean;
  human_review_required: boolean;
  source: Record<string, string>;
}

// GET /api/mind_window — resident reasoning loop (src/agent_runtime/loop.py)
// "mind-window": what the loop NOTICED (observations), BELIEVES (diagnosis),
// DID-OR-WOULD-DO (outcomes), and NEEDS FROM THE OPERATOR (pending_operator_
// proposals). Has not run in production yet (has_run=false) — render the
// honest empty state, never fabricate a cycle.
export type MindWindowTier = "operational" | "deescalation" | "escalation" | string;

export interface MindWindowObservation {
  ok: boolean;
  data?: unknown;
  error?: string;
}
export interface MindWindowProposedAction {
  action: string;
  params?: Record<string, unknown>;
  rationale?: string;
  tier?: MindWindowTier;
}
export interface MindWindowDiagnosis {
  available: boolean;
  reasoning?: string | null;
  beliefs?: string | null;
  proposed_actions?: MindWindowProposedAction[];
}
export interface MindWindowOutcome {
  action: string;
  tier?: MindWindowTier;
  executed: boolean;
  shadow: boolean;
  proposal: boolean;
  denied: boolean;
  detail?: Record<string, unknown>;
  proposal_id?: string | null;
}
export interface ProposalDisposition {
  proposal_id: string;
  status: "accepted" | "denied";
  actor: string;
  reason?: string;
  detail?: Record<string, unknown>;
  decided_at: string;
}
export interface MindWindowCycle {
  cycle_id: string;
  ts: string;
  actor?: string;
  cli_available?: boolean;
  autonomy_enabled?: boolean;
  observations: Record<string, MindWindowObservation>;
  diagnosis: MindWindowDiagnosis;
  proposed_actions?: MindWindowProposedAction[];
  outcomes: MindWindowOutcome[];
  verified?: boolean;
  verify_detail?: Record<string, unknown>;
}
// pending_operator_proposals flattens every escalation outcome with proposal=true
// across recent_cycles, deduped by proposal_id (same proposal re-surfacing across
// cycles shows once), prefixed with cycle_id/ts. resolved_proposals holds the ones
// the operator has already accepted/denied (joined disposition, not hidden).
export interface MindWindowPendingProposal extends MindWindowOutcome {
  cycle_id?: string;
  ts?: string;
  disposition?: ProposalDisposition | null;
}
export interface MindWindow {
  has_run: boolean;
  autonomy_enabled: boolean;
  recent_cycles: MindWindowCycle[];
  last_cycle: MindWindowCycle | null;
  pending_operator_proposals: MindWindowPendingProposal[];
  resolved_proposals?: MindWindowPendingProposal[];
  error?: string;
  source: Record<string, string>;
}

export interface StreamPayload {
  ts: number;
  account: Account;
  status: Status;
  prices: Prices;
}

// GET /api/axiom_training — Phase-M evidence cockpit. All fields are factual
// projections; nullable monitoring metrics mean the lane is not reporting.
export interface TrainingDataTier {
  tier: string;
  manifest_count: number;
  partition_count: number;
  latest_manifest: string | null;
  latest_manifest_at: string | null;
  freshness_age_seconds: number | null;
}
export interface TrainingDataDomain {
  domain: string;
  available: boolean;
  tiers: TrainingDataTier[];
}
export interface P2Readiness {
  available: boolean;
  ready: boolean;
  minimum_trading_days?: number;
  exposure_trading_days?: number;
  tick_trading_days_by_pair?: Record<string, number>;
  generated_at_utc?: string;
  age_seconds?: number | null;
  blocking_reasons?: string[];
  error?: string;
}
export interface EvidenceCheck {
  check_id?: string;
  passed?: boolean;
  details?: string | null;
}
export interface EvidencePackageView {
  family: string;
  lane_id: string;
  package_id?: string | null;
  package_digest: string;
  state?: string | null;
  sequence?: number | null;
  disposition_head_digest?: string | null;
  job_manifest_digest?: string | null;
  derived_from_package_digests?: string[];
  created_at?: string | null;
  negative_result?: boolean;
  is_champion?: boolean;
  gate_results?: EvidenceCheck[];
  verification?: {
    decision?: string | null;
    rejection_reason?: string | null;
    failed_checks?: string[];
  };
}
export interface EvidenceLaneView {
  lane_id: string;
  family: string;
  current: EvidencePackageView;
  champion?: Record<string, unknown> | null;
  prior_champion?: EvidencePackageView | null;
  package_count: number;
  negative_count: number;
}
export interface TrainingJobView {
  job_id: string;
  status: string;
  source?: string;
  lane?: string;
  lanes?: string[];
  manifest_digest?: string | null;
  container_digest?: string | null;
  resource_class?: string | null;
  resource_usage?: { wall_seconds?: number | null } | null;
  cost?: number | null;
  submitted_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
}
export interface ForwardLaneMonitor {
  lane_id: string;
  package_digest?: string | null;
  state?: string | null;
  available: boolean;
  status?: string;
  shadow_started_at?: string | null;
  shadow_age_seconds?: number | null;
  updated_at?: string | null;
  forward_sharpe?: number | null;
  drawdown?: number | null;
  turnover?: number | null;
  cost?: number | null;
  exposure?: number | null;
  baseline_deviation?: number | null;
  retirement_warnings?: string[];
}
export interface TrainingControlReadiness {
  enabled: boolean;
  operator_trust_configured: boolean;
  dataset_configured: boolean;
  dataset_sha256?: string | null;
  run_request_template?: Record<string, unknown> | null;
  run_subject_digest?: string | null;
  operator_private_key_on_server: boolean;
  actions: string[];
  invariants: string[];
}
export interface AxiomTrainingCockpit {
  connected: boolean;
  generated_at?: string;
  error?: string;
  data: {
    data_root: string;
    domains: TrainingDataDomain[];
    quality_failure_count: number;
    quality_failures: { kind: string; path: string }[];
    quality_failures_truncated: boolean;
    p2_readiness: P2Readiness;
  };
  jobs: {
    jobs: TrainingJobView[];
    counts: Record<string, number>;
    total: number;
    source: string;
  };
  evidence: {
    available: boolean;
    readers: { family: string; available: boolean; reason?: string | null }[];
    packages: EvidencePackageView[];
    lanes: EvidenceLaneView[];
    state_counts: Record<string, number>;
    negative_result_count: number;
    champions: Record<string, Record<string, unknown>>;
  };
  forward_monitoring: {
    lanes: ForwardLaneMonitor[];
    reporting_count: number;
    total: number;
  };
  controls: TrainingControlReadiness;
}
