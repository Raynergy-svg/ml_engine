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

export interface StreamPayload {
  ts: number;
  account: Account;
  status: Status;
  prices: Prices;
}
