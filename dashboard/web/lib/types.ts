// API response types — mirror the FastAPI data layer (dashboard/server).

export interface Position {
  instrument: string;
  net_units: number;
  unrealized_pl: number;
  // Optional bracket levels — present once the bot writes TP/SL into account_state.json
  // (additive, contract-safe). Absent today → rendered as "—" / no chart line.
  take_profit?: number | null;
  stop_loss?: number | null;
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
}

export interface Status {
  halted: boolean;
  mode: string | null;
  status: string | null;
  scan_cycle_count: number | null;
  environment: string;
  running: boolean;
  lane_running: boolean;
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
  tradeable: boolean;
}
export interface Prices {
  connected: boolean;
  prices: Record<string, Price>;
  error?: string;
}

export interface Trade {
  id: string;
  time: string;
  instrument: string;
  units: number;
  side: "BUY" | "SELL";
  price: number | null;
  pl: number;
  financing: number;
  half_spread_cost: number;
  reason: string;
  balance: number | null;
  tag: string | null;
}
export interface Trades { connected: boolean; source: string; count: number; trades: Trade[]; }

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
export interface Candle { time: number; open: number; high: number; low: number; close: number; }
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
  note: string;
  books: Record<string, { price: string; longCountPercent: string; shortCountPercent: string }[]>;
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
  bounded?: boolean | null;
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

export interface StreamPayload {
  ts: number;
  account: Account;
  status: Status;
  prices: Prices;
}
