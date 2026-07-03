// Shared lane metadata — the StateEngine KNOWN_LANES, labeled for display.
// Keep in sync with src/scanner/automation/state_engine.py:KNOWN_LANES.
export const LANE_META: Record<string, { label: string; hint: string }> = {
  oanda_fx: { label: "OANDA FX", hint: "trend lane — SMA(100) directional trades" },
  equity: { label: "Equity harvester", hint: "single-stock harvester — shadow unless armed live" },
  brain: { label: "Brain loop", hint: "Sonnet autonomous research → promotion loop" },
  crypto_momentum: { label: "Crypto momentum", hint: "XS-momentum shadow lane — observation only, no LiveGate arm exists" },
  track_b: { label: "Track B (filing-text)", hint: "SEC filing-text research-alpha shadow lane — underpowered, no LiveGate arm exists" },
};

export const LANE_ORDER = ["oanda_fx", "equity", "brain", "crypto_momentum", "track_b"] as const;
