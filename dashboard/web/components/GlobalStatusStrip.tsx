"use client";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { ApiHealth, SystemHealth } from "@/lib/types";
import { StatusDot } from "./ui";
import { ago } from "@/lib/format";

const SEV_COLOR: Record<string, string> = {
  INFO: "#22d3ee", WARNING: "#f5b14c", CRITICAL: "#ff4d6d", ALARM: "#ff4d6d",
};

function Pill({ color, label, value, title }: { color: string; label: string; value: string; title?: string }) {
  return (
    <div className="flex items-center gap-2 whitespace-nowrap" title={title}>
      <StatusDot color={color} />
      <span className="eyebrow">{label}</span>
      <span className="font-mono text-[11px] font-semibold" style={{ color }}>{value}</span>
    </div>
  );
}

export function GlobalStatusStrip() {
  const { data } = usePoll<SystemHealth>("/api/system_health", 5000);
  const { data: apiHealth } = usePoll<ApiHealth>("/api/health", 10000);
  const { payload } = useStream();
  const halted = payload?.status?.halted ?? null;

  const lane = data?.lanes;
  const laneRunning = lane?.running === true;
  const gates = data?.gates;
  const gatesGreen = gates?.status === "GREEN";
  const alerts = data?.alerts;
  const sev = alerts?.max_severity;

  return (
    <div className="card flex flex-wrap items-center gap-x-6 gap-y-2 rounded-none border-x-0 border-t-0 px-6 py-2">
      <Pill
        color={laneRunning ? "#2bd17e" : "#f5b14c"}
        label="Live lane"
        value={laneRunning ? "RUNNING" : "OFFLINE"}
        title={lane?.available
          ? `oanda_trend_proc=${lane.oanda_trend_proc} · account snapshot ${ago(lane.account_state_age_s)}`
          : "oracle unavailable"}
      />
      <Pill
        color={halted === false ? "#2bd17e" : halted === true ? "#ff4d6d" : "#5a6677"}
        label="Halt"
        value={halted === true ? "HALTED" : halted === false ? "CLEAR" : "—"}
        title={halted === true ? "trading halted" : halted === false ? "not halted — trading allowed" : ""}
      />
      <Pill
        color={gatesGreen ? "#2bd17e" : gates?.status === "RED" ? "#ff4d6d" : "#5a6677"}
        label="Gates"
        value={gates?.available ? gates.status : "—"}
        title={gates?.available ? `${gates.hard_no_count} Hard-NO checks · verdict ${ago(gates.verdict_age_s ?? undefined)}` : ""}
      />
      <Pill
        color={alerts?.count ? (SEV_COLOR[sev ?? "WARNING"] ?? "#f5b14c") : "#2bd17e"}
        label="Alerts"
        value={alerts?.count ? `${alerts.count} ${sev ?? ""}`.trim() : "none"}
      />
      <div
        className="ml-auto flex items-center gap-2 whitespace-nowrap"
        title={apiHealth?.control_enabled
          ? "OANDA fxPractice (demo) — bounded control enabled"
          : "OANDA fxPractice (demo) — read-only"}
      >
        <StatusDot color={apiHealth?.control_enabled ? "#f5b14c" : "#22d3ee"} />
        <span className="font-mono text-[10px] text-faint">
          PRACTICE · {apiHealth?.control_enabled ? "control enabled" : "read-only"}
        </span>
      </div>
    </div>
  );
}
