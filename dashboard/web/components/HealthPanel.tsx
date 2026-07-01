"use client";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { SystemHealth } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { ago, shortTime } from "@/lib/format";

const SEV_COLOR: Record<string, string> = {
  INFO: "#22d3ee", WARNING: "#f5b14c", CRITICAL: "#ff4d6d", ALARM: "#ff4d6d",
};

function Row({ ok, label, detail }: { ok: boolean; label: string; detail?: string }) {
  return (
    <div className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0">
      <StatusDot color={ok ? "#2bd17e" : "#ff4d6d"} />
      <div className="min-w-0 flex-1">
        <span className="font-mono text-[12px] text-text">{label}</span>
        {detail && <div className="text-[10.5px] text-faint">{detail}</div>}
      </div>
      <span className="font-mono text-[10px]" style={{ color: ok ? "#2bd17e" : "#ff4d6d" }}>
        {ok ? "OK" : "FAIL"}
      </span>
    </div>
  );
}

export function HealthPanel() {
  const { data, loading } = usePoll<SystemHealth>("/api/system_health", 5000);
  const { payload } = useStream();
  const halted = payload?.status?.halted ?? null;
  const lane = data?.lanes;
  const gates = data?.gates;
  const alerts = data?.alerts;

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>System Health</SectionTitle><Loading /></Card>;
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* Lanes oracle */}
      <Card className="flex flex-col">
        <SectionTitle right={
          <Badge color={lane?.running ? "#2bd17e" : "#f5b14c"} dot pulse={lane?.running}>
            {lane?.running ? "LIVE LANE RUNNING" : "LIVE LANE OFFLINE"}
          </Badge>
        }>
          Lanes · running oracle
        </SectionTitle>
        <div className="p-3 font-mono text-[12px]">
          {!lane?.available ? (
            <NotConnected label="Oracle unavailable" />
          ) : (
            <>
              <div className="mb-2 text-[11px] text-dim">OANDA trend lane (the live bot):</div>
              <Row ok={!!lane.oanda_trend_proc} label="trend process alive" />
              <Row ok={!!lane.account_state_fresh} label="account snapshot fresh"
                   detail={`written ${ago(lane.account_state_age_s)}`} />
              <Row ok={!!lane.tier7_heartbeat_fresh} label="tier7 heartbeat fresh"
                   detail={`age ${ago(lane.tier7_heartbeat_age_s)} · pid_alive ${lane.tier7_pid_alive}`} />
              <div className="mt-3 flex items-start gap-2 rounded-md border p-2 hairline">
                <StatusDot color="#5a6677" />
                <span className="text-[10.5px] text-faint">
                  Harvester lane (legacy IBKR): <span className="text-dim">dormant — superseded</span>.
                  running:NO is expected, not a fault.
                </span>
              </div>
            </>
          )}
        </div>
      </Card>

      {/* Gates */}
      <Card className="flex flex-col">
        <SectionTitle right={
          <Badge color={gates?.status === "GREEN" ? "#2bd17e" : gates?.status === "RED" ? "#ff4d6d" : "#5a6677"} dot>
            {gates?.available ? gates.status : "—"}
          </Badge>
        }>
          Gates · Hard-NO checks (verify_gate)
        </SectionTitle>
        <div className="p-3">
          {!gates?.available ? (
            <NotConnected label="No verdict recorded" hint="verify_gate hasn't written a verdict yet." />
          ) : (
            <>
              <div className="font-mono text-[12px]">
                {gates.checks.map((c) => (
                  <Row key={c.name} ok={c.ok} label={c.name} detail={c.detail} />
                ))}
              </div>
              <div className="mt-2 font-mono text-[10px] text-faint">
                {gates.hard_no_count} Hard-NO rails · verdict {ago(gates.verdict_age_s ?? undefined)}
                {" · "}halt {halted === true ? "ON" : halted === false ? "off" : "—"}
              </div>
            </>
          )}
        </div>
      </Card>

      {/* Alerts */}
      <Card className="flex flex-col lg:col-span-2">
        <SectionTitle right={
          <Badge color={alerts?.count ? (SEV_COLOR[alerts.max_severity ?? "WARNING"] ?? "#f5b14c") : "#2bd17e"} dot>
            {alerts?.count ? `${alerts.count} ACTIVE` : "ALL CLEAR"}
          </Badge>
        }>
          AlertManager · active alerts
        </SectionTitle>
        <div className="p-3">
          {!alerts?.active?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">No active alerts.</div>
          ) : (
            <div className="font-mono text-[12px]">
              {alerts.active.map((a, i) => (
                <div key={i} className="flex items-start gap-2 border-t py-2 hairline first:border-t-0">
                  <StatusDot color={SEV_COLOR[a.severity] ?? "#f5b14c"} />
                  <div className="min-w-0 flex-1">
                    <span style={{ color: SEV_COLOR[a.severity] ?? "#f5b14c" }}>{a.severity}</span>
                    <span className="text-text"> · {a.message}</span>
                    {a.pair && <span className="text-faint"> · {a.pair}</span>}
                  </div>
                  <span className="shrink-0 text-[10px] text-faint tnum">{shortTime(a.timestamp)}</span>
                </div>
              ))}
            </div>
          )}
          {alerts?.last_updated && (
            <div className="mt-2 font-mono text-[10px] text-faint">updated {shortTime(alerts.last_updated)}</div>
          )}
        </div>
      </Card>
    </div>
  );
}
