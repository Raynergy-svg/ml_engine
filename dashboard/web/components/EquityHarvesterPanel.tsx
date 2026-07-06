"use client";
import type { ReactNode } from "react";
import { usePoll } from "@/lib/api";
import type { EquitySleeve, LaneStatus } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime } from "@/lib/format";

const DECISION_COLOR: Record<string, string> = {
  refuse: "#ff4d6d", halt: "#ff4d6d", no_act: "#f5b14c", abstain: "#f5b14c", continue: "#2bd17e",
};

function Row({ label, value, mono = true }: { label: string; value: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between border-t py-1.5 hairline first:border-t-0">
      <span className="font-mono text-[11px] text-faint">{label}</span>
      <span className={`text-[12px] text-text ${mono ? "font-mono tnum" : ""}`}>{value}</span>
    </div>
  );
}

export function EquityHarvesterPanel() {
  const { data, loading } = usePoll<EquitySleeve>("/api/equity_sleeve", 5000);
  const { data: lanes } = usePoll<LaneStatus>("/api/lanes", 4000);
  const laneHalted = lanes?.lanes?.equity ?? null;

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Equity Harvester</SectionTitle><Loading /></Card>;
  }
  if (!data) {
    return <Card className="p-3"><SectionTitle>Equity Harvester</SectionTitle><NotConnected label="No rebalance state on disk yet" /></Card>;
  }

  const gate = data.gate_decision;
  const gateColor = gate?.decision ? DECISION_COLOR[gate.decision] ?? "#5a6677" : "#5a6677";
  const sg = data.ship_gate;
  const lg = data.live_gate;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="flex flex-col">
        <SectionTitle right={
          <span className="flex items-center gap-1.5">
            <Badge color={laneHalted === true ? "#ff4d6d" : laneHalted === false ? "#2bd17e" : "#5a6677"} dot>
              lane {laneHalted === true ? "HALTED" : laneHalted === false ? "unhalted" : "—"}
            </Badge>
            <Badge color={data.mode === "live" ? "#ff4d6d" : "#22d3ee"} dot>
              {data.mode === "live" ? "LIVE (armed)" : "SHADOW"}
            </Badge>
          </span>
        }>
          Equity Harvester · lane status
        </SectionTitle>
        <div className="p-3">
          <Row label="Runner mode" value={data.mode === "live" ? "live (IBKR paper, armed)" : "shadow (no broker contact)"} />
          <Row label="LiveGate armed" value={lg.armed ? "YES" : "no"} />
          {lg.available && (
            <>
              <Row label="Last gate event" value={lg.last_event ?? "—"} />
              <Row label="Event reason" value={lg.last_event_reason ?? "—"} mono={false} />
              <Row label="Event time" value={lg.last_event_ts ? shortTime(new Date(lg.last_event_ts * 1000).toISOString()) : "—"} />
            </>
          )}
          {!lg.available && (
            <div className="py-2 font-mono text-[10.5px] text-faint">
              No live_gate_state.json on disk — harvester has never been armed for live execution.
            </div>
          )}
          <Row label="Rebalance asof" value={data.asof ?? "—"} />
          <Row label="Rebalance id" value={data.rebalance_id ?? "—"} />
          <Row label="Universe size" value={Object.keys(data.target_weights ?? {}).length} />
        </div>
      </Card>

      <Card className="flex flex-col">
        <SectionTitle right={
          <Badge color={gateColor} dot>{gate?.available ? (gate.decision ?? "—").toUpperCase() : "UNAVAILABLE"}</Badge>
        }>
          Risk gate + rails · this cycle
        </SectionTitle>
        <div className="p-3">
          {!gate?.available ? (
            <NotConnected label="Gate verdict unavailable" hint={gate?.error} />
          ) : (
            <>
              <Row label="Decision" value={gate.decision} />
              <Row label="Actionable" value={gate.actionable ? "yes" : "no"} />
              <Row label="Reasons" value={gate.reasons?.length ? gate.reasons.join(", ") : "none"} mono={false} />
            </>
          )}
          <div className="mt-2 border-t pt-2 hairline">
            <div className="mb-1 font-mono text-[10.5px] text-faint">Ship gate (US-007 canonical bar)</div>
            {!sg.available ? (
              <NotConnected label="No SHIP_GATE.json recorded" />
            ) : (
              <>
                <Row label="Gate pass" value={sg.gate_pass ? "PASS" : "FAIL"} />
                <Row label="Net Sharpe" value={sg.net_sharpe?.toFixed(3) ?? "—"} />
                <Row label="Max drawdown" value={sg.max_dd != null ? `${(sg.max_dd * 100).toFixed(1)}%` : "—"} />
                <Row label="As of" value={sg.asof ?? "—"} />
              </>
            )}
          </div>
        </div>
      </Card>

      <Card className="flex flex-col lg:col-span-2">
        <SectionTitle>Recent cycle-ledger decisions</SectionTitle>
        <div className="p-3">
          {!data.last_cycles?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">No cycles recorded yet.</div>
          ) : (
            <div className="font-mono text-[11px]">
              {data.last_cycles.map((c) => (
                <div key={c.seq} className="flex items-center gap-2 border-t py-1.5 hairline first:border-t-0">
                  <StatusDot color={DECISION_COLOR[c.decision] ?? "#5a6677"} />
                  <span className="w-10 shrink-0 text-faint">#{c.seq}</span>
                  <span className="w-20 shrink-0 uppercase" style={{ color: DECISION_COLOR[c.decision] ?? "#8b98a9" }}>{c.decision}</span>
                  <span className="min-w-0 flex-1 truncate text-faint">{c.reasons?.join(", ") || "—"}</span>
                  <span className="shrink-0 text-faint">{c.n_orders} orders</span>
                  <span className="shrink-0 text-[10px] text-faint tnum">{shortTime(c.asof)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
