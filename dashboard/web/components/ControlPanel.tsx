"use client";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import { control, type ControlResult } from "@/lib/control";
import type { SystemHealth } from "@/lib/types";
import { Card, SectionTitle, Badge, StatusDot } from "./ui";
import { shortTime } from "@/lib/format";

interface AuditEntry {
  ts: string; action: string; allowed?: boolean; reason?: string; result?: string;
  params?: Record<string, unknown>;
}
interface AuditResp { entries: AuditEntry[]; count: number }

/** Two-step confirm button. Click → "Confirm?" → executes. Auto-disarms. */
function ConfirmButton({
  label, color, onRun, disabled,
}: { label: string; color: string; onRun: () => Promise<ControlResult>; disabled?: boolean }) {
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<ControlResult | null>(null);

  async function go() {
    setBusy(true);
    const r = await onRun();
    setRes(r);
    setBusy(false);
    setArmed(false);
    setTimeout(() => setRes(null), 6000);
  }

  if (armed) {
    return (
      <span className="inline-flex items-center gap-1">
        <button onClick={go} disabled={busy}
          className="rounded-md px-2.5 py-1 font-mono text-[11px] font-semibold text-base"
          style={{ background: color }}>
          {busy ? "…" : "Confirm"}
        </button>
        <button onClick={() => setArmed(false)} className="rounded-md border px-2 py-1 font-mono text-[11px] text-faint hairline">
          cancel
        </button>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2">
      <button onClick={() => setArmed(true)} disabled={disabled || busy}
        className="rounded-md border px-2.5 py-1 font-mono text-[11px] hairline disabled:opacity-40"
        style={{ color }}>
        {label}
      </button>
      {res && (
        <span className="font-mono text-[10px]" style={{ color: res.ok ? "#2bd17e" : "#ff4d6d" }}>
          {res.ok ? String(res.data.result ?? "ok") : `${res.status}: ${String(res.data.detail ?? res.data.error ?? "denied")}`}
        </span>
      )}
    </span>
  );
}

export function ControlPanel() {
  const { data: audit, error } = usePoll<AuditResp>("/api/control/audit?limit=25", 4000);
  const { data: health } = usePoll<SystemHealth>("/api/system_health", 5000);
  const { payload } = useStream();
  const [lev, setLev] = useState(3);

  const halted = payload?.status?.halted ?? null;
  const laneRunning = health?.lanes?.running === true;
  const disabled = !!error && /404/.test(error);

  if (disabled) {
    return (
      <Card className="p-5">
        <SectionTitle right={<Badge color="#5a6677" dot>CONTROL DISABLED</Badge>}>Operator Control</SectionTitle>
        <div className="py-6 text-center font-mono text-[12px] text-dim">
          The control layer is built but <span className="text-warn">disabled</span> (AXIOM_CONTROL_ENABLED off).
          <div className="mt-1 text-[11px] text-faint">Start the data layer with AXIOM_CONTROL_ENABLED=1 to operate the bot.</div>
        </div>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Card className="p-4">
        <SectionTitle right={<Badge color="#f5b14c" dot>WRITE PATH · practice · audited</Badge>}>
          Operator Control · bounded actions
        </SectionTitle>
        <div className="mb-3 flex flex-wrap items-center gap-4 px-1 font-mono text-[11px]">
          <span className="flex items-center gap-1.5">
            <StatusDot color={halted === true ? "#ff4d6d" : "#2bd17e"} />
            halt {halted === true ? "ON" : halted === false ? "off" : "—"}
          </span>
          <span className="flex items-center gap-1.5">
            <StatusDot color={laneRunning ? "#2bd17e" : "#f5b14c"} pulse={laneRunning} />
            trend lane {laneRunning ? "running" : "offline"}
          </span>
          <span className="text-faint">every action: explicit confirm · server re-enforces practice + caps</span>
        </div>

        <div className="grid gap-2.5 sm:grid-cols-2">
          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Halt trading</div>
              <div className="font-mono text-[10px] text-faint">fail-safe — stops new orders</div></div>
            <ConfirmButton label="HALT" color="#ff4d6d" onRun={() => control("halt")} disabled={halted === true} />
          </div>
          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Unhalt trading</div>
              <div className="font-mono text-[10px] text-faint">gated: drawdown &lt; 20% · gates GREEN · practice</div></div>
            <ConfirmButton label="UNHALT" color="#2bd17e" onRun={() => control("unhalt")} disabled={halted === false} />
          </div>
          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Trend loop</div>
              <div className="font-mono text-[10px] text-faint">start / stop the OANDA trend lane</div></div>
            <span className="flex gap-1.5">
              <ConfirmButton label="START" color="#2bd17e" onRun={() => control("start_loop", { loop: "trend" })} />
              <ConfirmButton label="STOP" color="#ff4d6d" onRun={() => control("stop_loop", { loop: "trend" })} />
            </span>
          </div>
          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Tier 7 loop</div>
              <div className="font-mono text-[10px] text-faint">start / stop the self-heal loop</div></div>
            <span className="flex gap-1.5">
              <ConfirmButton label="START" color="#2bd17e" onRun={() => control("start_loop", { loop: "tier7" })} />
              <ConfirmButton label="STOP" color="#ff4d6d" onRun={() => control("stop_loop", { loop: "tier7" })} />
            </span>
          </div>
          <div className="flex items-center justify-between rounded-md border p-3 hairline sm:col-span-2">
            <div><div className="font-mono text-[12px] text-text">Gross leverage</div>
              <div className="font-mono text-[10px] text-faint">total exposure × NAV · hard-capped 15× server-side</div></div>
            <span className="flex items-center gap-2">
              <input type="range" min={0} max={15} step={0.5} value={lev}
                onChange={(e) => setLev(parseFloat(e.target.value))} className="w-32" />
              <span className="w-10 font-mono text-[13px] font-semibold text-cyan tnum">{lev.toFixed(1)}×</span>
              <ConfirmButton label="SET" color="#22d3ee" onRun={() => control("set_gross_leverage", { gross_leverage: lev })} />
            </span>
          </div>
        </div>
      </Card>

      {/* audit log */}
      <Card className="flex flex-col p-3">
        <SectionTitle right={audit ? <span className="font-mono text-[10px] text-faint tnum">{audit.count} total</span> : undefined}>
          Control audit trail
        </SectionTitle>
        {!audit?.entries?.length ? (
          <div className="py-3 text-center font-mono text-[11px] text-faint">No control actions yet.</div>
        ) : (
          <div className="scroll-thin max-h-[280px] overflow-auto font-mono text-[11px]">
            {audit.entries.map((e, i) => (
              <div key={i} className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0">
                <StatusDot color={e.allowed ? "#2bd17e" : "#ff4d6d"} />
                <div className="min-w-0 flex-1">
                  <span className="text-text">{e.action}</span>
                  <span className="text-faint"> · {e.result ?? (e.allowed ? "ok" : "denied")}</span>
                  {e.reason && e.allowed === false && <div className="truncate text-[10px] text-neg" title={e.reason}>{e.reason}</div>}
                </div>
                <span className="shrink-0 text-[10px] text-faint tnum">{shortTime(e.ts)}</span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
