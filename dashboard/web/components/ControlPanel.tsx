"use client";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import { control, type ControlResult } from "@/lib/control";
import type { ControlState, SystemHealth, Tier7 } from "@/lib/types";
import { LANE_META } from "@/lib/lanes";
import { Card, SectionTitle, Badge, StatusDot } from "./ui";
import { ago, shortTime } from "@/lib/format";

interface AuditEntry {
  ts: string; action: string; allowed?: boolean; reason?: string; result?: string;
  params?: Record<string, unknown>;
  /** Attribution (2026-07-01+): absent on entries written before the actor-attribution
   * fix — render "—" rather than a fabricated identity for those older rows. */
  actor?: string;
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
  const { data: audit } = usePoll<AuditResp>("/api/control/audit?limit=25", 4000);
  const { data: polledControlState, error: stateError } = usePoll<ControlState>("/api/control/state", 3000);
  const { data: tier7 } = usePoll<Tier7>("/api/tier7", 4000);
  const { data: health } = usePoll<SystemHealth>("/api/system_health", 5000);
  const { payload } = useStream();
  const [actionControlState, setActionControlState] = useState<ControlState | null>(null);
  const [pendingLev, setPendingLev] = useState<number | null>(null);

  async function runControl(action: string, params: Record<string, unknown> = {}) {
    const result = await control(action, params);
    if (result.data.state) {
      setActionControlState(result.data.state);
      if (typeof result.data.state.gross_leverage === "number") setPendingLev(null);
    }
    return result;
  }

  const controlState = polledControlState ?? actionControlState;
  const halted = controlState?.halted ?? payload?.status?.halted ?? null;
  const laneRunning = controlState?.loops?.trend?.running ?? health?.lanes?.running === true;
  const tier7Proc = controlState?.loops?.tier7;
  const tier7ProcRunning = tier7Proc?.running === true;
  const tier7SnapshotConnected = tier7?.connected === true;
  const tier7BotRunning = tier7SnapshotConnected ? tier7?.running === true : null;
  const tier7Running = tier7BotRunning ?? tier7ProcRunning;
  const tier7Mismatch = tier7SnapshotConnected && tier7ProcRunning !== (tier7?.running === true);
  const tier7LatestHeal = [...(tier7?.self_heal?.recent_events ?? [])].reverse()[0];
  const tier7StatusColor = tier7Mismatch ? "#f5b14c" : tier7Running ? "#2bd17e" : "#ff4d6d";
  const disabled = !!stateError && /404/.test(stateError);
  const leverageCap = controlState?.leverage_cap ?? 15;
  const persistedLeverage = typeof controlState?.gross_leverage === "number" ? controlState.gross_leverage : null;
  const lev = pendingLev ?? persistedLeverage ?? 3;
  const armed = controlState?.armed === true;

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

        {/* ARM lockdown (2026-07-01): the server REFUSES flatten / leverage-change /
            start_loop / unhalt as a structural no-op unless armed — this banner
            reflects that, it doesn't itself enforce anything. Default DISARMED. */}
        <div className="mb-3 flex items-center justify-between rounded-md border p-2.5 hairline"
             style={{ borderColor: armed ? "#ff4d6d55" : undefined }}>
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <StatusDot color={armed ? "#ff4d6d" : "#5a6677"} pulse={armed} />
            <span className={armed ? "text-warn font-semibold" : "text-dim"}>
              {armed ? "ARMED — live-order actions enabled" : "DISARMED — flatten/leverage/start/unhalt are no-ops"}
            </span>
            {armed && controlState?.arm_expires_at && (
              <span className="text-faint">· expires {shortTime(controlState.arm_expires_at)}</span>
            )}
          </div>
          {armed ? (
            <ConfirmButton label="DISARM" color="#5a6677" onRun={() => runControl("disarm")} />
          ) : (
            <ConfirmButton label="ARM" color="#ff4d6d" onRun={() => runControl("arm")} />
          )}
        </div>

        <div className="mb-3 flex flex-wrap items-center gap-4 px-1 font-mono text-[11px]">
          <span className="flex items-center gap-1.5">
            <StatusDot color={halted === true ? "#ff4d6d" : "#2bd17e"} />
            halt {halted === true ? "ON" : halted === false ? "off" : "—"}
          </span>
          <span className="flex items-center gap-1.5">
            <StatusDot color={laneRunning ? "#2bd17e" : "#f5b14c"} pulse={laneRunning} />
            trend lane {laneRunning ? "running" : "offline"}
          </span>
          <span className="flex items-center gap-1.5">
            <StatusDot color={tier7StatusColor} pulse={tier7Running === true && !tier7Mismatch} />
            tier 7 {tier7Running === true ? "running" : tier7Running === false ? "offline" : "unknown"}
            {tier7Mismatch && <span className="text-warn">mismatch</span>}
          </span>
          <span className="text-faint">every action: explicit confirm · server re-enforces practice + caps</span>
        </div>

        <div className="grid gap-2.5 sm:grid-cols-2">
          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Halt trading</div>
              <div className="font-mono text-[10px] text-faint">fail-safe — stops new orders</div></div>
            <ConfirmButton label="HALT" color="#ff4d6d" onRun={() => runControl("halt")} disabled={halted === true} />
          </div>
          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Unhalt trading</div>
              <div className="font-mono text-[10px] text-faint">gated: drawdown &lt; 20% · gates GREEN · practice</div></div>
            <ConfirmButton label="UNHALT" color="#2bd17e" onRun={() => runControl("unhalt")} disabled={halted === false || !armed} />
          </div>
          <div className="rounded-md border p-3 hairline sm:col-span-2">
            <div className="mb-2 flex items-center justify-between font-mono text-[11px] text-dim">
              <span>Per-lane halt / unhalt</span>
              <span className="text-faint">global HALT above overrides every lane</span>
            </div>
            <div className="grid gap-2 sm:grid-cols-3">
              {Object.entries(LANE_META).map(([lane, meta]) => {
                const laneHalted = controlState?.lanes?.[lane] ?? null;
                return (
                  <div key={lane} className="flex flex-col gap-1.5 rounded-md border p-2.5 hairline">
                    <div className="flex items-center gap-1.5 font-mono text-[11px]">
                      <StatusDot color={laneHalted === true ? "#ff4d6d" : laneHalted === false ? "#2bd17e" : "#5a6677"} />
                      <span className="text-text">{meta.label}</span>
                    </div>
                    <div className="font-mono text-[10px] text-faint">{meta.hint}</div>
                    <div className="mt-0.5 flex gap-1.5">
                      <ConfirmButton label="HALT" color="#ff4d6d" onRun={() => runControl("halt", { lane })} disabled={laneHalted === true} />
                      <ConfirmButton label="UNHALT" color="#2bd17e" onRun={() => runControl("unhalt", { lane })} disabled={laneHalted === false || !armed} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="flex items-center justify-between rounded-md border p-3 hairline">
            <div><div className="font-mono text-[12px] text-text">Trend loop</div>
              <div className="font-mono text-[10px] text-faint">start / stop the OANDA trend lane</div></div>
            <span className="flex gap-1.5">
              <ConfirmButton label="START" color="#2bd17e" onRun={() => runControl("start_loop", { loop: "trend" })} disabled={laneRunning || !armed} />
              <ConfirmButton label="STOP" color="#ff4d6d" onRun={() => runControl("stop_loop", { loop: "trend" })} disabled={!laneRunning} />
            </span>
          </div>
          <div className="rounded-md border p-3 hairline">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <StatusDot color={tier7StatusColor} pulse={tier7Running === true && !tier7Mismatch} />
                  <div className="font-mono text-[12px] text-text">Tier 7 loop</div>
                </div>
                <div className="mt-0.5 font-mono text-[10px] text-faint">
                  controller pids {tier7Proc?.pids?.length ? tier7Proc.pids.join(", ") : "—"}
                  {" · "}
                  bot {tier7SnapshotConnected ? (tier7?.running ? "heartbeat running" : "not running") : "snapshot missing"}
                </div>
              </div>
              <span className="flex shrink-0 gap-1.5">
                <ConfirmButton label="START" color="#2bd17e" onRun={() => runControl("start_loop", { loop: "tier7" })} disabled={tier7ProcRunning || !armed} />
                <ConfirmButton label="STOP" color="#ff4d6d" onRun={() => runControl("stop_loop", { loop: "tier7" })} disabled={!tier7ProcRunning} />
              </span>
            </div>
            <div className="mt-2 grid gap-1.5 font-mono text-[10.5px] text-faint sm:grid-cols-2">
              <div className="truncate" title={tier7?.running_reason ?? undefined}>
                reason: <span className={tier7Mismatch ? "text-warn" : "text-dim"}>{tier7?.running_reason ?? "—"}</span>
              </div>
              <div>
                heartbeat: <span className="tnum">{tier7?.last_cycle?.age_seconds != null ? ago(tier7.last_cycle.age_seconds) : "—"}</span>
                {tier7?.last_cycle?.pid_alive === false && <span className="text-warn"> · pid dead</span>}
              </div>
              <div>
                cycle: <span className="tnum">#{tier7?.scan_cycle_count ?? tier7?.last_cycle?.cycle_count ?? "—"}</span>
              </div>
              <div>
                snapshot:{" "}
                <span className={tier7?.snapshot_stale ? "text-warn" : "text-dim"}>
                  {tier7?.snapshot_age_s != null ? ago(tier7.snapshot_age_s) : "—"}
                  {tier7?.snapshot_stale ? " stale" : ""}
                </span>
              </div>
              <div className="truncate sm:col-span-2" title={tier7LatestHeal?.actions?.join(", ")}>
                self-heal:{" "}
                <span className="text-dim">
                  {tier7LatestHeal
                    ? `cycle ${tier7LatestHeal.cycle ?? "—"} · ${tier7LatestHeal.status ?? "—"} · ${(tier7LatestHeal.actions ?? []).join(" · ") || "no actions"}`
                    : "—"}
                </span>
              </div>
              {tier7Mismatch && (
                <div className="sm:col-span-2 text-warn">
                  controller process state and bot heartbeat disagree; trust heartbeat for liveness, STOP targets controller pids.
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center justify-between rounded-md border p-3 hairline sm:col-span-2">
            <div><div className="font-mono text-[12px] text-text">Gross leverage</div>
              <div className="font-mono text-[10px] text-faint">
                total exposure × NAV · persisted {persistedLeverage === null ? "—" : `${persistedLeverage.toFixed(1)}×`} · cap {leverageCap.toFixed(0)}×
              </div></div>
            <span className="flex items-center gap-2">
              <input type="range" min={0} max={leverageCap} step={0.5} value={lev}
                onChange={(e) => setPendingLev(parseFloat(e.target.value))} className="w-32" />
              <span className="w-10 font-mono text-[13px] font-semibold text-cyan tnum">{lev.toFixed(1)}×</span>
              <ConfirmButton label="SET" color="#22d3ee" onRun={() => runControl("set_gross_leverage", { gross_leverage: lev })} disabled={!armed} />
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
                  <span className="text-faint"> · by {e.actor ?? "—"}</span>
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
