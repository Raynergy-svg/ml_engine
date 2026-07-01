"use client";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { control, type ControlResult } from "@/lib/control";
import type { ControlState, Tier7 } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime, ago } from "@/lib/format";

const STAGE_COLOR: Record<string, string> = {
  shadow: "#8b98a9", canary: "#f5b14c", live: "#2bd17e",
  aborted: "#ff4d6d", soak: "#22d3ee", promoted: "#34e5a1",
};

function Tile({ label, children, sub }: { label: string; children: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="rounded-md border p-3 hairline">
      <div className="eyebrow mb-1">{label}</div>
      <div className="font-mono text-[15px] font-semibold text-text">{children}</div>
      {sub && <div className="mt-0.5 font-mono text-[10.5px] text-faint">{sub}</div>}
    </div>
  );
}

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
          {busy ? "..." : "Confirm"}
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

export function Tier7Cockpit() {
  const { data, loading } = usePoll<Tier7>("/api/tier7", 4000);
  const { data: polledControlState, error: controlError } = usePoll<ControlState>("/api/control/state", 3000);
  const [actionControlState, setActionControlState] = useState<ControlState | null>(null);

  async function runControl(action: string, params: Record<string, unknown> = {}) {
    const result = await control(action, params);
    if (result.data.state) setActionControlState(result.data.state);
    return result;
  }

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Tier 7 — autonomous loop</SectionTitle><Loading label="Reading loop state…" /></Card>;
  }
  if (!data?.connected) {
    return (
      <Card className="p-3">
        <SectionTitle right={<Badge color="#5a6677" dot>NOT CONNECTED</Badge>}>Tier 7 — autonomous loop</SectionTitle>
        <NotConnected label="Tier 7 snapshot not published"
          hint={`Waiting on ${data?.source ?? ".claude/tier7_state.json"}.`} />
      </Card>
    );
  }

  const running = data.running === true;
  const halted = data.halted === true;
  const controlState = polledControlState ?? actionControlState;
  const tier7Proc = controlState?.loops?.tier7;
  const tier7ProcRunning = tier7Proc?.running === true;
  const controlDisabled = !!controlError && /404/.test(controlError);
  const mismatch = data.connected === true && tier7Proc != null && tier7ProcRunning !== running;
  const sh = data.self_heal;
  const ab = sh?.action_budget;
  // Defensive: self_heal is untyped passthrough from disk. If the bot ever writes
  // these as a non-array/non-object (contract drift), a bare spread/Object.entries
  // throws and — with no error boundary above — white-screens the whole tab (F-M4).
  const actionsToday = Array.isArray(ab?.actions_today) ? ab.actions_today : [];
  const debounce = sh?.debounce && typeof sh.debounce === "object" && !Array.isArray(sh.debounce)
    ? Object.entries(sh.debounce) : [];
  const events = Array.isArray(sh?.recent_events) ? [...sh.recent_events].reverse() : [];
  const degraded = events[0]?.degraded === true || events[0]?.status === "degraded";
  const lc = data.last_cycle;
  const meta = data.meta_last_event;
  const autonomy = data.autonomy_level ?? (typeof data.max_autonomy === "number" ? `L${data.max_autonomy}` : null);

  return (
    <div className="flex flex-col gap-4">
      {/* status tiles */}
      <Card className="p-3">
        <SectionTitle right={
          halted ? <Badge color="#ff4d6d" dot>HALTED</Badge>
          : running ? <Badge color="#2bd17e" dot pulse>RUNNING</Badge>
          : <Badge color="#f5b14c" dot>OFFLINE</Badge>
        }>
          Tier 7 — autonomous self-healing loop · incident → propose → gate → soak → promote → close
        </SectionTitle>
        <div className="grid grid-cols-2 gap-2 p-1 sm:grid-cols-3 lg:grid-cols-5">
          <Tile label="Loop" sub={data.running_reason ?? undefined}>
            <span style={{ color: running ? "#2bd17e" : "#f5b14c" }}>{running ? "RUNNING" : "OFFLINE"}</span>
          </Tile>
          <Tile label="Autonomy" sub={autonomy == null ? "runtime not yet published" : data.bounded === false ? "unbounded?" : "bounded"}>
            {autonomy != null ? String(autonomy)
              : <span className="text-faint">—</span>}
          </Tile>
          <Tile label="Self-heal" sub={`${actionsToday.length} actions today`}>
            <span style={{ color: degraded ? "#f5b14c" : "#2bd17e" }}>{degraded ? "DEGRADED" : "nominal"}</span>
          </Tile>
          <Tile label="Cycle" sub={lc?.pid_alive === false ? "pid not alive" : `pid ${lc?.pid ?? "—"}`}>
            #{data.scan_cycle_count ?? lc?.cycle_count ?? "—"}
          </Tile>
          <Tile label="Heartbeat" sub={`snapshot ${data.snapshot_stale ? "stale" : "live"} ${ago(data.snapshot_age_s ?? undefined)}`}>
            {lc?.age_seconds != null ? ago(lc.age_seconds) : "—"}
          </Tile>
        </div>
      </Card>

      <Card className="p-3">
        <SectionTitle right={
          controlDisabled
            ? <Badge color="#5a6677" dot>CONTROL DISABLED</Badge>
            : <Badge color="#f5b14c" dot>WRITE PATH · practice · audited</Badge>
        }>
          Tier 7 loop control
        </SectionTitle>
        <div className="grid gap-2.5 lg:grid-cols-[1fr_auto]">
          <div className="grid gap-2 rounded-md border p-3 hairline sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <div className="eyebrow mb-1">controller</div>
              <div className="flex items-center gap-1.5 font-mono text-[12px] text-text">
                <StatusDot color={tier7ProcRunning ? "#2bd17e" : "#f5b14c"} pulse={tier7ProcRunning} />
                {tier7ProcRunning ? "process running" : controlDisabled ? "not available" : "not running"}
              </div>
              <div className="mt-0.5 truncate font-mono text-[10px] text-faint" title={tier7Proc?.pids?.join(", ")}>
                pids {tier7Proc?.pids?.length ? tier7Proc.pids.join(", ") : "—"}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-1">heartbeat</div>
              <div className="flex items-center gap-1.5 font-mono text-[12px] text-text">
                <StatusDot color={running ? "#2bd17e" : "#f5b14c"} pulse={running} />
                {running ? "fresh + alive" : "not live"}
              </div>
              <div className="mt-0.5 truncate font-mono text-[10px] text-faint" title={data.running_reason ?? undefined}>
                {data.running_reason ?? "—"}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-1">cycle</div>
              <div className="font-mono text-[12px] text-text tnum">#{data.scan_cycle_count ?? lc?.cycle_count ?? "—"}</div>
              <div className="mt-0.5 font-mono text-[10px] text-faint">
                heartbeat {lc?.age_seconds != null ? ago(lc.age_seconds) : "—"}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-1">snapshot</div>
              <div className={data.snapshot_stale ? "font-mono text-[12px] text-warn" : "font-mono text-[12px] text-text"}>
                {data.snapshot_stale ? "stale" : "live"}
              </div>
              <div className="mt-0.5 font-mono text-[10px] text-faint">{ago(data.snapshot_age_s ?? undefined)}</div>
            </div>
            {mismatch && (
              <div className="rounded border border-warn/40 p-2 font-mono text-[10.5px] text-warn sm:col-span-2 lg:col-span-4">
                Controller process state and Tier 7 heartbeat disagree. START/STOP targets controller pids; heartbeat remains the liveness truth.
              </div>
            )}
          </div>
          <div className="flex items-center justify-end gap-1.5 rounded-md border p-3 hairline lg:flex-col lg:items-stretch lg:justify-center">
            <ConfirmButton
              label="START"
              color="#2bd17e"
              onRun={() => runControl("start_loop", { loop: "tier7" })}
              disabled={controlDisabled || tier7ProcRunning}
            />
            <ConfirmButton
              label="STOP"
              color="#ff4d6d"
              onRun={() => runControl("stop_loop", { loop: "tier7" })}
              disabled={controlDisabled || !tier7ProcRunning}
            />
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* current action + meta */}
        <Card className="flex min-w-0 flex-col p-3">
          <SectionTitle>Doing now</SectionTitle>
          <div className="rounded-md border p-3 hairline">
            <div className="break-all font-mono text-[14px] text-cyan">{data.current_action ?? "idle"}</div>
            <div className="mt-1 font-mono text-[11px] text-dim">
              mode {data.mode ?? "—"}{data.goal ? ` · goal ${data.goal}` : ""}{halted ? " · halted" : ""}
            </div>
          </div>
          <div className="eyebrow mt-3 mb-1">Last meta-pipeline event</div>
          {meta?.change_id ? (
            <div className="min-w-0 overflow-hidden rounded-md border p-2.5 font-mono text-[11.5px] hairline">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <span className="text-text">{meta.event}</span>
                <span className="rounded px-1.5 py-0.5 text-[9.5px] uppercase"
                  style={{ color: STAGE_COLOR[meta.stage ?? ""] ?? "#8b98a9", border: `1px solid ${STAGE_COLOR[meta.stage ?? ""] ?? "#1e2733"}` }}>
                  {meta.stage}
                </span>
                <span className="text-faint">{meta.deploy_target}</span>
              </div>
              <div className="mt-0.5 truncate text-[10.5px] text-faint tnum">{meta.change_id} · {meta.kind} · {shortTime(meta.updated_at)}</div>
            </div>
          ) : <div className="font-mono text-[11px] text-faint">no meta event recorded</div>}
        </Card>

        {/* self-heal control plane */}
        <Card className="flex min-w-0 flex-col p-3">
          <SectionTitle right={
            ab?.last_action_label
              ? <Badge color="#e84ff0">{actionsToday.length} today</Badge>
              : undefined
          }>
            Self-heal control plane
          </SectionTitle>
          {ab?.last_action_label && (
            <div className="rounded-md border p-2.5 hairline">
              <div className="eyebrow mb-0.5">last action</div>
              <span className="break-all font-mono text-[12px] text-magenta">{ab.last_action_label}</span>
              {ab.last_action_at && <span className="font-mono text-[10.5px] text-faint"> · {shortTime(ab.last_action_at)}</span>}
            </div>
          )}
          {actionsToday.length > 0 && (
            <div className="mt-2">
              <div className="eyebrow mb-1">action-budget timeline ({actionsToday.length})</div>
              <div className="scroll-thin flex flex-wrap gap-1 overflow-auto" style={{ maxHeight: 64 }}>
                {actionsToday.slice(-12).map((t, i) => (
                  <span key={i} className="rounded border px-1.5 py-0.5 font-mono text-[9.5px] text-faint hairline tnum">
                    {shortTime(t)}
                  </span>
                ))}
              </div>
            </div>
          )}
          {debounce.length > 0 && (
            <div className="mt-2">
              <div className="eyebrow mb-1">debounce ({debounce.length})</div>
              <div className="scroll-thin overflow-auto font-mono text-[10px] text-dim" style={{ maxHeight: 56 }}>
                {debounce.map(([k, v]) => (
                  <div key={k} className="truncate" title={`${k} · ${v}`}>{k}</div>
                ))}
              </div>
            </div>
          )}
          {!ab?.last_action_label && actionsToday.length === 0 && (
            <div className="font-mono text-[11px] text-faint">no self-heal actions recorded</div>
          )}
        </Card>
      </div>

      {/* self-heal event / cycle stream */}
      <Card className="flex flex-col p-3">
        <SectionTitle right={<span className="font-mono text-[10px] text-faint tnum">{events.length} cycles</span>}>
          Self-heal event stream · cycle history
        </SectionTitle>
        {events.length === 0 ? (
          <div className="py-4 text-center font-mono text-[11px] text-faint">
            No self-heal cycle events in the current snapshot — the loop is healthy / idle (honest empty state).
          </div>
        ) : (
          <div className="scroll-thin max-h-[260px] overflow-auto font-mono text-[11px]">
            {events.map((e, i) => (
              <div key={i} className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0">
                <StatusDot color={e.degraded || e.status === "degraded" ? "#f5b14c" : "#2bd17e"} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-dim">
                      cycle {e.cycle} · <span style={{ color: e.degraded ? "#f5b14c" : "#8b98a9" }}>{e.status}</span>
                      {e.n_actions != null && ` · ${e.n_actions} actions`}
                    </span>
                    <span className="shrink-0 text-faint tnum">{shortTime(e.ts)}</span>
                  </div>
                  {e.actions?.length ? (
                    <div className="truncate text-[10px] text-faint" title={e.actions.join(", ")}>{e.actions.join(" · ")}</div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
