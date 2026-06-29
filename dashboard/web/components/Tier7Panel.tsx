"use client";
import { usePoll } from "@/lib/api";
import type { Tier7 } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime, ago } from "@/lib/format";

// The Tier 7 control loop: incident -> propose -> gate -> soak -> promote -> close.
const STAGE_COLOR: Record<string, string> = {
  shadow: "#8b98a9", canary: "#f5b14c", live: "#2bd17e",
  aborted: "#ff4d6d", soak: "#22d3ee", promoted: "#34e5a1",
};

export function Tier7Panel() {
  const { data, error, loading } = usePoll<Tier7>("/api/tier7", 5000);

  if (loading && !data) {
    return (
      <Card className="flex h-full flex-col">
        <SectionTitle>Autonomous Loop · Tier 7</SectionTitle>
        <Loading label="Reading loop state…" />
      </Card>
    );
  }

  // Honest not-connected: the Tier 7 snapshot isn't published yet (or unreadable).
  if (!data?.connected) {
    return (
      <Card className="flex h-full flex-col">
        <SectionTitle right={<Badge color="#5a6677" dot>NOT CONNECTED</Badge>}>
          Autonomous Loop · Tier 7
        </SectionTitle>
        <NotConnected
          label={error ? "Tier 7 endpoint error" : "Tier 7 state not published yet"}
          hint={
            data?.pending_contract
              ? `Waiting on the bot to write ${data.source}. Live loop status, current action and self-healing actions appear the moment that file exists — no activity is shown until then.`
              : `Source ${data?.source ?? "tier7_state.json"} unreadable.`
          }
        />
        <div className="mt-auto border-t px-4 py-2 hairline">
          <span className="font-mono text-[10px] text-faint">
            incident → propose → gate → soak → promote → close · read-only
          </span>
        </div>
      </Card>
    );
  }

  const halted = data.halted === true;
  const running = data.running === true;
  const sh = data.self_heal?.action_budget;
  const actionsToday = sh?.actions_today?.length ?? 0;
  const meta = data.meta_last_event;
  const lc = data.last_cycle;
  // Self-heal event feed (newest first) + the bounded-autonomy status it reports.
  const healEvents = [...(data.self_heal?.recent_events ?? [])].reverse();
  const latestHeal = healEvents[0];
  const degraded = latestHeal?.degraded === true || latestHeal?.status === "degraded";
  const healStatusColor = degraded ? "#f5b14c" : latestHeal?.status === "ok" ? "#2bd17e" : "#8b98a9";

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={
          halted ? <Badge color="#ff4d6d" dot>HALTED</Badge>
          : running ? <Badge color="#2bd17e" dot pulse>RUNNING</Badge>
          : <Badge color="#f5b14c" dot>OFFLINE</Badge>
        }
      >
        Autonomous Loop · Tier 7
      </SectionTitle>

      <div className="flex min-h-0 flex-1 flex-col gap-2.5 p-3">
        {/* running / reason — the honest liveness line */}
        <div className="flex items-center justify-between rounded-md border px-2.5 py-2 hairline">
          <div className="flex items-center gap-2">
            <StatusDot color={running ? "#2bd17e" : "#f5b14c"} pulse={running} />
            <span className="font-mono text-[12px] text-text">
              {running ? "loop running" : "loop not running"}
            </span>
          </div>
          {data.running_reason && (
            <span className="truncate font-mono text-[10.5px] text-faint" title={data.running_reason}>
              {data.running_reason}
            </span>
          )}
        </div>

        {/* current decision/action */}
        <div className="rounded-md border p-2.5 hairline">
          <div className="mb-1 flex items-center justify-between">
            <span className="eyebrow">Current action</span>
            <span className="font-mono text-[10px] text-faint">
              {data.mode && `mode ${data.mode}`}{halted ? " · halted" : ""}
            </span>
          </div>
          <div className="font-mono text-[13px] text-cyan">{data.current_action ?? "—"}</div>
          {data.goal && <div className="mt-0.5 font-mono text-[10.5px] text-dim">goal: {data.goal}</div>}
        </div>

        {/* last meta-pipeline event (incident→…→close) — compact one-liner */}
        {meta?.change_id && (
          <div className="flex items-center gap-2 px-0.5 font-mono text-[10.5px]">
            <span className="eyebrow">meta</span>
            <span className="text-dim">{meta.event}</span>
            <span
              className="rounded px-1.5 py-0.5 text-[9px] uppercase"
              style={{
                color: STAGE_COLOR[meta.stage ?? ""] ?? "#8b98a9",
                border: `1px solid ${STAGE_COLOR[meta.stage ?? ""] ?? "#1e2733"}`,
              }}
            >
              {meta.stage}
            </span>
            <span className="truncate text-faint">{meta.change_id} · {shortTime(meta.updated_at)}</span>
          </div>
        )}

        {/* self-heal control plane + bounded-autonomy status + event feed */}
        <div className="flex min-h-0 flex-1 flex-col rounded-md border p-2.5 hairline">
          <div className="flex items-center justify-between">
            <span className="eyebrow">Self-heal</span>
            <div className="flex items-center gap-2">
              {latestHeal?.status && (
                <Badge color={healStatusColor} dot>
                  {degraded ? "DEGRADED · bounded" : String(latestHeal.status).toUpperCase()}
                </Badge>
              )}
              <span className="font-mono text-[10px] text-faint tnum">{actionsToday} today</span>
            </div>
          </div>
          {sh?.last_action_label && (
            <div className="mt-1 font-mono text-[11px]">
              <span className="text-magenta">{sh.last_action_label}</span>
              {sh.last_action_at && <span className="text-faint"> · {shortTime(sh.last_action_at)}</span>}
            </div>
          )}
          {healEvents.length > 0 ? (
            <div className="scroll-thin mt-2 min-h-0 flex-1 overflow-auto">
              {healEvents.map((e, i) => (
                <div key={i} className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0">
                  <StatusDot color={e.degraded || e.status === "degraded" ? "#f5b14c" : "#2bd17e"} />
                  <div className="min-w-0 flex-1 font-mono text-[10.5px]">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-dim">
                        cycle {e.cycle} · <span style={{ color: e.degraded ? "#f5b14c" : "#8b98a9" }}>{e.status}</span>
                        {e.n_actions != null && ` · ${e.n_actions} actions`}
                      </span>
                      <span className="shrink-0 text-faint tnum">{shortTime(e.ts)}</span>
                    </div>
                    {e.actions?.length ? (
                      <div className="truncate text-[10px] text-faint" title={e.actions.join(", ")}>
                        {e.actions.join(" · ")}
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-2 font-mono text-[11px] text-faint">no self-heal events recorded</div>
          )}
        </div>

        {/* footer: last cycle + snapshot freshness */}
        <div className="mt-auto flex items-center justify-between border-t pt-2 font-mono text-[10px] text-faint hairline tnum">
          <span>
            cycle #{data.scan_cycle_count ?? lc?.cycle_count ?? "—"}
            {lc?.age_seconds != null && ` · heartbeat ${ago(lc.age_seconds)}`}
            {lc?.pid_alive === false && <span className="text-warn"> · pid dead</span>}
          </span>
          <span title={data.generated_at ?? ""}>
            snapshot {data.snapshot_stale ? <span className="text-warn">stale</span> : "live"}
            {data.snapshot_age_s != null && ` ${ago(data.snapshot_age_s)}`}
          </span>
        </div>
      </div>
    </Card>
  );
}
