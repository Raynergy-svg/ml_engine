"use client";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { Tier7, SystemHealth } from "@/lib/types";
import { Card, SectionTitle, Badge, StatusDot, NotConnected, Loading } from "./ui";
import { ago, shortTime } from "@/lib/format";

interface CheckRow { label: string; ok: boolean | null; meta: string }

export function Tier7Panel({ onViewLog }: { onViewLog?: () => void }) {
  const { data: tier7, loading } = usePoll<Tier7>("/api/tier7", 4000);
  const { data: health } = usePoll<SystemHealth>("/api/system_health", 5000);
  const { payload } = useStream();
  const acct = payload?.account;

  if (loading && !tier7) {
    return <Card className="flex h-full flex-col"><SectionTitle>Tier 7 Self-Healing</SectionTitle><Loading /></Card>;
  }
  if (!tier7?.connected) {
    return (
      <Card className="flex h-full flex-col">
        <SectionTitle right={<Badge color="#5a6677" dot>NOT CONNECTED</Badge>}>Tier 7 Self-Healing</SectionTitle>
        <NotConnected label="Tier 7 snapshot not published" hint={`Waiting on ${tier7?.source ?? ".claude/tier7_state.json"}.`} />
      </Card>
    );
  }

  const latestHeal = [...(tier7.self_heal?.recent_events ?? [])].reverse()[0];
  const degraded = latestHeal?.degraded === true || latestHeal?.status === "degraded";

  // Every row here is a REAL, independently-sourced signal — no fabricated latencies.
  // Health score = the fraction of these checks currently passing.
  const checks: CheckRow[] = [
    { label: "Trend Lane", ok: health?.lanes?.available ? !!health.lanes.oanda_trend_proc : null,
      meta: health?.lanes?.account_state_age_s != null ? ago(health.lanes.account_state_age_s) : "—" },
    { label: "Tier 7 Loop", ok: tier7.running ?? null,
      meta: tier7.last_cycle?.age_seconds != null ? ago(tier7.last_cycle.age_seconds) : "—" },
    { label: "Gate Verdict", ok: health?.gates?.available ? health.gates.status === "GREEN" : null,
      meta: health?.gates?.verdict_age_s != null ? ago(health.gates.verdict_age_s) : "—" },
    { label: "Self-Heal", ok: latestHeal ? !degraded : null,
      meta: latestHeal ? (degraded ? "degraded" : "nominal") : "no events" },
    { label: "Account Snapshot", ok: acct ? !acct.stale : null,
      meta: acct?.snapshot_age_s != null ? ago(acct.snapshot_age_s) : "—" },
    { label: "Alerts", ok: health?.alerts?.available ? health.alerts.count === 0 : null,
      meta: health?.alerts?.available ? `${health.alerts.count} active` : "—" },
  ];
  const known = checks.filter((c) => c.ok !== null);
  const passing = known.filter((c) => c.ok).length;
  const score = known.length ? (passing / known.length) * 100 : null;

  const ab = tier7.self_heal?.action_budget;

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle right={
        tier7.halted ? <Badge color="#ff4d6d" dot>HALTED</Badge>
        : tier7.running ? <Badge color="#2bd17e" dot pulse>RUNNING</Badge>
        : <Badge color="#f5b14c" dot>OFFLINE</Badge>
      }>
        Tier 7 Self-Healing {tier7.autonomy_level != null && <span className="text-faint">· {tier7.autonomy_level}</span>}
      </SectionTitle>

      <div className="flex min-h-0 flex-1 flex-col gap-4 p-4">
        <div>
          <div className="mb-2 flex items-end justify-between font-mono">
            <span className="text-[11px] uppercase tracking-wide text-dim" title="Fraction of the checks below currently passing">
              Health score
            </span>
            <span className="text-[13px] tnum" style={{ color: score == null ? "#8b98a9" : score >= 80 ? "#2bd17e" : score >= 50 ? "#f5b14c" : "#ff4d6d" }}>
              {score == null ? "—" : score.toFixed(1)} <span className="text-faint">/ 100</span>
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-surface2">
            <div
              className="h-full rounded-full"
              style={{ width: `${score ?? 0}%`, background: score == null ? "#5a6677" : score >= 80 ? "#2bd17e" : score >= 50 ? "#f5b14c" : "#ff4d6d" }}
            />
          </div>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-2">
          {checks.map((c) => (
            <div key={c.label} className="grid grid-cols-[1fr_auto_auto] items-center gap-3 font-mono text-[12px]">
              <span className="flex min-w-0 items-center gap-2 text-dim">
                <StatusDot color={c.ok === null ? "#5a6677" : c.ok ? "#2bd17e" : "#ff4d6d"} />
                <span className="truncate">{c.label}</span>
              </span>
              <span style={{ color: c.ok === null ? "#8b98a9" : c.ok ? "#2bd17e" : "#ff4d6d" }}>
                {c.ok === null ? "—" : c.ok ? "OK" : "FAIL"}
              </span>
              <span className="min-w-[70px] text-right text-faint tnum">{c.meta}</span>
            </div>
          ))}
        </div>

        <div className="mt-auto flex items-center gap-3 border-t pt-3 font-mono text-[11px] hairline">
          <span className="uppercase text-faint">Last self-heal</span>
          {ab?.last_action_at ? (
            <>
              <span className="text-text tnum">{shortTime(ab.last_action_at)}</span>
              <span className="min-w-0 flex-1 truncate text-dim" title={ab.last_action_label}>{ab.last_action_label}</span>
            </>
          ) : (
            <span className="min-w-0 flex-1 truncate text-faint">no self-heal actions recorded</span>
          )}
          <button onClick={onViewLog} className="rounded border px-2 py-1 text-[10px] text-dim hairline hover:text-text">
            VIEW LOG
          </button>
        </div>
      </div>
    </Card>
  );
}
