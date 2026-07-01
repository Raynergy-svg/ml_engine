"use client";
import { usePoll } from "@/lib/api";
import type { Tier7, Tier7Diagnostics } from "@/lib/types";
import { Card, SectionTitle, Badge, NotConnected, Loading } from "./ui";
import { shortTime } from "@/lib/format";

function CheckIcon({ ok }: { ok: boolean | null }) {
  const color = ok === null ? "#5a6677" : ok ? "#2bd17e" : "#ff4d6d";
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="shrink-0">
      <rect x="1" y="1" width="12" height="12" rx="3" stroke={color} strokeWidth="1.3" />
      {ok !== false && <path d="M3.8 7.1 5.7 9 10.2 4.3" stroke={color} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />}
    </svg>
  );
}

function OkPill({ ok }: { ok: boolean | null }) {
  const color = ok === null ? "#8b98a9" : ok ? "#2bd17e" : "#ff4d6d";
  return (
    <span className="flex items-center gap-1 font-mono text-[11.5px] font-semibold" style={{ color }}>
      <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="7" r="6" stroke={color} strokeWidth="1.3" />
        {ok !== false && <path d="M4 7.2 6 9.2 10 4.8" stroke={color} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />}
      </svg>
      {ok === null ? "—" : ok ? "OK" : "FAIL"}
    </span>
  );
}

export function Tier7Panel({ onViewLog }: { onViewLog?: () => void }) {
  const { data: tier7, loading } = usePoll<Tier7>("/api/tier7", 4000);
  const { data: diag } = usePoll<Tier7Diagnostics>("/api/tier7_diagnostics", 5000);

  if (loading && !tier7) {
    return <Card className="flex h-full flex-col"><SectionTitle>TIER 7 SELF-HEALING</SectionTitle><Loading /></Card>;
  }
  if (!tier7?.connected) {
    return (
      <Card className="flex h-full flex-col">
        <SectionTitle right={<Badge color="#5a6677" dot>NOT CONNECTED</Badge>}>TIER 7 SELF-HEALING</SectionTitle>
        <NotConnected label="Tier 7 snapshot not published" hint={`Waiting on ${tier7?.source ?? ".claude/tier7_state.json"}.`} />
      </Card>
    );
  }

  const ab = tier7.self_heal?.action_budget;
  const score = diag?.score ?? null;
  const scoreColor = score == null ? "#8b98a9" : score >= 80 ? "#2bd17e" : score >= 50 ? "#f5b14c" : "#ff4d6d";

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle right={
        tier7.halted ? <Badge color="#ff4d6d" dot>HALTED</Badge>
        : tier7.running ? <Badge color="#2bd17e" dot pulse>RUNNING</Badge>
        : <Badge color="#f5b14c" dot>OFFLINE</Badge>
      }>
        TIER 7 SELF-HEALING
      </SectionTitle>

      <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
        <div>
          <div className="mb-2 flex items-end justify-between font-mono">
            <span className="text-[11px] uppercase tracking-wide text-dim">Health score</span>
            <span className="text-[13px] tnum" style={{ color: scoreColor }}>
              {score == null ? "—" : score.toFixed(1)} <span className="text-faint">/ 100</span>
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-surface2">
            <div className="h-full rounded-full" style={{ width: `${score ?? 0}%`, background: scoreColor }} />
          </div>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-2.5">
          {(diag?.checks ?? []).map((c) => (
            <div key={c.label} className="grid grid-cols-[1fr_auto_auto] items-center gap-3 font-mono text-[12px]">
              <span className="flex min-w-0 items-center gap-2 text-dim">
                <CheckIcon ok={c.ok} />
                <span className="truncate">{c.label}</span>
              </span>
              <OkPill ok={c.ok} />
              <span className="min-w-[86px] text-right text-faint tnum">{c.metric ?? "—"}</span>
            </div>
          ))}
          {!diag && <span className="font-mono text-[11px] text-faint">loading diagnostics…</span>}
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
