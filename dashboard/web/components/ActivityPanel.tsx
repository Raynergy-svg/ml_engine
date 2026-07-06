"use client";
import { useMemo, useState } from "react";
import { usePoll } from "@/lib/api";
import { control, type ControlResult } from "@/lib/control";
import { runAxiomOperator } from "@/lib/operator";
import { useStream } from "@/lib/stream";
import type {
  ActivityLogFeed,
  ActivitySnapshot,
  AxiomOperator,
  ControlState,
  SystemHealth,
  Tier7,
} from "@/lib/types";
import { ago, shortTime } from "@/lib/format";
import { Badge, Card, Loading, NotConnected, SectionTitle, StatusDot } from "./ui";

interface AuditEntry {
  ts: string;
  action: string;
  allowed?: boolean;
  reason?: string;
  result?: string;
  actor?: string;
  lane?: string;
}
interface AuditResp { entries: AuditEntry[]; count: number }

const ACTIVE_STATUSES = new Set(["queued", "running", "waiting", "starting"]);
const EXPECTED_LOG_AGE_S: Record<string, number> = {
  tier7: 120,
  trend: 75 * 60,
  equity: 75 * 60,
  safety: 10 * 60,
};
const OPERATOR_STATUS_COLOR: Record<string, string> = {
  idle: "#8b98a9",
  running: "#35c7e8",
  acted: "#3ee287",
  handoff: "#35c7e8",
  held: "#ffc23e",
  waiting: "#ffc23e",
  unavailable: "#ff6b5e",
};
type FeedState = { label: string; color: string; pulse?: boolean };
type EvidenceTone = "normal" | "good" | "warn" | "bad";
type EvidenceRow = { label: string; value: string; tone?: EvidenceTone };
type IncidentAction = {
  id: string;
  label: string;
  detail: string;
  color: string;
  disabled?: boolean;
  run: () => Promise<ControlResult | { ok: boolean; data: Record<string, unknown>; status: number }>;
};
type Incident = {
  id: string;
  severity: "critical" | "watch" | "info";
  title: string;
  summary: string;
  badge: string;
  evidence: EvidenceRow[];
  lines: string[];
  actions: IncidentAction[];
  reviewed?: boolean;
};

function ageSeconds(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return null;
  return Math.max(0, (Date.now() - t) / 1000);
}

function heldLabel(ageS: number | null | undefined, expectedS: number): FeedState {
  if (ageS == null) return { label: "NO LOG", color: "#5a6677" };
  if (ageS <= expectedS) return { label: "FRESH", color: "#3ee287" };
  if (ageS <= expectedS * 2) return { label: "QUIET", color: "#ffc23e" };
  return { label: "HELD", color: "#ff6b5e" };
}

function LogLine({ line }: { line: string }) {
  const warn = /WARN|REFUSE|BLOCK|NEEDS ATTENTION|halt|stale/i.test(line);
  const bad = /ERROR|FAILED|Traceback|bucket_over_cap/i.test(line);
  return (
    <div className={`truncate border-t py-1 hairline first:border-t-0 ${bad ? "text-neg" : warn ? "text-warn" : "text-dim"}`} title={line}>
      {line}
    </div>
  );
}

function LogDrawer({ feed, stateOverride }: { feed: ActivityLogFeed; stateOverride?: FeedState }) {
  const expected = EXPECTED_LOG_AGE_S[feed.id] ?? 600;
  const state = stateOverride ?? heldLabel(feed.age_s, expected);
  return (
    <details className="rounded-md border p-2.5 hairline">
      <summary className="grid cursor-pointer list-none grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
        <span className="min-w-0">
          <span className="block truncate font-mono text-[12px] text-text" title={feed.title}>{feed.title}</span>
          <span className="block truncate font-mono text-[10px] text-faint" title={feed.path}>
            {feed.path} · updated {feed.age_s == null ? "—" : ago(feed.age_s)}
          </span>
        </span>
        <Badge color={state.color} dot pulse={state.pulse ?? state.label === "FRESH"}>{state.label}</Badge>
      </summary>
      <div className="mt-2">
        {!feed.exists ? (
          <NotConnected label="Log not present yet" />
        ) : !feed.lines.length ? (
          <div className="py-4 text-center font-mono text-[11px] text-faint">No lines written yet.</div>
        ) : (
          <div className="scroll-thin max-h-[180px] overflow-auto rounded-md border bg-black/15 p-2 font-mono text-[10.5px] leading-5 hairline">
            {feed.lines.map((line, i) => <LogLine key={`${feed.id}-${i}-${line}`} line={line} />)}
          </div>
        )}
      </div>
    </details>
  );
}

function evidenceToneClass(tone: EvidenceTone | undefined): string {
  if (tone === "good") return "text-pos";
  if (tone === "warn") return "text-warn";
  if (tone === "bad") return "text-neg";
  return "text-text";
}

function severityColor(severity: Incident["severity"]): string {
  if (severity === "critical") return "#ff6b5e";
  if (severity === "watch") return "#ffc23e";
  return "#3ee287";
}

function ConfirmActionButton({ action }: { action: IncidentAction }) {
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  async function go() {
    setBusy(true);
    const res = await action.run();
    const message = res.ok
      ? String(res.data.result ?? res.data.message ?? "ok")
      : `${res.status}: ${String(res.data.detail ?? res.data.error ?? "denied")}`;
    setResult(message);
    setBusy(false);
    setArmed(false);
    setTimeout(() => setResult(null), 6000);
  }

  return (
    <div className="flex flex-col items-end gap-1">
      {armed ? (
        <span className="inline-flex items-center gap-1">
          <button
            onClick={go}
            disabled={busy}
            className="rounded-md px-2.5 py-1 font-mono text-[11px] font-semibold text-base"
            style={{ background: action.color }}
          >
            {busy ? "…" : "Confirm"}
          </button>
          <button onClick={() => setArmed(false)} className="rounded-md border px-2 py-1 font-mono text-[11px] text-faint hairline">
            cancel
          </button>
        </span>
      ) : (
        <button
          onClick={() => setArmed(true)}
          disabled={action.disabled || busy}
          className="rounded-md border px-2.5 py-1 font-mono text-[11px] hairline disabled:opacity-40"
          style={{ color: action.color }}
        >
          {action.label}
        </button>
      )}
      {result && <span className="max-w-[220px] truncate font-mono text-[10px] text-faint" title={result}>{result}</span>}
    </div>
  );
}

function IncidentCard({ incident }: { incident: Incident }) {
  const color = severityColor(incident.severity);
  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 border-b p-3 hairline">
        <div className="h-10 w-2 rounded-full" style={{ background: color, boxShadow: `0 0 18px ${color}66` }} />
        <div className="min-w-0">
          <div className="truncate font-mono text-[14px] font-semibold text-text" title={incident.title}>{incident.title}</div>
          <div className="mt-0.5 truncate font-mono text-[11px] text-dim" title={incident.summary}>{incident.summary}</div>
        </div>
        <Badge color={color} dot>{incident.reviewed ? "REVIEWED" : incident.badge}</Badge>
      </div>
      <div className="grid gap-3 p-3 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
        <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline">
          <div className="eyebrow mb-2 tracking-[0.15em]">Proof / evidence</div>
          <div className="font-mono text-[11px]">
            {incident.evidence.map((row) => (
              <div key={`${incident.id}-${row.label}`} className="grid grid-cols-[120px_minmax(0,1fr)] gap-3 border-t py-1.5 hairline first:border-t-0">
                <span className="text-faint">{row.label}</span>
                <span className={`truncate ${evidenceToneClass(row.tone)}`} title={row.value}>{row.value}</span>
              </div>
            ))}
          </div>
          {!!incident.lines.length && (
            <div className="mt-3 rounded-md border bg-black/20 p-2 font-mono text-[10.5px] leading-5 hairline">
              {incident.lines.slice(-4).map((line, i) => <LogLine key={`${incident.id}-line-${i}-${line}`} line={line} />)}
            </div>
          )}
        </div>
        <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline">
          <div className="eyebrow mb-2 tracking-[0.15em]">Recovery / safe actions</div>
          <div className="grid gap-2">
            {incident.actions.map((action) => (
              <div key={action.id} className="flex items-center justify-between gap-3 rounded-md border p-2.5 hairline">
                <div className="min-w-0">
                  <div className="truncate font-mono text-[12px] text-text" title={action.label}>{action.label}</div>
                  <div className="mt-0.5 line-clamp-2 font-mono text-[10px] text-faint" title={action.detail}>{action.detail}</div>
                </div>
                <ConfirmActionButton action={action} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

function Tile({
  label, value, detail, color, pulse = false,
}: { label: string; value: string; detail?: string; color: string; pulse?: boolean }) {
  return (
    <div className="rounded-md border p-3 hairline">
      <div className="mb-2 flex items-center gap-2">
        <StatusDot color={color} pulse={pulse} />
        <span className="eyebrow tracking-[0.16em]">{label}</span>
      </div>
      <div className="truncate font-mono text-[18px] font-semibold text-text" title={value}>{value}</div>
      {detail && <div className="mt-1 truncate font-mono text-[10.5px] text-faint" title={detail}>{detail}</div>}
    </div>
  );
}

function OperatorPanel({
  operator,
  onRun,
  running,
  result,
}: {
  operator?: AxiomOperator | null;
  onRun: () => Promise<void>;
  running: boolean;
  result: string | null;
}) {
  const session = operator?.session;
  const status = session?.status ?? "idle";
  const color = OPERATOR_STATUS_COLOR[status] ?? "#8b98a9";
  const usedPct = typeof session?.context_used_pct === "number" ? Math.round(session.context_used_pct * 100) : null;
  const tools = session?.last_tools ?? [];
  const blockers = session?.blocked_reasons ?? [];
  return (
    <Card className="p-3">
      <SectionTitle right={<Badge color={color} dot pulse={status === "running" || running}>{running ? "THINKING" : status.toUpperCase()}</Badge>}>
        AXIOM operator
      </SectionTitle>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(260px,0.75fr)]">
        <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline">
          <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-2 font-mono text-[11px]">
            <span className="text-faint">Provider</span>
            <span className="truncate text-text">{session?.provider ?? "claude_subscription"}</span>
            <span className="text-faint">Epoch</span>
            <span className="text-text tnum">#{session?.epoch ?? 0}</span>
            <span className="text-faint">Context</span>
            <span className={usedPct != null && usedPct >= 80 ? "text-warn" : "text-dim"}>
              {usedPct == null ? "not measured" : `${usedPct}% used`}
            </span>
            <span className="text-faint">Last action</span>
            <span className="truncate text-text" title={session?.last_action ?? "observe"}>{session?.last_action ?? "observe"}</span>
            <span className="text-faint">Next</span>
            <span className="truncate text-text" title={session?.next_action ?? "observe"}>{session?.next_action ?? "observe"}</span>
          </div>
          <div className="mt-3 border-t pt-3 font-mono text-[11px] leading-5 text-dim hairline">
            <div className="eyebrow mb-1 tracking-[0.14em]">Reasoning</div>
            <div className="line-clamp-3" title={session?.last_reasoning ?? session?.handoff_summary ?? "No operator epoch has run yet."}>
              {session?.last_reasoning ?? session?.handoff_summary ?? "No operator epoch has run yet."}
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t pt-3 hairline">
            <button
              onClick={onRun}
              disabled={running}
              className="rounded-md border px-3 py-1.5 font-mono text-[11px] font-semibold text-pos hairline disabled:opacity-40"
            >
              {running ? "Thinking..." : "Think now"}
            </button>
            {result && <span className="max-w-[420px] truncate font-mono text-[10.5px] text-faint" title={result}>{result}</span>}
          </div>
        </div>
        <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline">
          <div className="eyebrow mb-2 tracking-[0.14em]">Held / tools</div>
          {session?.api_key_refused && (
            <div className="mb-2 rounded-md border border-warn/40 bg-warn/10 px-2 py-1.5 font-mono text-[10.5px] text-warn">
              API key detected; subscription mode refused paid API billing.
            </div>
          )}
          <div className="grid gap-1.5 font-mono text-[10.5px]">
            {(blockers.length ? blockers : ["No active operator hold."]).slice(0, 3).map((item, i) => (
              <div key={`op-block-${i}-${item}`} className="truncate text-dim" title={item}>- {item}</div>
            ))}
          </div>
          <div className="mt-3 border-t pt-2 hairline">
            <div className="flex flex-wrap gap-1.5">
              {(tools.length ? tools : ["no tools called"]).slice(0, 5).map((tool) => (
                <Badge key={tool} color={tools.length ? "#35c7e8" : "#5a6677"}>{tool}</Badge>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

export function ActivityPanel() {
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [reviewed, setReviewed] = useState<Set<string>>(() => new Set());
  const [operatorBusy, setOperatorBusy] = useState(false);
  const [operatorResult, setOperatorResult] = useState<string | null>(null);
  const refreshSuffix = refreshNonce ? `?r=${refreshNonce}` : "";
  const auditSuffix = refreshNonce ? `&r=${refreshNonce}` : "";
  const { data, loading, error } = usePoll<ActivitySnapshot>(`/api/activity${refreshSuffix}`, 3000);
  const { data: tier7 } = usePoll<Tier7>(`/api/tier7${refreshSuffix}`, 4000);
  const { data: health } = usePoll<SystemHealth>(`/api/system_health${refreshSuffix}`, 5000);
  const { data: controlState, error: controlError } = usePoll<ControlState>(`/api/control/state${refreshSuffix}`, 5000);
  const { data: audit } = usePoll<AuditResp>(`/api/control/audit?limit=8${auditSuffix}`, 5000);
  const { data: operator } = usePoll<AxiomOperator>(`/api/axiom_operator${refreshSuffix}`, 5000);
  const { payload, connected } = useStream();

  const activityItems = useMemo(() => {
    const items = [...(data?.background.active ?? []), ...(data?.background.history ?? [])];
    return items
      .sort((a, b) => (new Date(b.updated_at || b.started_at || 0).getTime() - new Date(a.updated_at || a.started_at || 0).getTime()))
      .slice(0, 8);
  }, [data]);

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Activity</SectionTitle><Loading label="Loading activity feed..." /></Card>;
  }

  if (!data && error) {
    return <Card className="p-3"><SectionTitle>Activity</SectionTitle><NotConnected label="Activity endpoint unavailable" hint={error} /></Card>;
  }

  const tier7Running = tier7?.connected ? tier7.running === true : null;
  const trendRunning = controlState?.loops?.trend?.running ?? health?.lanes?.running ?? payload?.status?.running ?? null;
  const halted = controlState?.halted ?? payload?.status?.halted ?? tier7?.halted ?? null;
  const alertCount = health?.alerts?.count ?? 0;
  const logFeeds = data?.log_feeds ?? [];
  const heldFeeds = (data?.log_feeds ?? []).filter((feed) => {
    if (feed.id === "tier7" && tier7?.running === true && (tier7.last_cycle?.age_seconds ?? Infinity) <= 120) {
      return false;
    }
    const expected = EXPECTED_LOG_AGE_S[feed.id] ?? 600;
    return feed.age_s != null && feed.age_s > expected * 2;
  });
  const quietActive = activityItems.filter((item) => ACTIVE_STATUSES.has(String(item.status).toLowerCase()) && (ageSeconds(item.updated_at) ?? 0) > 5 * 60);
  const latestHeal = [...(tier7?.self_heal?.recent_events ?? [])].slice(-5).reverse();
  const tier7Feed = logFeeds.find((feed) => feed.id === "tier7");
  const tier7HeartbeatFresh = tier7?.running === true && (tier7.last_cycle?.age_seconds ?? Infinity) <= 120;
  const tier7LogQuiet = !!tier7Feed?.age_s && tier7Feed.age_s > (EXPECTED_LOG_AGE_S.tier7 * 2);
  const deniedAudit = (audit?.entries ?? []).find((entry) => entry.allowed === false);
  const controlUnavailable = !!controlError && /404|405|502/.test(controlError);
  const armed = controlState?.armed === true;

  function runControl(action: string, params: Record<string, unknown> = {}) {
    return async () => {
      const result = await control(action, params);
      setRefreshNonce((n) => n + 1);
      return result;
    };
  }

  function recheck(id: string) {
    return async () => {
      setRefreshNonce((n) => n + 1);
      return { status: 200, ok: true, data: { result: "rechecked", message: id } };
    };
  }

  function markReviewed(id: string) {
    return async () => {
      setReviewed((prev) => {
        const next = new Set(prev);
        next.add(id);
        return next;
      });
      return { status: 200, ok: true, data: { result: "reviewed" } };
    };
  }

  async function runOperatorEpoch() {
    setOperatorBusy(true);
    setOperatorResult(null);
    const res = await runAxiomOperator();
    const data = res.data as Record<string, unknown>;
    const action = String(data.action ?? "observe");
    const status = String(data.status ?? (res.ok ? "ok" : "failed"));
    const error = data.error ? ` · ${String(data.error)}` : "";
    setOperatorResult(res.ok ? `epoch ${data.epoch ?? "?"} · ${status} · ${action}${error}` : `${res.status}: ${String(data.error ?? "operator run failed")}`);
    setRefreshNonce((n) => n + 1);
    setOperatorBusy(false);
  }

  const incidents: Incident[] = [];

  for (const alert of health?.alerts?.active ?? []) {
    const id = `alert-${alert.alert_type}-${alert.timestamp}`;
    incidents.push({
      id,
      severity: String(alert.severity).toLowerCase() === "critical" ? "critical" : "watch",
      title: alert.message || `${alert.alert_type} alert`,
      summary: `AlertManager · ${alert.alert_type} · ${shortTime(alert.timestamp)}`,
      badge: "ACTION REQUIRED",
      reviewed: reviewed.has(id) || alert.acknowledged === true,
      evidence: [
        { label: "Trigger", value: alert.alert_type, tone: "bad" },
        { label: "Severity", value: alert.severity || "unknown", tone: String(alert.severity).toLowerCase() === "critical" ? "bad" : "warn" },
        { label: "Value", value: alert.value == null ? "—" : String(alert.value), tone: "warn" },
        { label: "Threshold", value: alert.threshold == null ? "—" : String(alert.threshold) },
        { label: "AXIOM action", value: halted ? "global halt active" : "no emergency trade-risk action reported", tone: halted ? "warn" : "normal" },
      ],
      lines: [alert.message, `last updated ${health?.alerts?.last_updated ? shortTime(health.alerts.last_updated) : "—"}`].filter(Boolean),
      actions: [
        {
          id: `${id}-pause-oanda`,
          label: "Pause trend lane",
          detail: controlUnavailable
            ? "Control write path is offline."
            : controlState?.lanes?.oanda_fx === true
              ? "Already contained; the OANDA lane is halted."
              : "Contain new OANDA entries while keeping evidence intact.",
          color: "#ffc23e",
          disabled: controlUnavailable || controlState?.lanes?.oanda_fx === true,
          run: runControl("halt", { lane: "oanda_fx" }),
        },
        {
          id: `${id}-recheck`,
          label: "Recheck state",
          detail: "Refresh alerts, lane status, control audit, and log tails.",
          color: "#35c7e8",
          run: recheck(id),
        },
        {
          id: `${id}-review`,
          label: "Mark reviewed",
          detail: "Acknowledge this Activity card for this browser session.",
          color: "#3ee287",
          run: markReviewed(id),
        },
      ],
    });
  }

  for (const feed of heldFeeds) {
    const id = `held-feed-${feed.id}`;
    const loopName = feed.id === "trend" ? "trend" : feed.id === "tier7" ? "tier7" : null;
    const laneName = feed.id === "trend" ? "oanda_fx" : feed.id === "equity" ? "equity" : null;
    const loopRunning = loopName ? controlState?.loops?.[loopName]?.running === true : false;
    const actions: IncidentAction[] = [
      {
        id: `${id}-recheck`,
        label: "Recheck liveness",
        detail: "Refresh heartbeat, PID, snapshot, and log age before changing state.",
        color: "#35c7e8",
        run: recheck(id),
      },
      {
        id: `${id}-review`,
        label: "Mark reviewed",
        detail: "Keep the evidence visible but mark this card as reviewed.",
        color: "#3ee287",
        run: markReviewed(id),
      },
    ];
    if (loopName) {
      actions.splice(1, 0, {
        id: `${id}-${loopRunning ? "stop" : "start"}-${loopName}`,
        label: loopRunning ? `Stop ${loopName} loop` : `Start ${loopName} loop`,
        detail: loopRunning ? "Operational containment for a stuck loop." : armed ? "Restart the whitelisted loop after review." : "Requires ARM in Operator Control before a loop can start.",
        color: loopRunning ? "#ffc23e" : "#3ee287",
        disabled: controlUnavailable || (!loopRunning && !armed),
        run: runControl(loopRunning ? "stop_loop" : "start_loop", { loop: loopName }),
      });
    }
    if (laneName) {
      actions.push({
        id: `${id}-pause-${laneName}`,
        label: `Pause ${laneName} lane`,
        detail: controlUnavailable
          ? "Control write path is offline."
          : controlState?.lanes?.[laneName] === true
            ? "Already contained; this lane is halted."
            : "Contain this lane without using global emergency controls.",
        color: "#ffc23e",
        disabled: controlUnavailable || controlState?.lanes?.[laneName] === true,
        run: runControl("halt", { lane: laneName }),
      });
    }
    incidents.push({
      id,
      severity: "watch",
      title: `${feed.title} is held up`,
      summary: `Last log write ${feed.age_s == null ? "unknown" : ago(feed.age_s)} · expected ${ago(EXPECTED_LOG_AGE_S[feed.id] ?? 600)}`,
      badge: "WATCHING",
      reviewed: reviewed.has(id),
      evidence: [
        { label: "Feed", value: feed.id },
        { label: "Path", value: feed.path },
        { label: "Log age", value: feed.age_s == null ? "unknown" : ago(feed.age_s), tone: "warn" },
        { label: "Exists", value: feed.exists ? "yes" : "missing", tone: feed.exists ? "normal" : "bad" },
      ],
      lines: feed.lines,
      actions,
    });
  }

  if (tier7HeartbeatFresh && tier7LogQuiet && tier7Feed && !heldFeeds.some((feed) => feed.id === "tier7")) {
    const id = "tier7-heartbeat-log-mismatch";
    incidents.push({
      id,
      severity: "info",
      title: "Tier 7 heartbeat fresh but stdout is quiet",
      summary: `Heartbeat ${ago(tier7?.last_cycle?.age_seconds)} · stdout ${ago(tier7Feed.age_s)}`,
      badge: "LOG GAP",
      reviewed: reviewed.has(id),
      evidence: [
        { label: "Heartbeat", value: ago(tier7?.last_cycle?.age_seconds), tone: "good" },
        { label: "Stdout age", value: ago(tier7Feed.age_s), tone: "warn" },
        { label: "PID", value: tier7?.last_cycle?.pid_alive === false ? "dead" : "alive", tone: tier7?.last_cycle?.pid_alive === false ? "bad" : "good" },
        { label: "AXIOM action", value: "no restart required while heartbeat is fresh" },
      ],
      lines: tier7Feed.lines,
      actions: [
        {
          id: `${id}-recheck`,
          label: "Recheck liveness",
          detail: "Refresh Tier 7 heartbeat, PID, and stdout age.",
          color: "#35c7e8",
          run: recheck(id),
        },
        {
          id: `${id}-review`,
          label: "Mark reviewed",
          detail: "Treat as observed logging gap unless heartbeat degrades.",
          color: "#3ee287",
          run: markReviewed(id),
        },
      ],
    });
  }

  for (const item of quietActive.slice(0, 2)) {
    const id = `quiet-task-${item.id}`;
    const latest = item.events?.at(-1);
    incidents.push({
      id,
      severity: "watch",
      title: item.title || `${item.kind} has gone quiet`,
      summary: item.summary || latest?.message || item.source || "Background activity has not updated recently",
      badge: "QUIET TASK",
      reviewed: reviewed.has(id),
      evidence: [
        { label: "Status", value: String(item.status || "unknown"), tone: "warn" },
        { label: "Updated", value: ageSeconds(item.updated_at) == null ? "unknown" : ago(ageSeconds(item.updated_at)), tone: "warn" },
        { label: "Source", value: item.source || "—" },
        { label: "Error", value: item.error || "none", tone: item.error ? "bad" : "normal" },
      ],
      lines: (item.events ?? []).slice(-4).map((event) => `${shortTime(event.timestamp)} ${event.message || "event"}`),
      actions: [
        {
          id: `${id}-recheck`,
          label: "Recheck task",
          detail: "Refresh background activity before escalating.",
          color: "#35c7e8",
          run: recheck(id),
        },
        {
          id: `${id}-review`,
          label: "Mark reviewed",
          detail: "Keep the task in recent context but lower operator noise.",
          color: "#3ee287",
          run: markReviewed(id),
        },
      ],
    });
  }

  if (deniedAudit) {
    const id = `denied-audit-${deniedAudit.ts}-${deniedAudit.action}`;
    incidents.push({
      id,
      severity: "info",
      title: `Control action denied: ${deniedAudit.action}`,
      summary: deniedAudit.reason || "Backend safety layer refused the request",
      badge: "DENIED",
      reviewed: reviewed.has(id),
      evidence: [
        { label: "Action", value: deniedAudit.action, tone: "warn" },
        { label: "Result", value: deniedAudit.result || "denied", tone: "warn" },
        { label: "Actor", value: deniedAudit.actor || "—" },
        { label: "Reason", value: deniedAudit.reason || "—", tone: "bad" },
      ],
      lines: [deniedAudit.reason || "control denied"],
      actions: [
        {
          id: `${id}-recheck`,
          label: "Recheck control",
          detail: "Refresh control state and audit before retrying elsewhere.",
          color: "#35c7e8",
          run: recheck(id),
        },
        {
          id: `${id}-review`,
          label: "Mark reviewed",
          detail: "Acknowledge the denial in Activity.",
          color: "#3ee287",
          run: markReviewed(id),
        },
      ],
    });
  }

  const actionableIncidents = incidents
    .filter((incident) => !incident.reviewed || incident.severity === "critical")
  const activeIncidents = actionableIncidents.slice(0, 5);
  const reviewedCount = reviewed.size;
  const activeBadge = actionableIncidents.length
    ? `${activeIncidents.length}/${actionableIncidents.length} ACTIVE${reviewedCount ? ` · ${reviewedCount} REVIEWED` : ""}`
    : reviewedCount ? `${reviewedCount} REVIEWED` : "CLEAR";
  const recentContext = [
    ...latestHeal.slice(0, 3).map((event) => ({
      id: `heal-${event.ts}-${event.cycle}`,
      at: shortTime(event.ts),
      text: `Tier 7 cycle #${event.cycle ?? "—"} · ${event.status ?? "ok"} · ${(event.actions ?? []).join(" · ") || "no action"}`,
      color: event.degraded || event.status === "degraded" ? "#ffc23e" : "#3ee287",
    })),
    ...(audit?.entries ?? []).slice(0, 5).map((entry) => ({
      id: `audit-${entry.ts}-${entry.action}-${entry.lane ?? ""}`,
      at: shortTime(entry.ts),
      text: `${entry.action}${entry.lane ? ` · ${entry.lane}` : ""} · ${entry.result ?? (entry.allowed ? "allowed" : "denied")}`,
      color: entry.allowed === false ? "#ff6b5e" : "#8b98a9",
    })),
  ].slice(0, 6);

  return (
    <div className="flex flex-col gap-4">
      <Card className="p-4">
        <SectionTitle right={
          <Badge color={heldFeeds.length || quietActive.length || alertCount ? "#ffc23e" : "#3ee287"} dot pulse={!heldFeeds.length && !quietActive.length}>
            {heldFeeds.length || quietActive.length ? "WATCHING HOLDS" : "LIVE FEED"}
          </Badge>
        }>
          Activity · bot operations
        </SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Tile
            label="Terminal stream"
            value={connected ? "CONNECTED" : "OFFLINE"}
            detail={payload?.status?.last_updated ? `snapshot ${shortTime(payload.status.last_updated)}` : "SSE live-data channel"}
            color={connected ? "#3ee287" : "#ff6b5e"}
            pulse={connected}
          />
          <Tile
            label="Trend lane"
            value={trendRunning === true ? "RUNNING" : trendRunning === false ? "OFFLINE" : "UNKNOWN"}
            detail={`account snapshot ${ago(health?.lanes?.account_state_age_s ?? payload?.status?.account_snapshot_age_s ?? undefined)}`}
            color={trendRunning ? "#3ee287" : "#ffc23e"}
            pulse={trendRunning === true}
          />
          <Tile
            label="Tier 7"
            value={tier7Running === true ? "RUNNING" : tier7Running === false ? "OFFLINE" : "UNKNOWN"}
            detail={`${tier7?.current_action ?? tier7?.running_reason ?? "waiting for snapshot"} · cycle #${tier7?.last_cycle?.cycle_count ?? "—"}`}
            color={tier7Running ? "#3ee287" : "#ffc23e"}
            pulse={tier7Running === true}
          />
          <Tile
            label="Held up"
            value={halted ? "HALTED" : alertCount ? `${alertCount} ALERT` : heldFeeds.length ? `${heldFeeds.length} QUIET LOGS` : quietActive.length ? `${quietActive.length} QUIET TASKS` : "NO BLOCKER"}
            detail={health?.alerts?.active?.[0]?.message ?? tier7?.running_reason ?? "No active hold detected"}
            color={halted || heldFeeds.length || alertCount || quietActive.length ? "#ffc23e" : "#3ee287"}
          />
        </div>
      </Card>

      <OperatorPanel operator={operator} onRun={runOperatorEpoch} running={operatorBusy} result={operatorResult} />

      <div className="flex items-center justify-between px-1">
        <div>
          <div className="eyebrow tracking-[0.16em]">Incident queue</div>
          <div className="mt-1 font-mono text-[11px] text-faint">
            Proof and recovery stay attached to each live issue.
            {controlUnavailable ? " Control actions are disabled because the server write path is offline." : armed ? " Control write path is armed." : " Start-loop/unhalt actions require ARM in Operator Control."}
          </div>
        </div>
        <Badge color={activeIncidents.length ? "#ffc23e" : "#3ee287"} dot pulse={!activeIncidents.length}>
          {activeBadge}
        </Badge>
      </div>

      {activeIncidents.length ? (
        <div className="grid gap-4">
          {activeIncidents.map((incident) => <IncidentCard key={incident.id} incident={incident} />)}
        </div>
      ) : (
        <Card className="p-5">
          <SectionTitle right={<Badge color="#3ee287" dot pulse>NO ACTIVE INCIDENTS</Badge>}>
            Activity clear
          </SectionTitle>
          <div className="py-6 text-center font-mono text-[12px] text-faint">
            No active alert, held log, quiet task, or denied control action needs operator recovery.
          </div>
        </Card>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <Card className="p-3">
          <SectionTitle right={<span className="font-mono text-[10px] text-faint tnum">{recentContext.length} rows</span>}>
            Recent context
          </SectionTitle>
          {!recentContext.length ? (
            <div className="py-5 text-center font-mono text-[11px] text-faint">No recent self-heal or control decisions.</div>
          ) : (
            <div className="grid gap-1.5">
              {recentContext.map((row) => (
                <div key={row.id} className="grid grid-cols-[64px_auto_minmax(0,1fr)] items-center gap-2 border-t py-1.5 font-mono text-[10.5px] hairline first:border-t-0">
                  <span className="text-faint tnum">{row.at}</span>
                  <StatusDot color={row.color} />
                  <span className="truncate text-dim" title={row.text}>{row.text}</span>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card className="p-3">
          <SectionTitle right={<span className="font-mono text-[10px] text-faint tnum">{logFeeds.length} feeds</span>}>
            Raw evidence drawers
          </SectionTitle>
          <div className="grid gap-2">
            {logFeeds.map((feed) => {
              const tier7HeartbeatFreshForFeed = feed.id === "tier7" && tier7HeartbeatFresh;
              return (
                <LogDrawer
                  key={feed.id}
                  feed={feed}
                  stateOverride={tier7HeartbeatFreshForFeed ? { label: "HEARTBEAT", color: "#3ee287", pulse: true } : undefined}
                />
              );
            })}
          </div>
        </Card>
      </div>
    </div>
  );
}
