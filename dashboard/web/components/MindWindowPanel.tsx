"use client";
import { useState, type ReactNode } from "react";
import { usePoll } from "@/lib/api";
import { acceptProposal, denyProposal } from "@/lib/control";
import type {
  MindWindow,
  MindWindowCycle,
  MindWindowOutcome,
  MindWindowPendingProposal,
  MindWindowTier,
} from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime, ago } from "@/lib/format";

const TIER_COLOR: Record<string, string> = {
  operational: "#8b98a9",
  deescalation: "#22d3ee",
  escalation: "#f5b14c",
};
const TIER_LABEL: Record<string, string> = {
  operational: "OPERATIONAL",
  deescalation: "DE-ESCALATION",
  escalation: "ESCALATION",
};

function TierBadge({ tier }: { tier?: MindWindowTier }) {
  const key = tier ?? "unknown";
  const color = TIER_COLOR[key] ?? "#5a6677";
  const label = TIER_LABEL[key] ?? key.toUpperCase();
  return <Badge color={color}>{label}</Badge>;
}

function outcomeStatus(outcome: MindWindowOutcome): { label: string; color: string } {
  if (outcome.denied) return { label: "DENIED", color: "#ff6b5e" };
  if (outcome.proposal) return { label: "PROPOSED TO OPERATOR", color: "#f5b14c" };
  if (outcome.executed) return { label: "EXECUTED", color: "#3ee287" };
  if (outcome.shadow) return { label: "SHADOW-LOGGED (not executed)", color: "#22d3ee" };
  return { label: "NO-OP", color: "#5a6677" };
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between border-t py-1.5 hairline first:border-t-0">
      <span className="font-mono text-[11px] text-faint">{label}</span>
      <span className="min-w-0 truncate text-right font-mono text-[12px] text-text">{value}</span>
    </div>
  );
}

function AutonomyBadge({ enabled }: { enabled: boolean }) {
  return (
    <Badge color={enabled ? "#f59e0b" : "#8b98a9"} dot pulse={enabled}>
      {enabled ? "LIVE-OPERATIONAL" : "SHADOW"}
    </Badge>
  );
}

function summarizeObservations(observations: MindWindowCycle["observations"]): {
  total: number;
  ok: number;
  errored: string[];
} {
  const entries = Object.entries(observations ?? {});
  const errored = entries.filter(([, v]) => v && v.ok === false).map(([k]) => k);
  return { total: entries.length, ok: entries.length - errored.length, errored };
}

function OutcomeRow({ outcome }: { outcome: MindWindowOutcome }) {
  const status = outcomeStatus(outcome);
  return (
    <div className="flex flex-col gap-1 border-t py-2 hairline first:border-t-0">
      <div className="flex items-center justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[12px] text-text" title={outcome.action}>
          {outcome.action}
        </span>
        <span className="flex shrink-0 items-center gap-1.5">
          <TierBadge tier={outcome.tier} />
          <Badge color={status.color} dot>{status.label}</Badge>
        </span>
      </div>
      {outcome.detail && Object.keys(outcome.detail).length > 0 && (
        <div className="truncate font-mono text-[10.5px] text-faint" title={JSON.stringify(outcome.detail)}>
          {Object.entries(outcome.detail).slice(0, 3).map(([k, v]) => `${k}=${String(v)}`).join(" · ")}
        </div>
      )}
    </div>
  );
}

type RowStage = "idle" | "confirm-accept" | "confirm-deny" | "busy" | "done";

function ProposalActionBar({
  proposal,
  onResolved,
}: {
  proposal: MindWindowPendingProposal;
  onResolved: () => void;
}) {
  const [stage, setStage] = useState<RowStage>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const proposalId = proposal.proposal_id;

  if (!proposalId) {
    // Older cycles logged before proposal_id existed -- can't be actioned here.
    return <span className="font-mono text-[10px] text-faint">no id — re-propose to act</span>;
  }

  async function doAccept() {
    setStage("busy");
    const result = await acceptProposal(proposalId!, proposal.action);
    setStage("done");
    setMessage(
      result.ok
        ? "accepted — executing via existing safeguard"
        : `refused (${result.status}): ${String(result.data?.detail ?? result.data?.error ?? "denied")}`,
    );
    onResolved();
  }

  async function doDeny() {
    setStage("busy");
    const result = await denyProposal(proposalId!);
    setStage("done");
    setMessage(result.ok ? "denied" : `deny failed (${result.status})`);
    onResolved();
  }

  if (stage === "done") {
    return <span className="font-mono text-[10px] text-faint">{message}</span>;
  }
  if (stage === "busy") {
    return <span className="font-mono text-[10px] text-faint">working…</span>;
  }
  if (stage === "confirm-accept") {
    return (
      <span className="flex shrink-0 items-center gap-1.5">
        <span className="font-mono text-[10px] text-warn">really accept + execute?</span>
        <button
          className="rounded border border-pos/40 px-1.5 py-0.5 font-mono text-[10px] text-pos hover:bg-pos/10"
          onClick={doAccept}
        >
          Yes, execute
        </button>
        <button
          className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-faint hairline hover:bg-white/5"
          onClick={() => setStage("idle")}
        >
          Cancel
        </button>
      </span>
    );
  }
  if (stage === "confirm-deny") {
    return (
      <span className="flex shrink-0 items-center gap-1.5">
        <span className="font-mono text-[10px] text-faint">deny this proposal?</span>
        <button
          className="rounded border border-neg/40 px-1.5 py-0.5 font-mono text-[10px] text-neg hover:bg-neg/10"
          onClick={doDeny}
        >
          Yes, deny
        </button>
        <button
          className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-faint hairline hover:bg-white/5"
          onClick={() => setStage("idle")}
        >
          Cancel
        </button>
      </span>
    );
  }
  return (
    <span className="flex shrink-0 items-center gap-1.5">
      <button
        className="rounded border border-pos/40 px-1.5 py-0.5 font-mono text-[10px] text-pos hover:bg-pos/10"
        onClick={() => setStage("confirm-accept")}
      >
        Accept
      </button>
      <button
        className="rounded border border-neg/40 px-1.5 py-0.5 font-mono text-[10px] text-neg hover:bg-neg/10"
        onClick={() => setStage("confirm-deny")}
      >
        Deny
      </button>
    </span>
  );
}

function PendingProposalRow({
  proposal,
  onResolved,
}: {
  proposal: MindWindowPendingProposal;
  onResolved: () => void;
}) {
  return (
    <div className="flex flex-col gap-1.5 border-t py-2 hairline first:border-t-0">
      <div className="flex items-center justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[12px] font-semibold text-warn" title={proposal.action}>
          {proposal.action}
        </span>
        <span className="flex shrink-0 items-center gap-1.5">
          <TierBadge tier={proposal.tier} />
          <span className="font-mono text-[10px] text-faint tnum">{shortTime(proposal.ts)}</span>
        </span>
      </div>
      {proposal.detail && Object.keys(proposal.detail).length > 0 && (
        <div className="truncate font-mono text-[10.5px] text-faint" title={JSON.stringify(proposal.detail)}>
          {Object.entries(proposal.detail).slice(0, 3).map(([k, v]) => `${k}=${String(v)}`).join(" · ")}
        </div>
      )}
      <div className="flex justify-end">
        <ProposalActionBar proposal={proposal} onResolved={onResolved} />
      </div>
    </div>
  );
}

function ResolvedProposalRow({ proposal }: { proposal: MindWindowPendingProposal }) {
  const disposition = proposal.disposition;
  const isAccepted = disposition?.status === "accepted";
  return (
    <div className="flex flex-col gap-1 border-t py-1.5 hairline first:border-t-0">
      <div className="flex items-center justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[11px] text-dim" title={proposal.action}>
          {proposal.action}
        </span>
        <span className="flex shrink-0 items-center gap-1.5">
          <Badge color={isAccepted ? "#3ee287" : "#8b98a9"}>{isAccepted ? "ACCEPTED" : "DENIED"}</Badge>
          <span className="font-mono text-[10px] text-faint tnum">
            {disposition ? shortTime(disposition.decided_at) : ""}
          </span>
        </span>
      </div>
      {disposition?.reason && (
        <div className="truncate font-mono text-[10px] text-faint" title={disposition.reason}>
          {disposition.reason}
        </div>
      )}
    </div>
  );
}

function CycleDetail({ cycle }: { cycle: MindWindowCycle }) {
  const obs = summarizeObservations(cycle.observations);
  const diag = cycle.diagnosis;
  const outcomes = cycle.outcomes ?? [];
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline">
        <div className="eyebrow mb-2 tracking-[0.14em]">Noticed</div>
        <Row label="Tools read" value={`${obs.total}`} />
        <Row label="OK" value={<span className="text-pos">{obs.ok}</span>} />
        <Row
          label="Errored"
          value={obs.errored.length ? <span className="text-neg">{obs.errored.join(", ")}</span> : <span className="text-dim">none</span>}
        />
        <Row label="Cycle" value={cycle.cycle_id} />
        <Row label="At" value={shortTime(cycle.ts)} />
      </div>

      <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline">
        <div className="mb-2 flex items-center justify-between">
          <div className="eyebrow tracking-[0.14em]">Believes</div>
          {!diag?.available && <Badge color="#5a6677">DEGRADED</Badge>}
        </div>
        {!diag?.available ? (
          <div className="py-2 font-mono text-[11px] text-faint">
            claude CLI unavailable this cycle — no diagnosis.
          </div>
        ) : (
          <div className="grid gap-2 font-mono text-[11px] leading-5 text-dim">
            <div>
              <div className="mb-0.5 text-[10px] text-faint">Reasoning</div>
              <div className="text-text">{diag.reasoning || "—"}</div>
            </div>
            <div>
              <div className="mb-0.5 text-[10px] text-faint">Beliefs</div>
              <div className="text-text">{diag.beliefs || "—"}</div>
            </div>
          </div>
        )}
      </div>

      <div className="min-w-0 rounded-md border bg-black/10 p-3 hairline lg:col-span-2">
        <div className="eyebrow mb-1 tracking-[0.14em]">Did / would do</div>
        {!outcomes.length ? (
          <div className="py-2 font-mono text-[11px] text-faint">No actions proposed this cycle.</div>
        ) : (
          <div>{outcomes.map((o, i) => <OutcomeRow key={`${cycle.cycle_id}-outcome-${i}-${o.action}`} outcome={o} />)}</div>
        )}
      </div>
    </div>
  );
}

function CompactCycleRow({ cycle }: { cycle: MindWindowCycle }) {
  const obs = summarizeObservations(cycle.observations);
  const outcomes = cycle.outcomes ?? [];
  const executedCount = outcomes.filter((o) => o.executed).length;
  const proposalCount = outcomes.filter((o) => o.proposal).length;
  const deniedCount = outcomes.filter((o) => o.denied).length;
  return (
    <div className="flex items-center gap-2 border-t py-1.5 font-mono text-[11px] hairline first:border-t-0">
      <StatusDot color={deniedCount ? "#ff6b5e" : proposalCount ? "#f5b14c" : executedCount ? "#3ee287" : "#5a6677"} />
      <span className="w-[130px] shrink-0 text-faint tnum">{shortTime(cycle.ts)}</span>
      <span className="shrink-0 text-faint">{obs.ok}/{obs.total} ok</span>
      <span className="min-w-0 flex-1 truncate text-dim" title={cycle.diagnosis?.reasoning ?? ""}>
        {cycle.diagnosis?.available ? (cycle.diagnosis.reasoning || "—") : "no diagnosis"}
      </span>
      {!!executedCount && <span className="shrink-0 text-pos">{executedCount} exec</span>}
      {!!proposalCount && <span className="shrink-0 text-warn">{proposalCount} proposed</span>}
      {!!deniedCount && <span className="shrink-0 text-neg">{deniedCount} denied</span>}
    </div>
  );
}

export function MindWindowPanel() {
  const { data, loading, error, reload } = usePoll<MindWindow>("/api/mind_window", 5000);

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Mind Window</SectionTitle><Loading label="Loading resident loop cycles..." /></Card>;
  }
  if (!data) {
    return <Card className="p-3"><SectionTitle>Mind Window</SectionTitle><NotConnected label="Mind-window endpoint unavailable" hint={error ?? undefined} /></Card>;
  }

  const older = data.recent_cycles.slice(1);
  const pending = data.pending_operator_proposals ?? [];
  const resolved = (data.resolved_proposals ?? []).slice(0, 10);

  return (
    <Card className="flex flex-col">
      <SectionTitle right={<AutonomyBadge enabled={data.autonomy_enabled} />}>
        Mind Window · resident reasoning loop
      </SectionTitle>
      <div className="flex flex-col gap-4 p-3">
        {!data.has_run ? (
          <NotConnected
            label="No resident loop cycle has run yet"
            hint="agent_runtime_loop is built and gated but has not run a cycle in production — this is the honest starting state, not a fault."
          />
        ) : (
          <>
            {data.last_cycle && (
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="eyebrow tracking-[0.16em]">Most recent cycle</span>
                  <span className="font-mono text-[10px] text-faint tnum">{ago(
                    (Date.now() - new Date(data.last_cycle.ts).getTime()) / 1000
                  )}</span>
                </div>
                <CycleDetail cycle={data.last_cycle} />
              </div>
            )}

            {!!older.length && (
              <div className="rounded-md border bg-black/10 p-3 hairline">
                <div className="eyebrow mb-1 tracking-[0.14em]">Earlier cycles ({older.length})</div>
                <div className="scroll-thin max-h-[220px] overflow-auto">
                  {older.map((cycle) => <CompactCycleRow key={cycle.cycle_id} cycle={cycle} />)}
                </div>
              </div>
            )}
          </>
        )}

        <div className="rounded-md border bg-black/10 p-3 hairline">
          <div className="mb-1 flex items-center justify-between">
            <span className="eyebrow tracking-[0.16em]">Needs from you</span>
            <Badge color={pending.length ? "#f5b14c" : "#3ee287"} dot pulse={!!pending.length}>
              {pending.length ? `${pending.length} PENDING` : "NOTHING PENDING"}
            </Badge>
          </div>
          {!pending.length ? (
            <div className="py-3 text-center font-mono text-[11px] text-faint">
              No escalation actions awaiting operator approval.
            </div>
          ) : (
            <div>
              {pending.map((p, i) => (
                <PendingProposalRow
                  key={p.proposal_id ?? `${p.cycle_id ?? "pending"}-${i}-${p.action}`}
                  proposal={p}
                  onResolved={reload}
                />
              ))}
            </div>
          )}
        </div>

        {!!resolved.length && (
          <div className="rounded-md border bg-black/10 p-3 hairline">
            <div className="eyebrow mb-1 tracking-[0.14em]">Recent decisions</div>
            <div className="scroll-thin max-h-[160px] overflow-auto">
              {resolved.map((p, i) => (
                <ResolvedProposalRow key={p.proposal_id ?? `${p.cycle_id ?? "resolved"}-${i}-${p.action}`} proposal={p} />
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
