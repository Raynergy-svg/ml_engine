"use client";
import { usePoll } from "@/lib/api";
import type { LearningLoop } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime } from "@/lib/format";

function decisionColor(decision?: string) {
  if (decision === "accepted") return "#2bd17e";
  if (decision === "rejected") return "#f5b14c";
  return "#5a6677";
}

export function LearningLoopPanel() {
  const { data, loading } = usePoll<LearningLoop>("/api/learning_loop", 10000);

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Continual Learning</SectionTitle><Loading /></Card>;
  }
  if (!data) {
    return <Card className="p-3"><SectionTitle>Continual Learning</SectionTitle><NotConnected label="Learning loop data unavailable" /></Card>;
  }

  const last = data.last_cycle;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="flex flex-col">
        <SectionTitle right={
          <span className="flex items-center gap-1.5">
            <Badge color="#5a6677" dot>
              {data.consumed_by_live_sizing ? "LIVE SIZING" : "SHADOW — advisory only"}
            </Badge>
            <Badge color={data.pending_retrain_markers > 0 ? "#f5b14c" : "#2bd17e"} dot>
              {data.pending_retrain_markers} MARKER{data.pending_retrain_markers === 1 ? "" : "S"} PENDING
            </Badge>
            <Badge color={data.has_run ? decisionColor(last?.decision) : "#5a6677"} dot>
              {data.has_run ? (last?.decision ?? "—").toUpperCase() : "NOT YET RUN"}
            </Badge>
          </span>
        }>
          Market-closed learning cycle · risk/calibration only
        </SectionTitle>
        <div className="p-3">
          {!data.has_run ? (
            <NotConnected
              label="No offline cycle recorded yet"
              hint="scripts/offline_learning_cycle.py runs market-closed on a schedule — this is the honest starting state, not a fault."
            />
          ) : (
            <div className="font-mono text-[12px]">
              <div className="flex items-center justify-between border-t py-1.5 hairline first:border-t-0">
                <span className="text-faint">decision</span><span className="text-text">{last?.decision}</span>
              </div>
              <div className="flex items-center justify-between border-t py-1.5 hairline">
                <span className="text-faint">calibration error (Brier)</span>
                <span className="text-text tnum">
                  {last?.incumbent_brier != null ? last.incumbent_brier.toFixed(4) : "—"}
                  {" → "}
                  {last?.candidate_brier != null ? last.candidate_brier.toFixed(4) : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between border-t py-1.5 hairline">
                <span className="text-faint">max size multiplier</span>
                <span className="text-text tnum">{last?.max_size_multiplier != null ? last.max_size_multiplier.toFixed(2) : "—"} (≤ 1.0 always)</span>
              </div>
              <div className="flex items-center justify-between border-t py-1.5 hairline">
                <span className="text-faint">markers drained / new outcomes</span>
                <span className="text-text tnum">{last?.markers_drained ?? 0} / {last?.new_outcomes ?? 0}</span>
              </div>
              <div className="flex items-center justify-between border-t py-1.5 hairline">
                <span className="text-faint">time</span><span className="text-text tnum">{shortTime(last?.timestamp)}</span>
              </div>
              <div className="mt-2 text-[10px] text-faint">scope: {data.scope}</div>
            </div>
          )}
        </div>
      </Card>

      <Card className="flex flex-col">
        <SectionTitle right={<Badge color="#5a6677" dot>{data.recent_cycles?.length ?? 0} CYCLES</Badge>}>
          Recent cycle history
        </SectionTitle>
        <div className="p-3">
          {!data.recent_cycles?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">No cycles recorded yet.</div>
          ) : (
            <div className="font-mono text-[11px]">
              {data.recent_cycles.map((c, i) => (
                <div key={i} className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0">
                  <StatusDot color={decisionColor(c.decision)} />
                  <div className="min-w-0 flex-1">
                    <span className="text-text">{c.decision}</span>
                    {c.candidate_brier != null && c.incumbent_brier != null && (
                      <span className="text-faint"> · brier {c.incumbent_brier.toFixed(4)}→{c.candidate_brier.toFixed(4)}</span>
                    )}
                    <div className="text-[10px] text-faint">{c.markers_drained ?? 0} marker(s) drained · {c.new_outcomes ?? 0} new outcome(s)</div>
                  </div>
                  <span className="shrink-0 text-[10px] text-faint tnum">{shortTime(c.timestamp)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
