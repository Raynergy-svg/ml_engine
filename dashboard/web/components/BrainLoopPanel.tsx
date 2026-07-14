"use client";
import { usePoll } from "@/lib/api";
import type { BrainLoop, LaneStatus } from "@/lib/types";
import {
  Card,
  SectionTitle,
  NotConnected,
  Loading,
  Badge,
  StatusDot,
} from "./ui";
import { shortTime } from "@/lib/format";

export function BrainLoopPanel() {
  const { data, loading } = usePoll<BrainLoop>("/api/brain_loop", 5000);
  const { data: lanes } = usePoll<LaneStatus>("/api/lanes", 4000);
  const laneHalted = lanes?.lanes?.brain ?? null;

  if (loading && !data) {
    return (
      <Card className="p-3">
        <SectionTitle>Brain Loop</SectionTitle>
        <Loading />
      </Card>
    );
  }
  if (!data) {
    return (
      <Card className="p-3">
        <SectionTitle>Brain Loop</SectionTitle>
        <NotConnected label="Brain loop data unavailable" />
      </Card>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="flex flex-col">
        <SectionTitle
          right={
            <span className="flex items-center gap-1.5">
              <Badge
                color={
                  laneHalted === true
                    ? "#ff4d6d"
                    : laneHalted === false
                      ? "#2bd17e"
                      : "#5a6677"
                }
                dot
              >
                lane{" "}
                {laneHalted === true
                  ? "HALTED"
                  : laneHalted === false
                    ? "unhalted"
                    : "—"}
              </Badge>
              <Badge color={data.has_run ? "#2bd17e" : "#5a6677"} dot>
                {data.has_run ? "HAS RUN" : "NOT YET RUN"}
              </Badge>
            </span>
          }
        >
          Brain Loop · autonomous research → promotion
        </SectionTitle>
        <div className="p-3">
          {!data.has_run ? (
            <NotConnected
              label="No cycle recorded yet"
              hint={`No readable ledger, promotion-request, or de-risk event. Sources: ${Object.values(data.source).join(" · ")}`}
            />
          ) : (
            <>
              <div className="mb-2 font-mono text-[10.5px] text-faint">
                Most recent ledger event:
              </div>
              <div className="font-mono text-[12px]">
                <div className="flex items-center justify-between border-t py-1.5 hairline first:border-t-0">
                  <span className="text-faint">kind</span>
                  <span className="text-text">
                    {data.last_event?.kind ?? "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between border-t py-1.5 hairline">
                  <span className="text-faint">hypothesis</span>
                  <span className="text-text">
                    {data.last_event?.hypothesis_id ??
                      data.last_event?.name ??
                      "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between border-t py-1.5 hairline">
                  <span className="text-faint">status / decision</span>
                  <span className="text-text">
                    {data.last_event?.status ??
                      data.last_event?.decision ??
                      "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between border-t py-1.5 hairline">
                  <span className="text-faint">time</span>
                  <span className="text-text tnum">
                    {shortTime(data.last_event?.ts)}
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </Card>

      <Card className="flex flex-col">
        <SectionTitle
          right={
            <Badge
              color={data.pending_operator_count > 0 ? "#f5b14c" : "#2bd17e"}
              dot
            >
              {data.pending_operator_count > 0
                ? `${data.pending_operator_count} PENDING`
                : "NONE PENDING"}
            </Badge>
          }
        >
          Promotion requests
        </SectionTitle>
        <div className="p-3">
          {!data.promotion_requests?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">
              No promotion requests on disk.
            </div>
          ) : (
            <div className="font-mono text-[11px]">
              {data.promotion_requests.map((r, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0"
                >
                  <StatusDot
                    color={
                      r.status === "PENDING_OPERATOR" ? "#f5b14c" : "#2bd17e"
                    }
                  />
                  <div className="min-w-0 flex-1">
                    <span className="text-text">{r.hypothesis_id}</span>
                    <span className="text-faint">
                      {" "}
                      · target={r.target} · {r.status}
                    </span>
                    {r.requires_operator_arm && (
                      <div className="text-[10px] text-warn">
                        requires operator ARM + start_loop — no auto-promotion
                        path exists
                      </div>
                    )}
                  </div>
                  <span className="shrink-0 text-[10px] text-faint tnum">
                    {shortTime(r.created_at)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      <Card className="flex flex-col lg:col-span-2">
        <SectionTitle
          right={
            <Badge
              color={data.derisk_audit?.length ? "#f5b14c" : "#2bd17e"}
              dot
            >
              {data.derisk_audit?.length ?? 0} EVENTS
            </Badge>
          }
        >
          De-risk / halt audit (src/brain_loop/derisk.py)
        </SectionTitle>
        <div className="p-3">
          {!data.derisk_audit?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">
              No de-risk or halt actions taken by the brain loop.
            </div>
          ) : (
            <div className="font-mono text-[11px]">
              {data.derisk_audit.map((e, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 border-t py-1.5 hairline first:border-t-0"
                >
                  <StatusDot
                    color={e.action === "halt" ? "#ff4d6d" : "#f5b14c"}
                  />
                  <div className="min-w-0 flex-1">
                    <span className="text-text">{e.action}</span>
                    {e.reason && (
                      <span className="text-faint"> · {e.reason}</span>
                    )}
                    {e.risk_pct != null && (
                      <span className="text-faint">
                        {" "}
                        · risk_pct {e.prior_risk_pct} → {e.risk_pct}
                      </span>
                    )}
                    <div className="text-[10px] text-faint">by {e.actor}</div>
                  </div>
                  <span className="shrink-0 text-[10px] text-faint tnum">
                    {shortTime(e.ts)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
