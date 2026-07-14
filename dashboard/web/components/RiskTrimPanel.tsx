"use client";

import { usePoll } from "@/lib/api";
import type { RiskTrim } from "@/lib/types";
import { fmtMoney, fmtNum, shortTime } from "@/lib/format";
import { Badge, Card, Loading, NotConnected, SectionTitle } from "./ui";

export function RiskTrimPanel() {
  const { data, loading } = usePoll<RiskTrim>("/api/risk_trim", 10_000);
  if (loading && !data)
    return (
      <Card>
        <SectionTitle>Bucket Trim Proposal</SectionTitle>
        <Loading />
      </Card>
    );
  if (!data || !data.connected)
    return (
      <Card>
        <SectionTitle>Bucket Trim Proposal</SectionTitle>
        <NotConnected
          label="Risk trim readout unavailable"
          hint={data?.reason ?? "Waiting for the read-only OANDA client."}
        />
      </Card>
    );

  const over = data.over_cap_buckets ?? [];
  const candidates = data.candidates ?? [];
  return (
    <Card className="flex flex-col">
      <SectionTitle
        right={
          <span className="flex items-center gap-1.5">
            <Badge
              color={data.status === "over_cap" ? "#f5b14c" : "#2bd17e"}
              dot
            >
              {data.status === "over_cap" ? "OVER CAP" : "CLEAR"}
            </Badge>
            <Badge color="#22d3ee">PROPOSAL ONLY</Badge>
          </span>
        }
      >
        Bucket Trim Proposal
      </SectionTitle>
      <div className="grid gap-3 p-3 lg:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 flex items-center justify-between gap-2 font-mono text-[10.5px] text-faint">
            <span>Over-cap buckets</span>
            <span className="tnum">
              cap {fmtMoney(data.cap_home, 2)} · {shortTime(data.asof)}
            </span>
          </div>
          {!over.length ? (
            <div className="rounded-md border border-pos/25 bg-pos/10 px-3 py-2 font-mono text-[11px] text-pos">
              Current trend buckets are inside the 2R cap.
            </div>
          ) : (
            over.map((bucket) => (
              <div
                key={bucket.bucket}
                className="border-t py-1.5 font-mono text-[11px] hairline"
              >
                <div className="flex justify-between gap-2">
                  <span className="text-text">
                    {bucket.currency} {bucket.direction}
                  </span>
                  <span className="text-warn">
                    +{fmtMoney(bucket.over_home, 2)}
                  </span>
                </div>
                <div className="text-[10px] text-faint">
                  {bucket.instruments.join(", ")} · cap{" "}
                  {fmtMoney(bucket.cap_home, 2)}
                </div>
              </div>
            ))
          )}
        </div>
        <div className="min-w-0">
          <div className="mb-1 flex items-center justify-between gap-2 font-mono text-[10.5px] text-faint">
            <span>Deterministic reduce candidates</span>
            <span className="tnum">
              {data.open_trade_count ?? 0} open trades
            </span>
          </div>
          {!candidates.length ? (
            <div className="rounded-md border px-3 py-2 font-mono text-[11px] text-dim hairline">
              No trim candidate is needed right now.
            </div>
          ) : (
            candidates.slice(0, 5).map((candidate) => (
              <div
                key={`${candidate.trade_id}-${candidate.instrument}`}
                className="border-t py-1.5 font-mono text-[11px] hairline"
              >
                <div className="flex justify-between gap-2">
                  <span className="text-text">
                    {candidate.instrument.replace("_", "/")}
                  </span>
                  <span className="text-warn">
                    {fmtNum(candidate.reduce_units, 0)}u
                  </span>
                </div>
                <div className="text-[10px] text-faint">
                  trade #{candidate.trade_id} · est. reduction{" "}
                  {fmtMoney(candidate.estimated_risk_reduction_home, 2)}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
      <div className="border-t px-3 py-2 font-mono text-[10px] text-faint hairline">
        {data.note ?? "Read-only proposal. No broker order has been placed."}
      </div>
    </Card>
  );
}
