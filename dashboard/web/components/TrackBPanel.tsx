"use client";
import { isValidElement, type ReactNode } from "react";
import { usePoll } from "@/lib/api";
import type { TrackB, LaneStatus } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime, fmtPct, fmtSigned, pnlClass } from "@/lib/format";

// Contract-drift guard: a disk-backed endpoint that returns an object where a
// scalar is expected must NOT crash the cockpit ("Objects are not valid as a
// React child"). Primitives and real React nodes pass through untouched; a
// stray plain object is stringified as a last resort instead of throwing.
function safeChild(value: ReactNode): ReactNode {
  if (value == null || typeof value !== "object") return value;
  if (isValidElement(value) || Array.isArray(value)) return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function Row({ label, value, mono = true }: { label: string; value: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between border-t py-1.5 hairline first:border-t-0">
      <span className="font-mono text-[11px] text-faint">{label}</span>
      <span className={`text-[12px] text-text ${mono ? "font-mono tnum" : ""}`}>{safeChild(value)}</span>
    </div>
  );
}

function BookSide({ title, weights, color }: { title: string; weights: Record<string, number>; color: string }) {
  const entries = Object.entries(weights ?? {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return (
    <div className="min-w-0 flex-1">
      <div className="mb-1 font-mono text-[10.5px] text-faint">{title} ({entries.length})</div>
      {!entries.length ? (
        <div className="py-2 font-mono text-[11px] text-dim">none</div>
      ) : (
        <div className="font-mono text-[11px]">
          {entries.slice(0, 12).map(([sym, w]) => (
            <div key={sym} className="flex items-center justify-between border-t py-1 hairline first:border-t-0">
              <span className="truncate text-text">{sym}</span>
              <span className="tnum" style={{ color }}>{fmtPct(w * 100, 2)}</span>
            </div>
          ))}
          {entries.length > 12 && (
            <div className="pt-1 text-[10px] text-faint">+{entries.length - 12} more</div>
          )}
        </div>
      )}
    </div>
  );
}

export function TrackBPanel() {
  const { data, loading } = usePoll<TrackB>("/api/track_b", 5000);
  const { data: lanes } = usePoll<LaneStatus>("/api/lanes", 4000);
  const laneHalted = lanes?.lanes?.track_b ?? null;

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Track B — Filing-Text Research Alpha</SectionTitle><Loading /></Card>;
  }
  if (!data) {
    return <Card className="p-3"><SectionTitle>Track B — Filing-Text Research Alpha</SectionTitle><NotConnected label="Track B data unavailable" /></Card>;
  }

  const lg = data.live_gate;
  const c = data.construction;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="flex flex-col">
        <SectionTitle right={
          <span className="flex items-center gap-1.5">
            <Badge color={laneHalted === true ? "#ff4d6d" : laneHalted === false ? "#2bd17e" : "#5a6677"} dot>
              lane {laneHalted === true ? "HALTED" : laneHalted === false ? "unhalted" : "—"}
            </Badge>
            <Badge color={lg.armed ? "#ff4d6d" : "#22d3ee"} dot>
              {lg.armed ? "LIVE (armed)" : "SHADOW"}
            </Badge>
          </span>
        }>
          Track B · SEC filing-text research alpha, shadow lane
        </SectionTitle>
        <div className="p-3">
          <Row label="Scored filings (post-cutoff)" value={
            <span className={data.n_scored_filings != null && c && data.n_scored_filings < c.power_required_n_filings ? "text-warn" : ""}>
              {data.n_scored_filings ?? "—"} {c ? `/ ~${c.power_required_n_filings} needed` : ""}
            </span>
          } />
          {!data.has_run ? (
            <NotConnected
              label="No shadow cycle recorded yet"
              hint="Lane is built and gated but has not run a cycle in production — this is the honest starting state, not a fault. Run scripts/run_track_b_shadow.py to start accumulating a forward record."
            />
          ) : (
            <>
              <Row label="Forward cycles recorded" value={data.n_forward_cycles} />
              <Row label="First forward asof" value={data.first_asof_date ?? "—"} />
              <Row label="Last forward asof" value={data.last_asof_date ?? "—"} />
              <Row label="Cumulative shadow return" value={
                <span className={pnlClass(data.cumulative_shadow_return)}>{fmtSigned(data.cumulative_shadow_return * 100, 2)}%</span>
              } />
              <Row label="Forward Sharpe (annualized)" value={
                data.forward_sharpe_annualized != null ? data.forward_sharpe_annualized.toFixed(2)
                  : "n/a (n<2 cycles)"
              } />
              <Row label="Current gross leverage" value={data.current_gross_leverage != null ? `${data.current_gross_leverage.toFixed(2)}x` : "—"} />
              <Row label="LiveGate armed" value={lg.armed ? "YES" : "no"} />
              {!lg.available && (
                <div className="py-2 font-mono text-[10.5px] text-faint">
                  No live_gate_state.json on disk — lane has never been armed; the frozen harness&apos;s own
                  verdict for this signal is INSUFFICIENT, not REAL.
                </div>
              )}
            </>
          )}
        </div>
      </Card>

      <Card className="flex flex-col">
        <SectionTitle right={<Badge color="#f5b14c" dot>UNDERPOWERED, INSUFFICIENT</Badge>}>
          Frozen construction (reused verbatim — never re-derived)
        </SectionTitle>
        <div className="p-3">
          {!c ? (
            <NotConnected label="Construction manifest appears once scores are available" />
          ) : (
            <>
              <Row label="Signal" value={c.signal} mono={false} />
              <Row label="Composite weights" value={
                `quality ${c.composite_weights.fundamental_quality} / red-flags ${c.composite_weights.accounting_red_flags} / outlook ${c.composite_weights.forward_outlook}`
              } />
              <Row label="Book" value={c.book} mono={false} />
              <Row label="Rebalance" value={`every ${c.rebalance_step_days}d (held between scoring batches)`} />
              <Row label="Vol target (ann.)" value={fmtPct(c.vol_target_ann * 100, 0)} />
              <Row label="Leverage cap" value={`${c.max_leverage}x`} />
              <Row label="Cost (per side)" value={`${c.cost_bps} bps`} />
              <Row label="Model cutoff" value={c.model_cutoff} />
              <Row label="Ship gate verdict" value={c.gate_verdict} mono={false} />
              <div className="mt-2 border-t pt-2 font-mono text-[10.5px] text-warn hairline">
                {c.coverage_note}
              </div>
              <div className="mt-2 border-t pt-2 font-mono text-[10px] text-faint hairline">
                Sources: {c.source_prereg_doc} · {c.source_scaleup_doc} · {c.source_adversarial_review_doc}
              </div>
            </>
          )}
        </div>
      </Card>

      <Card className="flex flex-col">
        <SectionTitle>Current would-be book (as of {data.current_asof ?? "—"})</SectionTitle>
        <div className="flex gap-4 p-3">
          <BookSide title="Longs (Q5, equal-weight)" weights={data.current_book?.longs ?? {}} color="#2bd17e" />
          <BookSide title="Shorts (none — long-only book)" weights={data.current_book?.shorts ?? {}} color="#ff4d6d" />
        </div>
      </Card>

      <Card className="flex flex-col">
        <SectionTitle right={<Badge color={data.n_forward_cycles ? "#2bd17e" : "#5a6677"} dot>{data.n_forward_cycles} CYCLES</Badge>}>
          Forward shadow cycles (mark-to-market)
        </SectionTitle>
        <div className="p-3">
          {!data.recent_cycles?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">No forward cycles recorded yet.</div>
          ) : (
            <div className="font-mono text-[11px]">
              {data.recent_cycles.map((cy) => (
                <div key={cy.forward_cycle_seq} className="flex items-center gap-2 border-t py-1.5 hairline first:border-t-0">
                  <StatusDot color={cy.today_net_return >= 0 ? "#2bd17e" : "#ff4d6d"} />
                  <span className="w-10 shrink-0 text-faint">#{cy.forward_cycle_seq}</span>
                  <span className="w-24 shrink-0 text-faint tnum">{cy.asof_date}</span>
                  <span className="shrink-0 text-faint">{cy.n_longs}L ({cy.n_scored_filings} scored)</span>
                  <span className={`min-w-0 flex-1 tnum ${pnlClass(cy.today_net_return)}`}>{fmtSigned(cy.today_net_return * 100, 3)}%</span>
                  <span className="shrink-0 text-faint tnum">cum {fmtSigned(cy.cumulative_shadow_return * 100, 2)}%</span>
                  <span className="shrink-0 text-[10px] text-faint tnum">{shortTime(cy.cycle_ts)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
