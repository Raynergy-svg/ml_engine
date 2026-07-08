"use client";
import { isValidElement, type ReactNode } from "react";
import { usePoll } from "@/lib/api";
import type { Hedge, HedgeLeg, HedgeProposal } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge, StatusDot } from "./ui";
import { shortTime, fmtNum, fmtSigned, pnlClass } from "@/lib/format";

// Contract-drift guard (see CryptoCarryPanel / TrackBPanel): a disk-backed
// endpoint that returns an object where a scalar is expected must NEVER crash
// the cockpit ("Objects are not valid as a React child" — this exact bug has
// taken this dashboard down before). Primitives and real React nodes pass
// through untouched; a stray plain object is stringified as a last resort.
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
    <div className="flex items-center justify-between gap-3 border-t py-1.5 hairline first:border-t-0">
      <span className="shrink-0 font-mono text-[11px] text-faint">{label}</span>
      <span className={`min-w-0 text-right text-[12px] text-text ${mono ? "font-mono tnum" : ""}`}>{safeChild(value)}</span>
    </div>
  );
}

// Signed home-$ notional per currency: green long (>0) / red short (<0), sorted
// by magnitude. Never renders the record object — only .map()ed scalar rows.
function SignedExposure({ label, book }: { label: string; book: Record<string, number> }) {
  const entries = Object.entries(book ?? {})
    .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return (
    <div className="min-w-0">
      <div className="mb-1 font-mono text-[10.5px] text-faint">{label} ({entries.length})</div>
      {!entries.length ? (
        <div className="py-2 font-mono text-[11px] text-dim">none</div>
      ) : (
        <div className="font-mono text-[11px]">
          {entries.map(([ccy, v]) => {
            const color = v > 0 ? "#2bd17e" : v < 0 ? "#ff4d6d" : "#8b98a9";
            return (
              <div key={ccy} className="flex items-center justify-between gap-2 border-t py-1 hairline first:border-t-0">
                <span className="flex items-center gap-1.5 truncate text-text">
                  <StatusDot color={color} />
                  {ccy}
                </span>
                <span className="tnum" style={{ color }}>
                  {v >= 0 ? "+" : "−"}${fmtNum(Math.abs(v), 0)}
                  <span className="ml-1 text-[10px] text-faint">{v > 0 ? "long" : v < 0 ? "short" : "flat"}</span>
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// The applied hedge is an OBJECT (style / legs[] / neutralizes / isolates / cost
// fields). We NEVER drop it into JSX — every field below is an extracted scalar,
// and legs are .map()ed into "direction instrument (risk_home)" strings.
function AppliedHedge({ status, hedge }: { status: string | null; hedge: HedgeProposal | null }) {
  if (!hedge) {
    const msg =
      status === "no_valid_hedge" ? "No structurally valid hedge could be constructed for the current book."
      : status === "fail_closed" ? "Exposure netting failed closed — no hedge proposed (honest refusal, not a zero)."
      : status === "unsupported_asset_class" ? "Asset class is not hedge-supported (fx / equity only)."
      : "No applied hedge for the current book.";
    return <div className="py-2 font-mono text-[10.5px] text-faint">{msg}</div>;
  }
  const legs: HedgeLeg[] = Array.isArray(hedge.legs) ? hedge.legs : [];
  return (
    <>
      <Row label="Hedge style" value={hedge.style ?? "—"} mono={false} />
      <Row label="Neutralizes" value={hedge.neutralizes ?? "—"} mono={false} />
      <Row
        label="Exposure reduction"
        value={
          hedge.exposure_reduction != null
            ? `$${fmtNum(hedge.exposure_reduction, 0)}${hedge.exposure_reduction_pct != null ? ` (${fmtNum(hedge.exposure_reduction_pct * 100, 1)}%)` : ""}`
            : "—"
        }
      />
      <Row
        label="Est. cost"
        value={
          hedge.cost_known && hedge.cost_estimate_dollars != null
            ? `$${fmtNum(hedge.cost_estimate_dollars, 0)}${hedge.cost_to_reduction_ratio != null ? ` · ${fmtNum(hedge.cost_to_reduction_ratio, 3)} cost/reduction` : ""}`
            : "unknown — no cost source (FX forwards / ETF marks unavailable)"
        }
      />
      <div className="mt-1.5 border-t pt-1.5 hairline">
        <div className="mb-1 font-mono text-[10.5px] text-faint">Legs ({legs.length})</div>
        {!legs.length ? (
          <div className="font-mono text-[11px] text-dim">none</div>
        ) : (
          <div className="font-mono text-[11px]">
            {legs.map((leg, i) => (
              <div key={`${leg.instrument}-${i}`} className="flex items-center justify-between gap-2 border-t py-1 hairline first:border-t-0">
                <span className="flex items-center gap-1.5 truncate text-text">
                  <StatusDot color={leg.direction === "short" ? "#ff4d6d" : "#2bd17e"} />
                  {leg.direction} {leg.instrument}
                </span>
                <span className="tnum text-faint">${fmtNum(leg.risk_home, 0)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      {/* isolates: the one-line "what view survives" narrative — the whole point */}
      <div className="mt-2 border-t pt-2 font-mono text-[10.5px] text-warn hairline">
        Isolates: {hedge.isolates ?? "—"}
      </div>
    </>
  );
}

const HEDGE_STATUS_COLOR: Record<string, string> = {
  applied: "#2bd17e",
  no_valid_hedge: "#f5b14c",
  fail_closed: "#ff4d6d",
  unsupported_asset_class: "#5a6677",
};

export function HedgePanel() {
  const { data, loading } = usePoll<Hedge>("/api/hedge", 5000);

  if (loading && !data) {
    return <Card className="p-3"><SectionTitle>Hedge / Exposure</SectionTitle><Loading /></Card>;
  }
  if (!data) {
    return <Card className="p-3"><SectionTitle>Hedge / Exposure</SectionTitle><NotConnected label="Hedge layer data unavailable" /></Card>;
  }

  const fx = data.live_fx_exposure;
  const hist = data.exposure_history;
  const statusColor = fx?.hedge_status ? (HEDGE_STATUS_COLOR[fx.hedge_status] ?? "#8b98a9") : "#5a6677";

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* ---- LIVE FX EXPOSURE ------------------------------------------- */}
      <Card className="flex flex-col">
        <SectionTitle right={
          <span className="flex items-center gap-1.5">
            <Badge color="#22d3ee" dot>SHADOW · ANALYSIS-ONLY</Badge>
            <Badge color={data.paper_only ? "#2bd17e" : "#ff4d6d"} dot>
              {data.paper_only ? "paper_only" : "PAPER OFF"}
            </Badge>
          </span>
        }>
          Live FX Exposure · signed currency netting + hedge proposal
        </SectionTitle>
        <div className="p-3">
          {!fx ? (
            <NotConnected
              label="Flat book — no live FX exposure"
              hint="No open OANDA trend positions to net right now (or the book is stale / uncomputable). Computed live each request — this is the honest flat-book state, not a fault."
            />
          ) : (
            <>
              <Row label="As of" value={fx.asof_date ?? "—"} />
              <Row label="NAV (home $)" value={fx.nav != null ? `$${fmtNum(fx.nav, 0)}` : "—"} />
              <Row
                label="Positions (resolved / total)"
                value={`${fx.resolved_position_count ?? "—"} / ${fx.position_count ?? "—"}`}
              />

              <div className="mt-2 border-t pt-2 hairline">
                <SignedExposure label="Net currency exposure (home $, signed)" book={fx.net_currency_exposure} />
              </div>

              {/* Concentration warnings — the "one USD bet wearing three
                  outfits" detector: rendered prominently, .map()ed strings. */}
              <div className="mt-3 border-t pt-2 hairline">
                <div className="mb-1 font-mono text-[10.5px] text-faint">Concentration warnings</div>
                {!fx.concentration_warnings?.length ? (
                  <div className="font-mono text-[11px] text-dim">none flagged</div>
                ) : (
                  <ul className="space-y-1">
                    {fx.concentration_warnings.map((w, i) => (
                      <li key={i} className="flex items-start gap-1.5 font-mono text-[11px] text-warn">
                        <span className="mt-1 shrink-0"><StatusDot color="#f5b14c" /></span>
                        <span className="min-w-0">{safeChild(w)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {/* Hedge status + applied proposal summary. */}
              <div className="mt-3 border-t pt-2 hairline">
                <div className="mb-1.5 flex items-center gap-1.5">
                  <Badge color={statusColor} dot>hedge {fx.hedge_status ?? "—"}</Badge>
                  {fx.hedge_decision && <Badge color="#8b98a9">{fx.hedge_decision}</Badge>}
                </div>
                <AppliedHedge status={fx.hedge_status} hedge={fx.applied_hedge} />
              </div>

              {fx.narrative?.length ? (
                <div className="mt-3 space-y-1 border-t pt-2 font-mono text-[10px] text-faint hairline">
                  {fx.narrative.map((line, i) => (
                    <p key={i}>{safeChild(line)}</p>
                  ))}
                </div>
              ) : null}
            </>
          )}
        </div>
      </Card>

      {/* ---- EXPOSURE HISTORY (training bridge) ------------------------- */}
      <Card className="flex flex-col">
        <SectionTitle right={
          <Badge color={hist.n_snapshots ? "#2bd17e" : "#5a6677"} dot>
            {hist.n_snapshots} SNAPSHOT{hist.n_snapshots === 1 ? "" : "S"}
          </Badge>
        }>
          Exposure History · P2 risk-target training bridge
        </SectionTitle>
        <div className="p-3">
          {!hist.n_snapshots ? (
            <NotConnected
              label="0 captured — capture loop not started yet"
              hint="The exposure-history capture loop has not begun accumulating snapshots. This is the honest pre-start state — the bridge fills in once the loop runs, not an error."
            />
          ) : (
            <>
              <Row label="Snapshots captured" value={hist.n_snapshots} />
              <Row label="First captured" value={shortTime(hist.first_captured_at)} />
              <Row label="Last captured" value={shortTime(hist.last_captured_at)} />
              <Row label="Last NAV (home $)" value={hist.last_nav != null ? `$${fmtNum(hist.last_nav, 0)}` : "—"} />
              {hist.last_net_currency && Object.keys(hist.last_net_currency).length ? (
                <div className="mt-2 border-t pt-2 hairline">
                  <SignedExposure label="Last snapshot net currency (home $)" book={hist.last_net_currency} />
                </div>
              ) : null}
            </>
          )}
          <div className="mt-2 border-t pt-2 font-mono text-[10px] text-faint hairline">
            {safeChild(hist.note)}
          </div>
        </div>
      </Card>

      {/* ---- RAW vs HEDGED ledger --------------------------------------- */}
      <Card className="flex flex-col lg:col-span-2">
        <SectionTitle right={
          <Badge color={data.n_ledger_cycles ? "#2bd17e" : "#5a6677"} dot>{data.n_ledger_cycles} LEDGER CYCLES</Badge>
        }>
          Raw vs Hedged · recent cycles (FX hedged basis is honestly &quot;unresolved&quot; — no forward-price source yet)
        </SectionTitle>
        <div className="p-3">
          {!data.recent_cycles?.length ? (
            <div className="py-3 text-center font-mono text-[12px] text-dim">
              No raw-vs-hedged ledger cycles recorded yet.
            </div>
          ) : (
            <div className="scroll-thin overflow-x-auto">
              <div className="min-w-[560px] font-mono text-[11px]">
                <div className="flex items-center gap-3 border-b pb-1.5 text-[10px] text-faint hairline">
                  <span className="w-28 shrink-0">strategy</span>
                  <span className="w-24 shrink-0">asof</span>
                  <span className="w-24 shrink-0 text-right">raw net</span>
                  <span className="w-24 shrink-0 text-right">hedge</span>
                  <span className="w-28 shrink-0 text-right">hedged basis</span>
                  <span className="flex-1 text-right">ts</span>
                </div>
                {data.recent_cycles.map((cy, i) => {
                  const rawNet = cy.raw?.net_return;
                  const basis = cy.hedged?.return_basis ?? "—";
                  const unresolved = basis === "unresolved" || basis === "gross_cost_unknown";
                  return (
                    <div key={`${cy.cycle_ts}-${i}`} className="flex items-center gap-3 border-t py-1.5 hairline first:border-t-0">
                      <span className="w-28 shrink-0 truncate text-text">{cy.strategy ?? "—"}</span>
                      <span className="w-24 shrink-0 text-faint tnum">{cy.asof_date ?? "—"}</span>
                      <span className={`w-24 shrink-0 text-right tnum ${pnlClass(rawNet)}`}>
                        {rawNet != null ? `${fmtSigned(rawNet * 100, 3)}%` : "—"}
                      </span>
                      <span className="w-24 shrink-0 truncate text-right text-faint">{cy.hedge?.status ?? "—"}</span>
                      <span className={`w-28 shrink-0 text-right ${unresolved ? "text-warn" : "text-faint"}`}>{basis}</span>
                      <span className="flex-1 text-right text-[10px] text-faint tnum">{shortTime(cy.cycle_ts)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <div className="mt-2 border-t pt-2 font-mono text-[10px] text-faint hairline">
            SHADOW / analysis-only: the hedge layer places no orders, unhalts nothing, and mutates no broker.
            {data.human_review_required ? " Human review required before anything could ever arm." : ""}
            {" "}FX hedged returns show &quot;unresolved&quot; because there is no FX forward-price source yet — surfaced, not hidden.
          </div>
        </div>
      </Card>
    </div>
  );
}
