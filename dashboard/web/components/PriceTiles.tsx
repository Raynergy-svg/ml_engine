"use client";
import { useEffect, useRef, useState } from "react";
import { usePoll } from "@/lib/api";
import type { Strategy, Prices } from "@/lib/types";
import { fmtPrice, fmtNum, fmtPct, prettyPair } from "@/lib/format";
import { SparklineIcon, ClockIcon } from "./icons";

// Exactly the 5 instruments in target.png's watchlist. XAU_USD isn't part of the
// trend lane's FX_MAJORS universe (no position sizing/strategy runs on it here) —
// it's a real, quotable OANDA instrument fetched purely for display, same as any
// other market-data quote a terminal would show without holding a position in it.
const MAJORS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "XAU_USD"];

export function PriceTiles({
  selected, onSelect,
}: { selected: string; onSelect: (i: string) => void }) {
  const { data: priceData } = usePoll<Prices>(`/api/prices?instruments=${MAJORS.join(",")}`, 4000);
  const { data: strat } = usePoll<Strategy>("/api/strategy", 60000);
  const prices = priceData?.prices ?? {};
  const onSet = new Set(strat?.on ?? []);
  const [compact, setCompact] = useState(false);
  const prev = useRef<Record<string, number>>({});
  const [flash, setFlash] = useState<Record<string, "up" | "down">>({});
  // Session-open reference mid per instrument (first tick seen after mount) — used
  // to show a real Δ%, honestly scoped to "since connect" rather than a fabricated
  // "daily change" the backend doesn't track per-tick.
  const sessionOpen = useRef<Record<string, number>>({});

  useEffect(() => {
    const next: Record<string, "up" | "down"> = {};
    for (const m of MAJORS) {
      const p = prices[m]?.mid;
      if (p == null) continue;
      if (sessionOpen.current[m] == null) sessionOpen.current[m] = p;
      const old = prev.current[m];
      if (old != null && p !== old) next[m] = p > old ? "up" : "down";
      prev.current[m] = p;
    }
    if (Object.keys(next).length) {
      setFlash(next);
      const t = setTimeout(() => setFlash({}), 600);
      return () => clearTimeout(t);
    }
  }, [prices]);

  // Real session-overlap readout from the actual current UTC hour — standard FX
  // session hours (Sydney/Tokyo/London/New York), not a fabricated label.
  const sessionLabel = (() => {
    const h = new Date().getUTCHours();
    const sessions: string[] = [];
    if (h >= 21 || h < 6) sessions.push("Sydney");
    if (h >= 0 && h < 9) sessions.push("Tokyo");
    if (h >= 8 && h < 17) sessions.push("London");
    if (h >= 13 && h < 22) sessions.push("New York");
    return sessions.length ? sessions.join(" + ") : "Between sessions";
  })();

  return (
    <div className="flex items-stretch gap-2">
      <div className="grid flex-1 grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {MAJORS.map((m) => {
          const p = prices[m];
          const on = onSet.has(m);
          const isSel = selected === m;
          const dir = flash[m];
          const openRef = sessionOpen.current[m];
          const pct = p && openRef ? ((p.mid - openRef) / openRef) * 100 : null;
          const color = pct == null ? "#8b98a9" : pct >= 0 ? "var(--color-pos)" : "var(--color-neg)";
          const bidAskColor = dir === "up" ? "var(--color-pos)" : dir === "down" ? "var(--color-neg)" : color;
          return (
            <button
              key={m}
              onClick={() => onSelect(m)}
              className={`card group relative overflow-hidden px-4 py-3 text-left transition-colors hover:border-[var(--color-faint)] hover:bg-surface2/80 ${compact ? "min-h-[64px]" : "min-h-[100px]"}`}
              title={`${isSel ? "shown on chart below" : "click to view on chart"} · trend: ${on ? "LONG" : "FLAT"}${pct != null ? " · Δ since AXIOM connected" : ""}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-[12px] font-semibold text-text">{prettyPair(m)}</span>
                <span className="font-mono text-[11px] tnum" style={{ color }}>
                  {pct == null ? "—" : `${pct >= 0 ? "+" : ""}${fmtPct(pct, 2)}`}
                </span>
              </div>
              {!compact && (
                <div className="mt-4 grid grid-cols-[1fr_1fr_52px] gap-3">
                  <div>
                    <div className="font-mono text-[10px] uppercase text-faint">Bid</div>
                    <div className="mt-1 font-mono text-[18px] font-semibold tnum transition-colors" style={{ color: bidAskColor }}>
                      {p ? fmtPrice(p.bid, m) : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase text-faint">Ask</div>
                    <div className="mt-1 font-mono text-[18px] font-semibold tnum transition-colors" style={{ color: bidAskColor }}>
                      {p ? fmtPrice(p.ask, m) : "—"}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-[10px] uppercase text-faint">Spread</div>
                    <div className="mt-1 font-mono text-[16px] text-text tnum">{p ? fmtNum(p.spread_pips, 2) : "—"}</div>
                  </div>
                </div>
              )}
              {compact && (
                <div className="mt-2 font-mono text-[16px] font-semibold tnum" style={{ color: bidAskColor }}>
                  {p ? fmtPrice(p.mid, m) : "—"}
                </div>
              )}
            </button>
          );
        })}
      </div>
      <div className="card flex w-11 shrink-0 flex-col divide-y hairline">
        <button
          onClick={() => setCompact((v) => !v)}
          className={`grid flex-1 place-items-center hover:bg-surface2 ${compact ? "text-pos" : "text-faint hover:text-text"}`}
          title="Toggle compact tile density (real layout switch)"
        >
          <SparklineIcon />
        </button>
        <button className="grid flex-1 place-items-center text-faint hover:bg-surface2 hover:text-text" title={`Session: ${sessionLabel} (UTC ${new Date().getUTCHours()}:00)`}>
          <ClockIcon />
        </button>
      </div>
    </div>
  );
}
