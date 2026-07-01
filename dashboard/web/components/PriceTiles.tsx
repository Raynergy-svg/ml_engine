"use client";
import { useEffect, useRef, useState } from "react";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { Strategy } from "@/lib/types";
import { fmtPrice, fmtNum, fmtPct, prettyPair } from "@/lib/format";

// The trend lane's real FX-major universe (mirrors FX_MAJORS in data_sources.py —
// what the strategy actually evaluates, not the mockup's arbitrary 5-pair sample).
const MAJORS = [
  "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD",
  "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
];

export function PriceTiles({
  selected, onSelect,
}: { selected: string; onSelect: (i: string) => void }) {
  const { payload } = useStream();
  const { data: strat } = usePoll<Strategy>("/api/strategy", 60000);
  const prices = payload?.prices?.prices ?? {};
  const onSet = new Set(strat?.on ?? []);
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

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
      {MAJORS.map((m) => {
        const p = prices[m];
        const on = onSet.has(m);
        const isSel = selected === m;
        const dir = flash[m];
        const openRef = sessionOpen.current[m];
        const pct = p && openRef ? ((p.mid - openRef) / openRef) * 100 : null;
        const color = pct == null ? "#8b98a9" : pct >= 0 ? "#2bd17e" : "#ff4d6d";
        return (
          <button
            key={m}
            onClick={() => onSelect(m)}
            className={`card group relative min-h-[100px] overflow-hidden px-4 py-3 text-left transition-colors ${
              isSel ? "bg-cyan/10 ring-1 ring-cyan/70" : "hover:border-[var(--color-faint)] hover:bg-surface2/80"
            }`}
            style={isSel ? { borderColor: "var(--color-cyan)" } : undefined}
            title={`trend: ${on ? "LONG" : "FLAT"}${pct != null ? " · Δ since AXIOM connected" : ""}`}
          >
            <span
              className="absolute left-0 top-0 h-full w-[3px]"
              style={{ background: isSel ? "var(--color-emerald)" : "transparent" }}
            />
            <div className="flex items-center justify-between">
              <span className="font-mono text-[12px] font-semibold text-text">{prettyPair(m)}</span>
              <span className="font-mono text-[11px] tnum" style={{ color }}>
                {pct == null ? "—" : `${pct >= 0 ? "+" : ""}${fmtPct(pct, 2)}`}
              </span>
            </div>
            <div className="mt-4 grid grid-cols-[1fr_1fr_52px] gap-3">
              <div>
                <div className="font-mono text-[10px] uppercase text-faint">Bid</div>
                <div
                  className="mt-1 font-mono text-[18px] font-semibold tnum transition-colors"
                  style={{ color: dir === "up" ? "#2bd17e" : dir === "down" ? "#ff4d6d" : "#e6edf3" }}
                >
                  {p ? fmtPrice(p.bid, m) : "—"}
                </div>
              </div>
              <div>
                <div className="font-mono text-[10px] uppercase text-faint">Ask</div>
                <div
                  className="mt-1 font-mono text-[18px] font-semibold tnum transition-colors"
                  style={{ color: dir === "up" ? "#2bd17e" : dir === "down" ? "#ff4d6d" : "#e6edf3" }}
                >
                  {p ? fmtPrice(p.ask, m) : "—"}
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-[10px] uppercase text-faint">Spread</div>
                <div className="mt-1 font-mono text-[16px] text-text tnum">{p ? fmtNum(p.spread_pips, 1) : "—"}</div>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
