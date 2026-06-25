"use client";
import { useEffect, useRef, useState } from "react";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { Strategy } from "@/lib/types";
import { fmtPrice, fmtNum, prettyPair } from "@/lib/format";
import { StatusDot } from "./ui";

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

  useEffect(() => {
    const next: Record<string, "up" | "down"> = {};
    for (const m of MAJORS) {
      const p = prices[m]?.mid;
      const old = prev.current[m];
      if (p != null && old != null && p !== old) next[m] = p > old ? "up" : "down";
      if (p != null) prev.current[m] = p;
    }
    if (Object.keys(next).length) {
      setFlash(next);
      const t = setTimeout(() => setFlash({}), 600);
      return () => clearTimeout(t);
    }
  }, [prices]);

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
      {MAJORS.map((m) => {
        const p = prices[m];
        const on = onSet.has(m);
        const isSel = selected === m;
        const dir = flash[m];
        return (
          <button
            key={m}
            onClick={() => onSelect(m)}
            className={`card group relative overflow-hidden px-3 py-2.5 text-left transition-colors ${
              isSel ? "ring-1 ring-cyan/70" : "hover:border-[var(--color-faint)]"
            }`}
            style={isSel ? { borderColor: "var(--color-cyan)" } : undefined}
          >
            <span
              className="absolute left-0 top-0 h-full w-[3px]"
              style={{ background: on ? "var(--color-emerald)" : "transparent" }}
              title={on ? "trend LONG" : "trend FLAT"}
            />
            <div className="flex items-center justify-between">
              <span className="font-mono text-[12px] font-semibold text-text">{prettyPair(m)}</span>
              <StatusDot color={on ? "#34e5a1" : "#3a4150"} />
            </div>
            <div
              className="mt-1 font-mono text-[17px] font-semibold tnum transition-colors"
              style={{ color: dir === "up" ? "#2bd17e" : dir === "down" ? "#ff4d6d" : "#e6edf3" }}
            >
              {p ? fmtPrice(p.mid, m) : "—"}
            </div>
            <div className="mt-0.5 flex items-center justify-between font-mono text-[10px] text-faint tnum">
              <span>{on ? "LONG" : "FLAT"}</span>
              <span>{p ? `${fmtNum(p.spread_pips, 1)}p` : ""}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
