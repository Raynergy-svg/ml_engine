"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart, AreaSeries, ColorType, CrosshairMode,
  type IChartApi, type ISeriesApi,
} from "lightweight-charts";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { Equity } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading } from "./ui";
import { fmtMoney, fmtSigned, fmtPct, pnlClass } from "@/lib/format";

const RANGES = [
  { label: "1D", days: 1 }, { label: "1W", days: 7 }, { label: "1M", days: 30 },
  { label: "3M", days: 90 }, { label: "6M", days: 182 }, { label: "YTD", days: null },
  { label: "1Y", days: 365 }, { label: "ALL", days: Infinity },
];

export function EquityCurve() {
  const { data, loading } = usePoll<Equity>("/api/equity", 20000);
  const { payload } = useStream();
  const [range, setRange] = useState("1W");
  const elRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!elRef.current) return;
    const chart = createChart(elRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#8b98a9", fontFamily: "var(--font-mono)" },
      grid: { vertLines: { color: "rgba(30,39,51,0.4)" }, horzLines: { color: "rgba(30,39,51,0.4)" } },
      crosshair: { mode: CrosshairMode.Magnet },
      rightPriceScale: { borderColor: "#1e2733" },
      timeScale: { borderColor: "#1e2733", timeVisible: true, secondsVisible: false },
      autoSize: true,
    });
    const series = chart.addSeries(AreaSeries, {
      lineColor: "#2bd17e", lineWidth: 2,
      topColor: "rgba(43,209,126,0.30)", bottomColor: "rgba(43,209,126,0.02)",
      priceLineVisible: false,
    });
    chartRef.current = chart;
    seriesRef.current = series;
    return () => { chart.remove(); chartRef.current = null; };
  }, []);

  // Real client-side range filter over the real ledger points — no fabricated data,
  // just a different honest window into the same real balance history.
  const filteredSeries = useMemo(() => {
    if (!data?.points?.length) return [];
    const byT = new Map<number, number>();
    for (const p of data.points) {
      const t = Math.floor(new Date(p.time).getTime() / 1000);
      if (!Number.isNaN(t)) byT.set(t, p.balance);
    }
    let entries = [...byT.entries()].sort((a, b) => a[0] - b[0]);
    const spec = RANGES.find((r) => r.label === range);
    if (spec && spec.days !== Infinity) {
      const now = Date.now() / 1000;
      let cutoff: number;
      if (spec.days === null) { // YTD
        const jan1 = new Date(new Date().getFullYear(), 0, 1).getTime() / 1000;
        cutoff = jan1;
      } else {
        cutoff = now - spec.days * 86400;
      }
      entries = entries.filter(([t]) => t >= cutoff);
    }
    return entries.map(([time, value]) => ({ time, value }));
  }, [data, range]);

  useEffect(() => {
    if (!filteredSeries.length || !seriesRef.current) return;
    seriesRef.current.setData(filteredSeries as never);
    chartRef.current?.timeScale().fitContent();
  }, [filteredSeries]);

  const empty = data && !data.points?.length;
  const currentEquity = payload?.account?.nav ?? (data?.points?.length ? data.points[data.points.length - 1].balance : null);
  // Real day% derived from the real ledger: first balance point at/after UTC midnight
  // vs the current live NAV — not a second network call, just the data already here.
  const dayOpenBalance = useMemo(() => {
    if (!data?.points?.length) return null;
    const startOfDay = new Date(); startOfDay.setUTCHours(0, 0, 0, 0);
    const cutoff = startOfDay.getTime();
    const todays = data.points.filter((p) => new Date(p.time).getTime() >= cutoff);
    return todays.length ? todays[0].balance : null;
  }, [data]);
  const dayPct = currentEquity != null && dayOpenBalance ? ((currentEquity - dayOpenBalance) / dayOpenBalance) * 100 : null;

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle right={<span className="font-mono text-[12px] text-faint">Realized P&L ⌄</span>}>
        Equity Curve
      </SectionTitle>

      <div className="flex items-baseline gap-3 px-4 pt-2">
        <span className="font-mono text-[22px] font-semibold tnum text-text">
          {currentEquity != null ? fmtMoney(currentEquity) : "—"}
        </span>
        {dayPct != null && (
          <span className={`font-mono text-[12px] tnum ${pnlClass(dayPct)}`}>
            {dayPct >= 0 ? "+" : ""}{fmtPct(dayPct, 2)} (Day)
          </span>
        )}
        {data && (
          <span
            className={`ml-auto font-mono text-[11px] tnum ${pnlClass(data.ledger_realized_pl)}`}
            title="P&L summed over this captured ledger window — not since-inception"
          >
            {fmtSigned(data.ledger_realized_pl)} this ledger
          </span>
        )}
      </div>

      <div className="relative min-h-0 flex-1 px-2 pb-1 pt-1">
        <div ref={elRef} className="h-full w-full" style={{ minHeight: 180 }} />
        {empty && <div className="absolute inset-0 grid place-items-center"><NotConnected label="No balance history yet" /></div>}
        {loading && !data && <div className="absolute inset-0 grid place-items-center"><Loading /></div>}
      </div>

      <div className="flex items-center gap-1 border-t px-3 py-2 font-mono text-[11px] hairline">
        {RANGES.map((r) => (
          <button
            key={r.label}
            onClick={() => setRange(r.label)}
            className={`rounded px-2 py-1 ${range === r.label ? "bg-pos/10 text-pos" : "text-faint hover:bg-surface2 hover:text-text"}`}
          >
            {r.label}
          </button>
        ))}
      </div>
    </Card>
  );
}
