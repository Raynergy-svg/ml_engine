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
import { ChevronDown } from "./icons";
import { fmtMoney, fmtSigned, fmtPct, pnlClass } from "@/lib/format";

const RANGES = [
  { label: "1D", days: 1 }, { label: "1W", days: 7 }, { label: "1M", days: 30 },
  { label: "3M", days: 90 }, { label: "6M", days: 182 }, { label: "YTD", days: null },
  { label: "1Y", days: 365 }, { label: "ALL", days: Infinity },
];

type EquityMode = "unrealized" | "realized";

export function EquityCurve() {
  const { data, loading } = usePoll<Equity>("/api/equity", 20000);
  const { payload } = useStream();
  const [range, setRange] = useState("1W");
  const [mode, setMode] = useState<EquityMode>("unrealized");
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
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
      lineColor: "#3ee287", lineWidth: 2,
      topColor: "rgba(62,226,135,0.30)", bottomColor: "rgba(62,226,135,0.02)",
      priceLineVisible: false,
    });
    chartRef.current = chart;
    seriesRef.current = series;
    return () => { chart.remove(); chartRef.current = null; };
  }, []);

  // Real client-side range filter over the real ledger points — no fabricated data,
  // just a different honest window into the same real balance history. This series
  // IS the realized curve: `balance` only moves on realized fills/financing, never
  // on the floating P&L of a still-open trade.
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

  const liveNav = payload?.account?.nav ?? null;
  // "Unrealized" appends ONE real, live point (now, current NAV = balance + open
  // trades' floating P&L). We don't have a historical floating-P&L series (that
  // would require replaying tick history against each past open position — not
  // attempted here), so we never fabricate a full unrealized curve; the honest
  // scope is: realized history is exact, and the live edge reflects the real,
  // currently-open floating P&L via the streaming NAV.
  const chartSeries = useMemo(() => {
    if (mode !== "unrealized" || liveNav == null || !filteredSeries.length) return filteredSeries;
    const nowSec = Math.floor(Date.now() / 1000);
    const lastPoint = filteredSeries[filteredSeries.length - 1];
    if (nowSec <= lastPoint.time) return filteredSeries;
    return [...filteredSeries, { time: nowSec, value: liveNav }];
  }, [filteredSeries, mode, liveNav]);

  useEffect(() => {
    if (!chartSeries.length || !seriesRef.current) return;
    seriesRef.current.setData(chartSeries as never);
    chartRef.current?.timeScale().fitContent();
  }, [chartSeries]);

  const empty = data && !data.points?.length;
  const lastBalance = data?.points?.length ? data.points[data.points.length - 1].balance : null;
  const currentEquity = mode === "unrealized" ? (liveNav ?? lastBalance) : lastBalance;
  // Real day% derived from the real ledger: first balance point at/after UTC midnight
  // vs the mode's current equity figure — not a second network call, just the data
  // already here.
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
      <SectionTitle
        right={
          <div className="relative">
            <button
              onClick={() => setModeMenuOpen((v) => !v)}
              className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-[12px] text-dim hairline hover:text-text"
            >
              {mode === "unrealized" ? "Unrealized (Live NAV)" : "Realized (Ledger)"}
              <ChevronDown />
            </button>
            {modeMenuOpen && (
              <div className="card absolute right-0 top-6 z-20 min-w-[210px] p-1.5">
                <button
                  onClick={() => { setMode("unrealized"); setModeMenuOpen(false); }}
                  className={`flex w-full items-center justify-between rounded px-2.5 py-1.5 text-left font-mono text-[12px] ${mode === "unrealized" ? "text-pos" : "text-dim hover:bg-surface2 hover:text-text"}`}
                >
                  <span>Unrealized (Live NAV)</span>
                  <span className="text-faint" title="Balance + floating P&L of open trades">incl. open</span>
                </button>
                <button
                  onClick={() => { setMode("realized"); setModeMenuOpen(false); }}
                  className={`flex w-full items-center justify-between rounded px-2.5 py-1.5 text-left font-mono text-[12px] ${mode === "realized" ? "text-pos" : "text-dim hover:bg-surface2 hover:text-text"}`}
                >
                  <span>Realized (Ledger)</span>
                  <span className="text-faint" title="Account balance only — closed trades + financing">closed only</span>
                </button>
              </div>
            )}
            {modeMenuOpen && <div className="fixed inset-0 z-10" onClick={() => setModeMenuOpen(false)} />}
          </div>
        }
      >
        Equity Curve
      </SectionTitle>

      <div
        className="flex items-baseline gap-3 px-4 pt-2"
        title={data ? `${fmtSigned(data.ledger_realized_pl)} this ledger window (not since-inception)` : undefined}
      >
        <span className={`font-mono text-[22px] font-semibold tnum ${pnlClass(dayPct)}`}>
          {currentEquity != null ? fmtMoney(currentEquity) : "—"}
        </span>
        {dayPct != null && (
          <span className={`font-mono text-[12px] tnum ${pnlClass(dayPct)}`}>
            {dayPct >= 0 ? "+" : ""}{fmtPct(dayPct, 2)} (Day)
          </span>
        )}
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden px-2 pb-1 pt-1">
        <div ref={elRef} className="h-full w-full" style={{ minHeight: 100 }} />
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
