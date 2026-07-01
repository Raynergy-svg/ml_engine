"use client";
import { useEffect, useRef } from "react";
import {
  createChart, AreaSeries, ColorType, CrosshairMode,
  type IChartApi, type ISeriesApi,
} from "lightweight-charts";
import { usePoll } from "@/lib/api";
import type { Equity } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading } from "./ui";
import { fmtSigned, pnlClass } from "@/lib/format";

export function EquityCurve() {
  const { data, loading } = usePoll<Equity>("/api/equity", 20000);
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

  useEffect(() => {
    if (!data?.points?.length || !seriesRef.current) return;
    // Collapse duplicate timestamps (multiple fills per second) — keep the last
    // balance for each second so lightweight-charts gets strictly ascending times.
    const byT = new Map<number, number>();
    for (const p of data.points) {
      const t = Math.floor(new Date(p.time).getTime() / 1000);
      if (!Number.isNaN(t)) byT.set(t, p.balance);
    }
    const series = [...byT.entries()].sort((a, b) => a[0] - b[0]).map(([time, value]) => ({ time, value }));
    seriesRef.current.setData(series as never);
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  const empty = data && !data.points?.length;

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={
          data ? (
            <span
              className={`font-mono text-[12px] font-semibold tnum ${pnlClass(data.ledger_realized_pl)}`}
              title="P&L summed over this captured ledger window — not since-inception (see header Total Realized)"
            >
              {fmtSigned(data.ledger_realized_pl)} this ledger
            </span>
          ) : undefined
        }
      >
        Equity Curve
      </SectionTitle>
      <div className="relative min-h-0 flex-1 px-2 pb-2">
        <div ref={elRef} className="h-full w-full" style={{ minHeight: 220 }} />
        {empty && <div className="absolute inset-0 grid place-items-center"><NotConnected label="No balance history yet" /></div>}
        {loading && !data && <div className="absolute inset-0 grid place-items-center"><Loading /></div>}
      </div>
    </Card>
  );
}
