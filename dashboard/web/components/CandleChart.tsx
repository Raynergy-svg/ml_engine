"use client";
import { useEffect, useRef, useState } from "react";
import {
  createChart, CandlestickSeries, LineSeries, ColorType, CrosshairMode, LineStyle,
  type IChartApi, type ISeriesApi, type IPriceLine,
} from "lightweight-charts";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { CandleResponse } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge } from "./ui";
import { fmtPrice, fmtPct, prettyPair } from "@/lib/format";

const GRANS = ["D", "H4", "H1"] as const;

export function CandleChart({ instrument }: { instrument: string }) {
  const [gran, setGran] = useState<(typeof GRANS)[number]>("D");
  const { data, error, loading } = usePoll<CandleResponse>(
    `/api/candles/${instrument}?granularity=${gran}&sma=100&count=300`, 30000,
  );

  // Bracket levels for THIS instrument's open position (if any) — from the live
  // stream. Present only once the bot writes TP/SL into account_state.json.
  const { payload } = useStream();
  const pos = payload?.account?.positions?.find((p) => p.instrument === instrument);
  const tp = pos?.take_profit ?? null;
  const sl = pos?.stop_loss ?? null;

  const elRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const smaRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bracketLinesRef = useRef<IPriceLine[]>([]);

  // Create the chart once.
  useEffect(() => {
    if (!elRef.current) return;
    const chart = createChart(elRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#8b98a9", fontFamily: "var(--font-mono)" },
      grid: { vertLines: { color: "rgba(30,39,51,0.5)" }, horzLines: { color: "rgba(30,39,51,0.5)" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#1e2733" },
      timeScale: { borderColor: "#1e2733", timeVisible: gran !== "D", secondsVisible: false },
      autoSize: true,
    });
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: "#2bd17e", downColor: "#ff4d6d", borderVisible: false,
      wickUpColor: "#2bd17e", wickDownColor: "#ff4d6d",
    });
    const sma = chart.addSeries(LineSeries, {
      color: "#22d3ee", lineWidth: 2, priceLineVisible: false, lastValueVisible: true,
      crosshairMarkerVisible: false,
    });
    chartRef.current = chart;
    candleRef.current = candle;
    smaRef.current = sma;
    return () => { chart.remove(); chartRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push data into the series when it arrives / changes.
  useEffect(() => {
    if (!data || !candleRef.current || !smaRef.current) return;
    // Show intraday time on the axis for sub-daily granularities (the chart is
    // created once, so re-apply when granularity changes).
    chartRef.current?.applyOptions({ timeScale: { timeVisible: gran !== "D", secondsVisible: false } });
    if (data.candles?.length) {
      // lightweight-charts v5 requires the literal time type; unix seconds are valid.
      candleRef.current.setData(data.candles as never);
      smaRef.current.setData(data.sma as never);
      chartRef.current?.timeScale().fitContent();
    }
  }, [data, gran]);

  // Draw TP/SL bracket lines for the selected instrument's position; clear + redraw
  // when the levels change or are removed. No-op (no lines) when brackets are absent.
  useEffect(() => {
    const series = candleRef.current;
    if (!series) return;
    for (const line of bracketLinesRef.current) series.removePriceLine(line);
    bracketLinesRef.current = [];
    const add = (price: number, color: string, title: string) =>
      bracketLinesRef.current.push(series.createPriceLine({
        price, color, lineWidth: 1, lineStyle: LineStyle.Dashed,
        axisLabelVisible: true, title,
      }));
    if (sl != null) add(sl, "#ff4d6d", "SL");
    if (tp != null) add(tp, "#2bd17e", "TP");
  }, [tp, sl, instrument, data]);

  const sig = data?.signal;
  const disconnected = (data && !data.connected) || !!error;

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={
          <div className="flex items-center gap-2">
            {sig && (
              <Badge color={sig.on ? "#34e5a1" : "#8b98a9"} dot pulse={sig.on}>
                {sig.state} · {fmtPct(sig.distance_pct, 2)} vs SMA
              </Badge>
            )}
            <div className="flex overflow-hidden rounded-md border hairline">
              {GRANS.map((g) => (
                <button
                  key={g}
                  onClick={() => setGran(g)}
                  className={`px-2.5 py-1 font-mono text-[11px] ${
                    gran === g ? "bg-surface2 text-cyan" : "text-faint hover:text-dim"
                  }`}
                >
                  {g}
                </button>
              ))}
            </div>
          </div>
        }
      >
        {prettyPair(instrument)} · trend SMA(100)
      </SectionTitle>

      <div className="relative min-h-0 flex-1 px-2 pb-2">
        <div ref={elRef} className="h-full w-full" style={{ minHeight: 320 }} />
        {disconnected && (
          <div className="absolute inset-0 grid place-items-center bg-surface/60 backdrop-blur-sm">
            <NotConnected
              label="Candles not available"
              hint="OANDA practice read endpoint unreachable (stale token?). The signal overlay needs live candles."
            />
          </div>
        )}
        {loading && !data && (
          <div className="absolute inset-0 grid place-items-center"><Loading label="Loading candles…" /></div>
        )}
      </div>

      {sig && (
        <div className="flex items-center gap-5 border-t px-4 py-2 font-mono text-[11px] hairline tnum text-dim">
          <span>last <span className="text-text">{fmtPrice(sig.price, instrument)}</span></span>
          <span>SMA <span className="text-cyan">{fmtPrice(sig.sma, instrument)}</span></span>
          {sl != null && <span>SL <span style={{ color: "#ff4d6d" }}>{fmtPrice(sl, instrument)}</span></span>}
          {tp != null && <span>TP <span style={{ color: "#2bd17e" }}>{fmtPrice(tp, instrument)}</span></span>}
          {tp == null && sl == null && (
            <span className="text-faint">price &gt; SMA ⇒ long · else flat (shift-1 causal)</span>
          )}
        </div>
      )}
    </Card>
  );
}
