"use client";
import { useEffect, useRef, useState } from "react";
import {
  createChart, CandlestickSeries, LineSeries, ColorType, CrosshairMode, LineStyle, PriceScaleMode,
  type IChartApi, type ISeriesApi, type IPriceLine,
} from "lightweight-charts";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { CandleResponse } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading } from "./ui";
import { fmtPrice, fmtPct, fmtNum, prettyPair } from "@/lib/format";

// Granularity buttons shown -> the real OANDA code the backend candle endpoint takes.
const GRAN_OPTIONS: { label: string; code: string }[] = [
  { label: "1m", code: "M1" }, { label: "5m", code: "M5" }, { label: "15m", code: "M15" },
  { label: "1h", code: "H1" }, { label: "4h", code: "H4" }, { label: "1D", code: "D" },
  { label: "1W", code: "W" }, { label: "1M", code: "M" },
];
const TOOL_LABELS = ["↖", "/", "□", "~", "T", "⌁", "≡", "✦", "⌁"];

function useClock(): string {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  if (!now) return "";
  return now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function CandleChart({ instrument }: { instrument: string }) {
  const [gran, setGran] = useState("H1");
  const { data, error, loading } = usePoll<CandleResponse>(
    `/api/candles/${instrument}?granularity=${gran}&sma=100&count=300`, 30000,
  );
  const clock = useClock();
  // Real, functional chart-scale toggles (lightweight-charts price-scale options) —
  // not decorative text. "auto" defaults on (matches lightweight-charts default).
  const [pctMode, setPctMode] = useState(false);
  const [logMode, setLogMode] = useState(false);
  const [autoScale, setAutoScale] = useState(true);

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

  // Real price-scale mode + autoscale — applied via lightweight-charts' own API,
  // not a fabricated toggle. Percentage mode takes precedence over log if both set
  // (matches lightweight-charts' own PriceScaleMode enum, which is mutually exclusive).
  useEffect(() => {
    const mode = pctMode ? PriceScaleMode.Percentage : logMode ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal;
    chartRef.current?.priceScale("right").applyOptions({ mode, autoScale });
  }, [pctMode, logMode, autoScale]);

  const sig = data?.signal;
  const disconnected = (data && !data.connected) || !!error;

  // Real OHLC readout from the latest closed bar + real volume sum, when reported.
  const candles = data?.candles ?? [];
  const last = candles.length ? candles[candles.length - 1] : null;
  const prevClose = candles.length > 1 ? candles[candles.length - 2].close : null;
  const change = last && prevClose != null ? last.close - prevClose : null;
  const changePct = change != null && prevClose ? (change / prevClose) * 100 : null;
  const volWindow = candles.slice(-9);
  const volSum = volWindow.length && volWindow.every((c) => c.volume != null)
    ? volWindow.reduce((a, c) => a + (c.volume ?? 0), 0)
    : null;

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <SectionTitle
        right={
          <button className="hidden rounded border px-2 py-1 font-mono text-[11px] text-dim hairline md:inline-flex">
            Indicators
          </button>
        }
      >
        {prettyPair(instrument)} · {GRAN_OPTIONS.find((g) => g.code === gran)?.label ?? gran} · AXIOM
      </SectionTitle>

      <div className="grid min-h-0 flex-1 grid-cols-1 sm:grid-cols-[44px_minmax(0,1fr)]">
        <div className="hidden border-r bg-base/35 py-2 hairline sm:flex sm:flex-col sm:items-center sm:gap-1.5">
          {TOOL_LABELS.map((label, i) => (
            <button
              key={`${label}-${i}`}
              className="grid h-8 w-8 place-items-center rounded-md font-mono text-[13px] text-dim hover:bg-surface2 hover:text-text"
              title="Chart tool (drawing tools not yet wired)"
            >
              {label}
            </button>
          ))}
        </div>

        <div className="relative min-h-0 overflow-hidden px-2 pb-2">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-2 py-2 font-mono text-[11px] text-dim tnum">
            {last ? (
              <>
                <span>O <span style={{ color: change != null && change >= 0 ? "#2bd17e" : "#ff4d6d" }}>{fmtPrice(last.open, instrument)}</span></span>
                <span>H <span style={{ color: change != null && change >= 0 ? "#2bd17e" : "#ff4d6d" }}>{fmtPrice(last.high, instrument)}</span></span>
                <span>L <span style={{ color: change != null && change >= 0 ? "#2bd17e" : "#ff4d6d" }}>{fmtPrice(last.low, instrument)}</span></span>
                <span>C <span style={{ color: change != null && change >= 0 ? "#2bd17e" : "#ff4d6d" }}>{fmtPrice(last.close, instrument)}</span></span>
                {change != null && (
                  <span style={{ color: change >= 0 ? "#2bd17e" : "#ff4d6d" }}>
                    {change >= 0 ? "+" : ""}{fmtPrice(change, instrument)} ({changePct != null ? fmtPct(changePct, 2) : "—"})
                  </span>
                )}
                {volSum != null && (
                  <span className="w-full text-dim">Volume Σ9 <span className="text-pos">{fmtNum(volSum, 0)}</span></span>
                )}
              </>
            ) : (
              <span className="text-faint">awaiting candles…</span>
            )}
          </div>
          <div ref={elRef} className="h-[calc(100%-74px)] w-full" style={{ minHeight: 280 }} />
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
          <div className="scroll-thin flex items-center gap-4 overflow-x-auto border-t px-2 py-2 font-mono text-[11px] text-dim hairline">
            {GRAN_OPTIONS.map(({ label, code }) => (
              <button
                key={code}
                onClick={() => setGran(code)}
                className={`rounded px-1.5 py-0.5 ${gran === code ? "bg-pos/10 text-pos" : "hover:bg-surface2 hover:text-text"}`}
              >
                {label}
              </button>
            ))}
            <button
              className="ml-auto grid h-6 w-6 place-items-center rounded text-dim hover:bg-surface2 hover:text-text"
              title="Chart settings (not yet wired)"
            >
              ⚙
            </button>
          </div>
        </div>
      </div>

      <div className="scroll-thin flex items-center gap-4 overflow-x-auto border-t px-4 py-2 font-mono text-[11px] hairline tnum text-dim">
        <span>{clock} (local)</span>
        {sig && <span className="text-faint">{sig.state} · {fmtPct(sig.distance_pct, 2)} vs SMA</span>}
        <span className="ml-auto flex items-center gap-3">
          <button onClick={() => setPctMode((v) => !v)} className={pctMode ? "text-pos" : "hover:text-text"}>%</button>
          <button onClick={() => setLogMode((v) => !v)} className={logMode ? "text-pos" : "hover:text-text"}>log</button>
          <button onClick={() => setAutoScale((v) => !v)} className={autoScale ? "text-pos" : "hover:text-text"}>auto</button>
        </span>
      </div>
    </Card>
  );
}
