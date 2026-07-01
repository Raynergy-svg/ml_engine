"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart, CandlestickSeries, LineSeries, HistogramSeries, ColorType, CrosshairMode, LineStyle, PriceScaleMode,
  type IChartApi, type ISeriesApi, type IPriceLine, type MouseEventParams, type UTCTimestamp,
} from "lightweight-charts";
import { usePoll } from "@/lib/api";
import { useStream } from "@/lib/stream";
import type { CandleResponse, Candle } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading } from "./ui";
import { fmtPrice, fmtPct, fmtNum, prettyPair } from "@/lib/format";
import {
  CursorIcon, TrendlineIcon, PriceZoneIcon, BrushIcon, TextToolIcon, PolygonIcon, FibIcon, MagnetIcon, RulerIcon,
  CompareIcon, IndicatorsIcon, ChevronDown, FullscreenIcon,
} from "./icons";

const GRAN_OPTIONS: { label: string; code: string }[] = [
  { label: "1m", code: "M1" }, { label: "5m", code: "M5" }, { label: "15m", code: "M15" },
  { label: "1h", code: "H1" }, { label: "4h", code: "H4" }, { label: "1D", code: "D" },
  { label: "1W", code: "W" }, { label: "1M", code: "M" },
];
const OTHER_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP", "XAU_USD"];
const POS = "#3ee287", NEG = "#ff6b5e", CYAN = "#22d3ee", WARN = "#ffc23e";

type ToolId = "cursor" | "trendline" | "zone" | "brush" | "text" | "polygon" | "fib" | "magnet" | "ruler";
const TOOLS: { id: ToolId; Icon: typeof CursorIcon; title: string; clicks: number }[] = [
  { id: "cursor", Icon: CursorIcon, title: "Cursor — select / pan", clicks: 0 },
  { id: "trendline", Icon: TrendlineIcon, title: "Trendline — click 2 points", clicks: 2 },
  { id: "zone", Icon: PriceZoneIcon, title: "Price zone — click 2 price levels", clicks: 2 },
  { id: "brush", Icon: BrushIcon, title: "Freehand — drag across the chart", clicks: 0 },
  { id: "text", Icon: TextToolIcon, title: "Text label — click to place", clicks: 1 },
  { id: "polygon", Icon: PolygonIcon, title: "Polygon — click 4 points", clicks: 4 },
  { id: "fib", Icon: FibIcon, title: "Fibonacci retracement — click 2 points", clicks: 2 },
  { id: "magnet", Icon: MagnetIcon, title: "Magnet crosshair — toggle", clicks: 0 },
  { id: "ruler", Icon: RulerIcon, title: "Measure — click 2 points", clicks: 2 },
];

const INDICATOR_DEFS = [
  { id: "sma20", label: "SMA 20", color: "#22d3ee" },
  { id: "sma50", label: "SMA 50", color: "#e84ff0" },
  { id: "ema20", label: "EMA 20", color: WARN },
  { id: "bb20", label: "Bollinger (20, 2)", color: "#8b98a9" },
] as const;

function rollingSma(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}
function rollingEma(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null);
  const k = 2 / (period + 1);
  let prev: number | null = null;
  for (let i = 0; i < values.length; i++) {
    if (i === period - 1) {
      prev = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
      out[i] = prev;
    } else if (i >= period && prev != null) {
      prev = values[i] * k + prev * (1 - k);
      out[i] = prev;
    }
  }
  return out;
}
function rollingBollinger(values: number[], period: number, mult: number) {
  const mid = rollingSma(values, period);
  const upper: (number | null)[] = new Array(values.length).fill(null);
  const lower: (number | null)[] = new Array(values.length).fill(null);
  for (let i = period - 1; i < values.length; i++) {
    const m = mid[i];
    if (m == null) continue;
    const window = values.slice(i - period + 1, i + 1);
    const variance = window.reduce((a, v) => a + (v - m) ** 2, 0) / period;
    const sd = Math.sqrt(variance);
    upper[i] = m + mult * sd;
    lower[i] = m - mult * sd;
  }
  return { upper, lower };
}

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

type Point = { time: number; price: number };
type Drawing =
  | { id: string; kind: "line"; series: ISeriesApi<"Line"> }
  | { id: string; kind: "zone"; top: number; bottom: number }
  | { id: string; kind: "text"; point: Point; text: string }
  | { id: string; kind: "pricelines"; lines: IPriceLine[] };

export function CandleChart({
  instrument, gran, onGranChange, chartType, resetZoomTick, sidePanelHidden, onToggleSidePanel,
}: {
  instrument: string;
  gran: string;
  onGranChange: (g: string) => void;
  chartType: "candle" | "line";
  resetZoomTick: number;
  sidePanelHidden: boolean;
  onToggleSidePanel: () => void;
}) {
  const { data, error, loading } = usePoll<CandleResponse>(
    `/api/candles/${instrument}?granularity=${gran}&sma=100&count=300`, 30000,
  );
  const clock = useClock();
  const [pctMode, setPctMode] = useState(false);
  const [logMode, setLogMode] = useState(false);
  const [autoScale, setAutoScale] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  const { payload } = useStream();
  const pos = payload?.account?.positions?.find((p) => p.instrument === instrument);
  const tp = pos?.take_profit ?? null;
  const sl = pos?.stop_loss ?? null;

  const elRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineRef = useRef<ISeriesApi<"Line"> | null>(null);
  const smaRef = useRef<ISeriesApi<"Line"> | null>(null);
  const volRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const bracketLinesRef = useRef<IPriceLine[]>([]);
  const activeSeries = () => (chartType === "line" ? lineRef.current : candleRef.current);

  // ---- drawing tools -------------------------------------------------------
  const [tool, setTool] = useState<ToolId>("cursor");
  const [magnetOn, setMagnetOn] = useState(false);
  const [pending, setPending] = useState<Point[]>([]);
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [overlayVersion, setOverlayVersion] = useState(0);
  const dragging = useRef(false);
  const dragPoints = useRef<Point[]>([]);

  function clearDrawings() {
    for (const d of drawings) {
      if (d.kind === "line") chartRef.current?.removeSeries(d.series);
      if (d.kind === "pricelines") for (const l of d.lines) candleRef.current?.removePriceLine(l);
    }
    setDrawings([]);
    setPending([]);
  }

  // reset drawings + pending tool state on instrument/granularity change
  useEffect(() => { clearDrawings(); setTool("cursor"); }, [instrument, gran]); // eslint-disable-line react-hooks/exhaustive-deps

  function commit(kind: ToolId, points: Point[]) {
    const chart = chartRef.current, series = candleRef.current;
    if (!chart || !series) return;
    const id = `${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    if (kind === "trendline" || kind === "ruler") {
      const line = chart.addSeries(LineSeries, { color: kind === "ruler" ? WARN : CYAN, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
      line.setData(points.map((p) => ({ time: p.time as UTCTimestamp, value: p.price })));
      setDrawings((d) => [...d, { id, kind: "line", series: line }]);
      if (kind === "ruler") {
        const [a, b] = points;
        const dPrice = b.price - a.price;
        const dSec = Math.abs(b.time - a.time);
        const dTime = dSec < 3600 ? `${Math.round(dSec / 60)}m` : dSec < 86400 ? `${(dSec / 3600).toFixed(1)}h` : `${(dSec / 86400).toFixed(1)}d`;
        setDrawings((d) => [...d, {
          id: id + "-label", kind: "text",
          point: { time: (a.time + b.time) / 2, price: (a.price + b.price) / 2 },
          text: `${dPrice >= 0 ? "+" : ""}${fmtPrice(dPrice, instrument)}  ·  ${dTime}`,
        }]);
      }
    } else if (kind === "zone") {
      const [a, b] = points;
      setDrawings((d) => [...d, { id, kind: "zone", top: Math.max(a.price, b.price), bottom: Math.min(a.price, b.price) }]);
    } else if (kind === "text") {
      const label = window.prompt("Label text:");
      if (!label) return;
      setDrawings((d) => [...d, { id, kind: "text", point: points[0], text: label }]);
    } else if (kind === "polygon") {
      const closed = [...points, points[0]];
      const line = chart.addSeries(LineSeries, { color: "#e84ff0", lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
      line.setData(closed.map((p) => ({ time: p.time as UTCTimestamp, value: p.price })));
      setDrawings((d) => [...d, { id, kind: "line", series: line }]);
    } else if (kind === "fib") {
      const [a, b] = points;
      const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
      const lines = levels.map((l) => series.createPriceLine({
        price: a.price + (b.price - a.price) * l, color: WARN, lineWidth: 1,
        lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: `${(l * 100).toFixed(1)}%`,
      }));
      setDrawings((d) => [...d, { id, kind: "pricelines", lines }]);
    }
  }

  function handleToolClick(param: MouseEventParams) {
    if (tool === "cursor" || tool === "brush" || !param.point || param.time == null) return;
    const series = candleRef.current;
    const price = series?.coordinateToPrice(param.point.y);
    if (price == null) return;
    if (tool === "magnet") return; // handled on toolbar click directly
    const point: Point = { time: param.time as number, price };
    const needed = TOOLS.find((t) => t.id === tool)?.clicks ?? 0;
    setPending((prev) => {
      const next = [...prev, point];
      if (next.length >= needed) {
        commit(tool, next);
        return [];
      }
      return next;
    });
  }

  function selectTool(id: ToolId) {
    if (id === "magnet") {
      setMagnetOn((v) => {
        const next = !v;
        chartRef.current?.applyOptions({ crosshair: { mode: next ? CrosshairMode.Magnet : CrosshairMode.Normal } });
        return next;
      });
      return;
    }
    setPending([]);
    setTool((t) => (t === id ? "cursor" : id));
  }

  // ---- indicators -----------------------------------------------------------
  const [indicatorsOpen, setIndicatorsOpen] = useState(false);
  const [activeIndicators, setActiveIndicators] = useState<Set<string>>(new Set());
  const indicatorSeriesRef = useRef<Record<string, ISeriesApi<"Line">[]>>({});

  // ---- compare overlay --------------------------------------------------------
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareInstrument, setCompareInstrument] = useState<string | null>(null);
  const [compareCandles, setCompareCandles] = useState<Candle[] | null>(null);
  const compareSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!compareInstrument) { setCompareCandles(null); return; }
    let cancelled = false;
    const load = () => {
      fetch(`/api/candles/${compareInstrument}?granularity=${gran}&count=300`)
        .then((r) => r.json())
        .then((j: CandleResponse) => { if (!cancelled) setCompareCandles(j.candles ?? []); })
        .catch(() => { if (!cancelled) setCompareCandles(null); });
    };
    load();
    const id = setInterval(load, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, [compareInstrument, gran]);

  // Create the chart once.
  useEffect(() => {
    if (!elRef.current) return;
    const chart = createChart(elRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#8b98a9", fontFamily: "var(--font-mono)" },
      grid: { vertLines: { color: "rgba(30,39,51,0.5)" }, horzLines: { color: "rgba(30,39,51,0.5)" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#1e2733", scaleMargins: { top: 0.08, bottom: 0.26 } },
      timeScale: { borderColor: "#1e2733", timeVisible: gran !== "D", secondsVisible: false },
      autoSize: true,
    });
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: POS, downColor: NEG, borderVisible: false, wickUpColor: POS, wickDownColor: NEG,
    });
    const line = chart.addSeries(LineSeries, {
      color: CYAN, lineWidth: 2, priceLineVisible: true, lastValueVisible: true, visible: false,
    });
    const sma = chart.addSeries(LineSeries, {
      color: "#22d3ee", lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false,
    });
    const vol = chart.addSeries(HistogramSeries, { priceScaleId: "vol", priceFormat: { type: "volume" } });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    chartRef.current = chart;
    candleRef.current = candle;
    lineRef.current = line;
    smaRef.current = sma;
    volRef.current = vol;
    const clickHandler = (p: MouseEventParams) => handleToolClickRef.current(p);
    chart.subscribeClick(clickHandler);
    const bump = () => setOverlayVersion((v) => v + 1);
    chart.timeScale().subscribeVisibleTimeRangeChange(bump);
    return () => {
      chart.unsubscribeClick(clickHandler);
      chart.timeScale().unsubscribeVisibleTimeRangeChange(bump);
      chart.remove();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // handleToolClick reads changing state (tool/pending) — keep a ref so the one
  // chart-lifetime subscription always calls the latest closure.
  const handleToolClickRef = useRef(handleToolClick);
  handleToolClickRef.current = handleToolClick;

  // Push candle/volume/sma data; toggle candle vs line series visibility.
  useEffect(() => {
    if (!data || !candleRef.current || !smaRef.current || !volRef.current || !lineRef.current) return;
    chartRef.current?.applyOptions({ timeScale: { timeVisible: gran !== "D", secondsVisible: false } });
    if (data.candles?.length) {
      candleRef.current.setData(data.candles.map((c) => ({ time: c.time as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close })));
      lineRef.current.setData(data.candles.map((c) => ({ time: c.time as UTCTimestamp, value: c.close })));
      smaRef.current.setData(data.sma.map((s) => ({ time: s.time as UTCTimestamp, value: s.value })));
      volRef.current.setData(data.candles.map((c) => ({
        time: c.time as UTCTimestamp, value: c.volume ?? 0,
        color: c.close >= c.open ? "rgba(62,226,135,0.5)" : "rgba(255,107,94,0.5)",
      })));
      if (!collapsed) chartRef.current?.timeScale().fitContent();
    }
  }, [data, gran, collapsed]);

  useEffect(() => {
    candleRef.current?.applyOptions({ visible: chartType === "candle" });
    lineRef.current?.applyOptions({ visible: chartType === "line" });
  }, [chartType]);

  useEffect(() => {
    volRef.current?.applyOptions({ visible: !collapsed });
  }, [collapsed]);

  // TP/SL brackets for this instrument's live position.
  useEffect(() => {
    const series = candleRef.current;
    if (!series) return;
    for (const line of bracketLinesRef.current) series.removePriceLine(line);
    bracketLinesRef.current = [];
    const add = (price: number, color: string, title: string) =>
      bracketLinesRef.current.push(series.createPriceLine({ price, color, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title }));
    if (sl != null) add(sl, NEG, "SL");
    if (tp != null) add(tp, POS, "TP");
  }, [tp, sl, instrument, data]);

  useEffect(() => {
    const mode = pctMode ? PriceScaleMode.Percentage : logMode ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal;
    chartRef.current?.priceScale("right").applyOptions({ mode, autoScale });
  }, [pctMode, logMode, autoScale]);

  useEffect(() => {
    if (resetZoomTick > 0) chartRef.current?.timeScale().fitContent();
  }, [resetZoomTick]);

  // Real indicator overlays computed client-side from the fetched candles.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !data?.candles?.length) return;
    const closes = data.candles.map((c) => c.close);
    const times = data.candles.map((c) => c.time as UTCTimestamp);
    const current = indicatorSeriesRef.current;
    for (const def of INDICATOR_DEFS) {
      const want = activeIndicators.has(def.id);
      const have = current[def.id];
      if (want && !have) {
        if (def.id === "bb20") {
          const { upper, lower } = rollingBollinger(closes, 20, 2);
          const up = chart.addSeries(LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
          const lo = chart.addSeries(LineSeries, { color: def.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
          up.setData(times.map((t, i) => ({ time: t, value: upper[i] ?? NaN })).filter((p) => !Number.isNaN(p.value)));
          lo.setData(times.map((t, i) => ({ time: t, value: lower[i] ?? NaN })).filter((p) => !Number.isNaN(p.value)));
          current[def.id] = [up, lo];
        } else {
          const period = def.id === "sma20" ? 20 : def.id === "sma50" ? 50 : 20;
          const values = def.id === "ema20" ? rollingEma(closes, period) : rollingSma(closes, period);
          const s = chart.addSeries(LineSeries, { color: def.color, lineWidth: 2, priceLineVisible: false, lastValueVisible: true });
          s.setData(times.map((t, i) => ({ time: t, value: values[i] ?? NaN })).filter((p) => !Number.isNaN(p.value)));
          current[def.id] = [s];
        }
      } else if (!want && have) {
        for (const s of have) chart.removeSeries(s);
        delete current[def.id];
      }
    }
  }, [activeIndicators, data]);

  // Real compare overlay: normalize the second instrument's % change onto this
  // instrument's own price axis (same technique any "compare" chart tool uses),
  // so it's a genuine second real series, not a fabricated one.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (compareSeriesRef.current) { chart.removeSeries(compareSeriesRef.current); compareSeriesRef.current = null; }
    if (!compareCandles?.length || !data?.candles?.length) return;
    const firstCompare = compareCandles[0].close;
    const firstBase = data.candles[0].close;
    const s = chart.addSeries(LineSeries, { color: "#e84ff0", lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
    s.setData(compareCandles.map((c) => ({
      time: c.time as UTCTimestamp, value: firstBase * (1 + (c.close - firstCompare) / firstCompare),
    })));
    compareSeriesRef.current = s;
  }, [compareCandles, data]);

  const sig = data?.signal;
  const disconnected = (data && !data.connected) || !!error;

  const candles = data?.candles ?? [];
  const last = candles.length ? candles[candles.length - 1] : null;
  const prevClose = candles.length > 1 ? candles[candles.length - 2].close : null;
  const change = last && prevClose != null ? last.close - prevClose : null;
  const changePct = change != null && prevClose ? (change / prevClose) * 100 : null;
  const volWindow = candles.slice(-9);
  const volSma9 = volWindow.length && volWindow.every((c) => c.volume != null)
    ? volWindow.reduce((a, c) => a + (c.volume ?? 0), 0) / volWindow.length
    : null;

  // Recompute HTML-overlay pixel positions from real chart coordinates whenever
  // the visible range changes (pan/zoom) or a new drawing is added.
  type OverlayPixel =
    | { id: string; kind: "zone"; top: number; height: number; width: number }
    | { id: string; kind: "text"; x: number; y: number; text: string };
  const overlayPixels = useMemo((): OverlayPixel[] => {
    void overlayVersion;
    const chart = chartRef.current, series = candleRef.current;
    if (!chart || !series || !elRef.current) return [];
    const width = elRef.current.clientWidth;
    const out: OverlayPixel[] = [];
    for (const d of drawings) {
      if (d.kind === "zone") {
        const yTop = series.priceToCoordinate(d.top);
        const yBottom = series.priceToCoordinate(d.bottom);
        if (yTop == null || yBottom == null) continue;
        out.push({ id: d.id, kind: "zone", top: yTop, height: Math.max(1, yBottom - yTop), width });
      } else if (d.kind === "text") {
        const x = chart.timeScale().timeToCoordinate(d.point.time as UTCTimestamp);
        const y = series.priceToCoordinate(d.point.price);
        if (x == null || y == null) continue;
        out.push({ id: d.id, kind: "text", x, y, text: d.text });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawings, overlayVersion, data]);

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <SectionTitle
        right={
          <div className="flex items-center gap-3">
            <div className="relative">
              <button onClick={() => setCompareOpen((v) => !v)} className={`grid h-7 w-7 place-items-center rounded hover:bg-surface2 ${compareInstrument ? "text-pos" : "text-dim hover:text-text"}`} title="Compare against another instrument">
                <CompareIcon />
              </button>
              {compareOpen && (
                <div className="card absolute right-0 top-8 z-20 max-h-64 min-w-[160px] overflow-y-auto p-1.5">
                  <button onClick={() => { setCompareInstrument(null); setCompareOpen(false); }} className={`block w-full rounded px-2.5 py-1.5 text-left font-mono text-[12px] ${!compareInstrument ? "text-pos" : "text-dim hover:bg-surface2 hover:text-text"}`}>None</button>
                  {OTHER_INSTRUMENTS.filter((i) => i !== instrument).map((i) => (
                    <button key={i} onClick={() => { setCompareInstrument(i); setCompareOpen(false); }} className={`block w-full rounded px-2.5 py-1.5 text-left font-mono text-[12px] ${compareInstrument === i ? "text-pos" : "text-dim hover:bg-surface2 hover:text-text"}`}>
                      {prettyPair(i)}
                    </button>
                  ))}
                </div>
              )}
              {compareOpen && <div className="fixed inset-0 z-10" onClick={() => setCompareOpen(false)} />}
            </div>
            <div className="relative">
              <button onClick={() => setIndicatorsOpen((v) => !v)} className="flex items-center gap-1.5 rounded border px-2 py-1 font-mono text-[11px] text-dim hairline hover:text-text">
                <IndicatorsIcon />
                Indicators
              </button>
              {indicatorsOpen && (
                <div className="card absolute right-0 top-8 z-20 min-w-[190px] p-1.5">
                  {INDICATOR_DEFS.map((def) => {
                    const on = activeIndicators.has(def.id);
                    return (
                      <button
                        key={def.id}
                        onClick={() => setActiveIndicators((prev) => {
                          const next = new Set(prev);
                          if (next.has(def.id)) next.delete(def.id); else next.add(def.id);
                          return next;
                        })}
                        className={`flex w-full items-center justify-between rounded px-2.5 py-1.5 text-left font-mono text-[12px] ${on ? "text-text" : "text-dim hover:bg-surface2 hover:text-text"}`}
                      >
                        <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: def.color }} />{def.label}</span>
                        {on && <span className="text-pos">✓</span>}
                      </button>
                    );
                  })}
                </div>
              )}
              {indicatorsOpen && <div className="fixed inset-0 z-10" onClick={() => setIndicatorsOpen(false)} />}
            </div>
            <button onClick={() => setCollapsed((v) => !v)} className="text-dim hover:text-text" title={collapsed ? "Expand volume pane" : "Collapse volume pane"}>
              <ChevronDown className={collapsed ? "" : "rotate-180"} />
            </button>
            <button onClick={onToggleSidePanel} className={`grid h-6 w-6 place-items-center ${sidePanelHidden ? "text-pos" : "text-dim hover:text-text"}`} title={sidePanelHidden ? "Restore side panel" : "Expand chart to full width"}>
              <FullscreenIcon />
            </button>
          </div>
        }
      >
        <span className="tnum">
          {prettyPair(instrument)} · {GRAN_OPTIONS.find((g) => g.code === gran)?.label ?? gran} · AXIOM
          {last && (
            <span className="ml-3 font-mono text-[11px] text-dim">
              O <span style={{ color: change != null && change >= 0 ? "var(--color-pos)" : "var(--color-neg)" }}>{fmtPrice(last.open, instrument)}</span>{" "}
              H <span style={{ color: change != null && change >= 0 ? "var(--color-pos)" : "var(--color-neg)" }}>{fmtPrice(last.high, instrument)}</span>{" "}
              L <span style={{ color: change != null && change >= 0 ? "var(--color-pos)" : "var(--color-neg)" }}>{fmtPrice(last.low, instrument)}</span>{" "}
              C <span style={{ color: change != null && change >= 0 ? "var(--color-pos)" : "var(--color-neg)" }}>{fmtPrice(last.close, instrument)}</span>
              {change != null && (
                <span className="ml-2" style={{ color: change >= 0 ? "var(--color-pos)" : "var(--color-neg)" }}>
                  {change >= 0 ? "+" : ""}{fmtPrice(change, instrument)} ({changePct != null ? fmtPct(changePct, 2) : "—"})
                </span>
              )}
            </span>
          )}
        </span>
      </SectionTitle>

      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden sm:grid-cols-[44px_minmax(0,1fr)]">
        <div className="hidden border-r bg-base/35 py-2 hairline sm:flex sm:flex-col sm:items-center sm:gap-1">
          {TOOLS.map(({ id, Icon, title }) => {
            const isActive = id === "magnet" ? magnetOn : tool === id || (id === "cursor" && tool === "cursor");
            return (
              <button
                key={id}
                onClick={() => selectTool(id)}
                title={title}
                className={`grid h-7 w-7 shrink-0 place-items-center rounded-md ${isActive ? "bg-surface2 text-pos" : "text-dim hover:bg-surface2 hover:text-text"}`}
              >
                <Icon />
              </button>
            );
          })}
        </div>

        <div className="relative min-h-0 overflow-hidden px-2 pb-2">
          <div className="px-2 py-1 font-mono text-[11px] text-dim tnum">
            Volume SMA 9 <span className="text-pos">{volSma9 != null ? fmtNum(volSma9, 0) : "—"}</span>
          </div>
          <div
            ref={elRef}
            className="relative h-[calc(100%-70px)] w-full"
            style={{ minHeight: 260, cursor: tool !== "cursor" && tool !== "magnet" ? "crosshair" : undefined }}
            onContextMenu={(e) => { e.preventDefault(); if (drawings.length) clearDrawings(); }}
            onMouseDown={(e) => {
              if (tool !== "brush" || !chartRef.current || !candleRef.current || !elRef.current) return;
              dragging.current = true;
              dragPoints.current = [];
              const rect = elRef.current.getBoundingClientRect();
              const t = chartRef.current.timeScale().coordinateToTime(e.clientX - rect.left);
              const p = candleRef.current.coordinateToPrice(e.clientY - rect.top);
              if (t != null && p != null) dragPoints.current.push({ time: t as number, price: p });
            }}
            onMouseMove={(e) => {
              if (!dragging.current || !chartRef.current || !candleRef.current || !elRef.current) return;
              const rect = elRef.current.getBoundingClientRect();
              const t = chartRef.current.timeScale().coordinateToTime(e.clientX - rect.left);
              const p = candleRef.current.coordinateToPrice(e.clientY - rect.top);
              if (t != null && p != null) dragPoints.current.push({ time: t as number, price: p });
            }}
            onMouseUp={() => {
              if (!dragging.current) return;
              dragging.current = false;
              const pts = dragPoints.current;
              if (pts.length > 1 && chartRef.current) {
                const line = chartRef.current.addSeries(LineSeries, { color: POS, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
                line.setData(pts.map((p) => ({ time: p.time as UTCTimestamp, value: p.price })));
                setDrawings((d) => [...d, { id: `brush-${Date.now()}`, kind: "line", series: line }]);
              }
              dragPoints.current = [];
            }}
          >
            <div className="pointer-events-none absolute inset-0 z-10">
              {overlayPixels.map((o) => o.kind === "zone" ? (
                <div key={o.id} className="absolute border-y border-warn/40 bg-warn/10" style={{ top: o.top, height: o.height, left: 0, width: o.width }} />
              ) : (
                <div key={o.id} className="absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded border border-warn/50 bg-base/85 px-1.5 py-0.5 font-mono text-[10px] text-warn" style={{ left: o.x, top: o.y }}>
                  {o.text}
                </div>
              ))}
            </div>
          </div>
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
                onClick={() => onGranChange(code)}
                className={`rounded px-1.5 py-0.5 ${gran === code ? "bg-pos/10 text-pos" : "hover:bg-surface2 hover:text-text"}`}
              >
                {label}
              </button>
            ))}
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
