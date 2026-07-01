"use client";
import { useEffect, useMemo, useState } from "react";
import { StreamProvider, useStream } from "@/lib/stream";
import { control } from "@/lib/control";
import { AccountHeader } from "@/components/AccountHeader";
import { GlobalStatusStrip } from "@/components/GlobalStatusStrip";
import { PriceTiles } from "@/components/PriceTiles";
import { CandleChart } from "@/components/CandleChart";
import { StrategyPanel } from "@/components/StrategyPanel";
import { PositionsTable } from "@/components/PositionsTable";
import { EquityCurve } from "@/components/EquityCurve";
import { TradeHistory } from "@/components/TradeHistory";
import { SentimentPlaceholder } from "@/components/SentimentPlaceholder";
import { Tier7Panel } from "@/components/Tier7Panel";
import { Tier7Cockpit } from "@/components/Tier7Cockpit";
import { ControlPanel } from "@/components/ControlPanel";
import { CommandPalette, type CommandItem } from "@/components/CommandPalette";

const TABS = ["Overview", "Markets", "Positions", "Orders", "Executions", "Risk", "Automation", "Journal", "Analytics", "Settings"] as const;
type Tab = (typeof TABS)[number];
const MAJORS = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP"];

function useStoredValue(key: string, fallback: string) {
  const [value, setValueState] = useState(fallback);

  // Hydrate from localStorage on mount — synchronously (no setTimeout). The prior
  // 1000ms delay showed the DEFAULT tab/pair for a full second after every load,
  // then jumped; a mid-second click acted on the wrong panel (F-H1). One render
  // is unavoidable with SSR (server has no localStorage); 1000ms was the bug.
  useEffect(() => {
    const stored = window.localStorage.getItem(key);
    if (stored === null || stored === value) return;
    let cancelled = false;
    queueMicrotask(() => {
      if (!cancelled) setValueState(stored);
    });
    return () => {
      cancelled = true;
    };
  }, [key, value]);

  const setValue = (next: string) => {
    setValueState(next);
    window.localStorage.setItem(key, next);
  };

  return [value, setValue] as const;
}

function DashboardBody() {
  const [selected, setSelected] = useState("EUR_USD");
  const [storedTab, setStoredTab] = useStoredValue("axiom:selectedTab", "Overview");
  const tab: Tab = TABS.includes(storedTab as Tab) ? (storedTab as Tab) : "Overview";
  const setTab = (next: Tab) => setStoredTab(next);
  const marketTabs = new Set<Tab>(["Markets"]);
  const ledgerTabs = new Set<Tab>(["Orders", "Executions", "Journal"]);
  const controlTabs = new Set<Tab>(["Risk", "Automation", "Settings"]);
  const { payload } = useStream();
  const halted = payload?.status?.halted ?? null;

  const [paletteOpen, setPaletteOpen] = useState(false);
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const commands: CommandItem[] = useMemo(() => {
    const nav: CommandItem[] = TABS.map((t) => ({
      id: `nav-${t}`, group: "Navigate", label: t, run: () => setTab(t),
    }));
    const instruments: CommandItem[] = MAJORS.map((m) => ({
      id: `inst-${m}`, group: "Instrument", label: m.replace("_", "/"),
      run: () => { setSelected(m); setTab("Overview"); },
    }));
    const actions: CommandItem[] = [
      {
        id: "act-halt", group: "Action", label: "Halt trading", hint: halted ? "already halted" : "fail-safe",
        run: async () => {
          if (halted) return;
          if (!window.confirm("Halt trading now?")) return;
          await control("halt");
        },
      },
      {
        id: "act-control", group: "Action", label: "Open Control panel",
        run: () => setTab("Automation"),
      },
    ];
    return [...nav, ...actions, ...instruments];
  }, [halted]);

  return (
    <div className="dashboard-shell">
      <div className="command-shell relative top-0 z-30 border-b hairline backdrop-blur-xl sm:sticky">
        <AccountHeader />
        <GlobalStatusStrip onOpenPalette={() => setPaletteOpen(true)} />

        <nav className="scroll-thin mx-auto flex w-full max-w-[1680px] gap-1 overflow-x-auto px-3 py-2 sm:px-4">
          <div className="flex min-w-max gap-4">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`shrink-0 border-b-2 pb-1 font-mono text-[12px] tracking-wide transition-colors ${
                  tab === t ? "border-pos font-semibold text-text" : "border-transparent text-faint hover:text-dim"
                }`}
              >
                {t.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="ml-auto hidden shrink-0 items-center gap-1 md:flex">
            {["⛶", "▥", "↗"].map((label) => (
              <button key={label} className="rounded-md border px-3 py-1.5 font-mono text-[11px] text-faint hairline hover:text-text" title="View option (chrome, not yet wired)">
                {label}
              </button>
            ))}
          </div>
        </nav>
      </div>

      <main className="mx-auto flex w-full min-w-0 max-w-[1680px] flex-col gap-4 px-3 py-4 sm:px-4 lg:py-5">
        {tab === "Overview" && (
          <>
            <PriceTiles selected={selected} onSelect={setSelected} />
            <div className="grid gap-3 xl:grid-cols-[minmax(0,2fr)_minmax(360px,0.95fr)]">
              <div className="h-[440px] min-w-0 lg:h-[500px]"><CandleChart instrument={selected} /></div>
              <div className="h-[440px] min-w-0 lg:h-[500px]"><Tier7Panel onViewLog={() => setTab("Analytics")} /></div>
            </div>
            <div className="grid gap-3 xl:grid-cols-[minmax(520px,2fr)_minmax(360px,0.95fr)]">
              <div className="h-[340px] min-w-0"><PositionsTable /></div>
              <div className="h-[340px] min-w-0"><EquityCurve /></div>
            </div>
          </>
        )}

        {tab === "Analytics" && <Tier7Cockpit />}

        {marketTabs.has(tab) && (
          <>
            <PriceTiles selected={selected} onSelect={setSelected} />
            <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(340px,0.9fr)]">
              <div className="h-[480px] min-w-0"><CandleChart instrument={selected} /></div>
              <div className="h-[480px] min-w-0"><StrategyPanel selected={selected} onSelect={setSelected} /></div>
            </div>
            <div className="h-[320px] min-w-0"><PositionsTable /></div>
          </>
        )}

        {tab === "Positions" && <div className="h-[520px] min-w-0"><PositionsTable /></div>}

        {controlTabs.has(tab) && <ControlPanel />}

        {ledgerTabs.has(tab) && (
          <>
            <div className="h-[300px] min-w-0"><EquityCurve /></div>
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="min-w-0 lg:col-span-2"><TradeHistory /></div>
              <div className="min-w-0"><SentimentPlaceholder /></div>
            </div>
          </>
        )}

      </main>

      <DashboardFooter />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} items={commands} />
    </div>
  );
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

function DashboardFooter() {
  const { payload, connected } = useStream();
  const acctId = payload?.account?.account_id;
  const clock = useClock();
  return (
    <footer className="mx-auto flex max-w-[1680px] flex-wrap items-center gap-4 border-t px-4 py-3 font-mono text-[10.5px] text-faint hairline">
      <span className={connected ? "text-pos" : "text-neg"}>● {connected ? "CONNECTED" : "OFFLINE"}</span>
      <span>LIVE DATA</span>
      <span>OANDA fxPractice ⌄</span>
      {acctId && <span title="Practice account id">Acct …{acctId.slice(-4)}</span>}
      <span className="text-pos">PRACTICE TRADING</span>
      <span className="ml-auto">{clock} (local)</span>
      <span className="text-pos">◎ SYNC</span>
      <span>v0.1.0</span>
    </footer>
  );
}

export default function Home() {
  return (
    <StreamProvider>
      <DashboardBody />
    </StreamProvider>
  );
}
