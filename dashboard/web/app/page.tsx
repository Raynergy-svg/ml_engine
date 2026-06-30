"use client";
import { useEffect, useState } from "react";
import { StreamProvider } from "@/lib/stream";
import { usePoll } from "@/lib/api";
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
import { HealthPanel } from "@/components/HealthPanel";
import { ControlPanel } from "@/components/ControlPanel";
import type { ApiHealth } from "@/lib/types";

const TABS = ["Overview", "Tier 7", "Strategy", "Health", "Ledger", "Control"] as const;
type Tab = (typeof TABS)[number];

function useStoredValue(key: string, fallback: string) {
  const [value, setValueState] = useState(fallback);

  // Hydrate from localStorage on mount — synchronously (no setTimeout). The prior
  // 1000ms delay showed the DEFAULT tab/pair for a full second after every load,
  // then jumped; a mid-second click acted on the wrong panel (F-H1). One render
  // is unavoidable with SSR (server has no localStorage); 1000ms was the bug.
  useEffect(() => {
    const stored = window.localStorage.getItem(key);
    if (stored !== null && stored !== value) setValueState(stored);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const setValue = (next: string) => {
    setValueState(next);
    window.localStorage.setItem(key, next);
  };

  return [value, setValue] as const;
}

function DashboardBody() {
  const { data: apiHealth } = usePoll<ApiHealth>("/api/health", 10000);
  const [selected, setSelected] = useStoredValue("axiom:selectedPair", "USD_JPY");
  const [storedTab, setStoredTab] = useStoredValue("axiom:selectedTab", "Overview");
  const tab: Tab = TABS.includes(storedTab as Tab) ? (storedTab as Tab) : "Overview";
  const setTab = (next: Tab) => setStoredTab(next);

  return (
    <>
      <AccountHeader />
      <GlobalStatusStrip />

      {/* tab nav */}
      <nav className="scroll-thin sticky top-0 z-10 flex gap-1 overflow-x-auto border-b bg-base/80 px-4 py-2 backdrop-blur hairline">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`shrink-0 rounded-md px-3.5 py-1.5 font-mono text-[12px] transition-colors ${
              tab === t ? "bg-surface2 text-cyan" : "text-faint hover:text-dim"
            }`}
            style={tab === t ? { boxShadow: "inset 0 -2px 0 var(--color-cyan)" } : undefined}
          >
            {t}
          </button>
        ))}
      </nav>

      <main className="mx-auto flex w-full min-w-0 max-w-[1640px] flex-col gap-4 px-4 py-4">
        {tab === "Overview" && (
          <>
            <PriceTiles selected={selected} onSelect={setSelected} />
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="h-[440px] lg:col-span-2"><CandleChart instrument={selected} /></div>
              <div className="h-[440px]"><Tier7Panel /></div>
            </div>
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="h-[300px] lg:col-span-2"><EquityCurve /></div>
              <div className="h-[300px]"><PositionsTable /></div>
            </div>
          </>
        )}

        {tab === "Tier 7" && <Tier7Cockpit />}

        {tab === "Strategy" && (
          <>
            <PriceTiles selected={selected} onSelect={setSelected} />
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="h-[480px] lg:col-span-2"><CandleChart instrument={selected} /></div>
              <div className="h-[480px]"><StrategyPanel selected={selected} onSelect={setSelected} /></div>
            </div>
            <div className="h-[320px]"><PositionsTable /></div>
          </>
        )}

        {tab === "Health" && <HealthPanel />}

        {tab === "Control" && <ControlPanel />}

        {tab === "Ledger" && (
          <>
            <div className="h-[300px]"><EquityCurve /></div>
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2"><TradeHistory /></div>
              <SentimentPlaceholder />
            </div>
          </>
        )}
      </main>

      <footer className="mx-auto max-w-[1640px] px-4 pb-8 pt-2 font-mono text-[10.5px] text-faint">
        AXIOM · operator cockpit for the Buddy trading engine · OANDA fxPractice (demo) ·
        mirrors the backend · {apiHealth?.control_enabled ? "bounded audited controls enabled" : "read-only mode"}
      </footer>
    </>
  );
}

export default function Home() {
  return (
    <StreamProvider>
      <DashboardBody />
    </StreamProvider>
  );
}
