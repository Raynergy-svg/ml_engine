"use client";
import { useState } from "react";
import { StreamProvider } from "@/lib/stream";
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

const TABS = ["Overview", "Tier 7", "Strategy", "Health", "Ledger", "Control"] as const;
type Tab = (typeof TABS)[number];

export default function Home() {
  const [selected, setSelected] = useState("USD_JPY");
  const [tab, setTab] = useState<Tab>("Overview");

  return (
    <StreamProvider>
      <AccountHeader />
      <GlobalStatusStrip />

      {/* tab nav */}
      <nav className="sticky top-0 z-10 flex gap-1 border-b bg-base/80 px-4 py-2 backdrop-blur hairline">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-md px-3.5 py-1.5 font-mono text-[12px] transition-colors ${
              tab === t ? "bg-surface2 text-cyan" : "text-faint hover:text-dim"
            }`}
            style={tab === t ? { boxShadow: "inset 0 -2px 0 var(--color-cyan)" } : undefined}
          >
            {t}
          </button>
        ))}
      </nav>

      <main className="mx-auto flex max-w-[1640px] flex-col gap-4 px-4 py-4">
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
        AXIOM · read-only ops terminal for the Buddy trading engine · OANDA fxPractice (demo) ·
        mirrors the backend · never trades · control layer built-but-disabled (Phase 2)
      </footer>
    </StreamProvider>
  );
}
