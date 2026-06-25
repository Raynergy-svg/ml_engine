"use client";
import { useState } from "react";
import { StreamProvider } from "@/lib/stream";
import { AccountHeader } from "@/components/AccountHeader";
import { PriceTiles } from "@/components/PriceTiles";
import { CandleChart } from "@/components/CandleChart";
import { StrategyPanel } from "@/components/StrategyPanel";
import { PositionsTable } from "@/components/PositionsTable";
import { EquityCurve } from "@/components/EquityCurve";
import { TradeHistory } from "@/components/TradeHistory";
import { SentimentPlaceholder } from "@/components/SentimentPlaceholder";

export default function Home() {
  const [selected, setSelected] = useState("USD_JPY");

  return (
    <StreamProvider>
      <AccountHeader />
      <main className="mx-auto flex max-w-[1640px] flex-col gap-4 px-4 py-4">
        <PriceTiles selected={selected} onSelect={setSelected} />

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="h-[460px] lg:col-span-2">
            <CandleChart instrument={selected} />
          </div>
          <div className="h-[460px]">
            <StrategyPanel selected={selected} onSelect={setSelected} />
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="h-[300px] lg:col-span-2">
            <EquityCurve />
          </div>
          <div className="h-[300px]">
            <PositionsTable />
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <TradeHistory />
          </div>
          <SentimentPlaceholder />
        </div>
      </main>

      <footer className="mx-auto max-w-[1640px] px-4 pb-8 pt-2 font-mono text-[10.5px] text-faint">
        AXIOM · read-only terminal for the Buddy trading engine · OANDA fxPractice (demo) ·
        visualizes, never trades · data from bot state files + read-only v20 reads
      </footer>
    </StreamProvider>
  );
}
