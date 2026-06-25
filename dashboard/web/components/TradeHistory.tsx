"use client";
import { usePoll } from "@/lib/api";
import type { Trades } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading } from "./ui";
import { fmtSigned, pnlClass, fmtUnits, fmtPrice, fmtMoney, shortTime, prettyPair } from "@/lib/format";

export function TradeHistory() {
  const { data, loading } = usePoll<Trades>("/api/trades?limit=200", 15000);

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle right={data ? <span className="font-mono text-[11px] text-faint tnum">{data.count} fills</span> : undefined}>
        Transaction Ledger · ORDER_FILL audit trail
      </SectionTitle>
      {loading && !data ? (
        <Loading label="Loading ledger…" />
      ) : !data?.trades?.length ? (
        <NotConnected label="No fills yet" hint="Reads trained_data/oanda/transactions.jsonl — the bot's audit trail." />
      ) : (
        <div className="scroll-thin max-h-[340px] overflow-auto">
          <table className="w-full font-mono text-[11.5px] tnum">
            <thead className="sticky top-0 bg-surface">
              <tr className="eyebrow text-left">
                <th className="px-3 py-2 font-normal">Time (UTC)</th>
                <th className="px-3 py-2 font-normal">Pair</th>
                <th className="px-3 py-2 font-normal">Side</th>
                <th className="px-3 py-2 text-right font-normal">Units</th>
                <th className="px-3 py-2 text-right font-normal">Price</th>
                <th className="px-3 py-2 text-right font-normal">Realized P&L</th>
                <th className="px-3 py-2 text-right font-normal">Balance</th>
              </tr>
            </thead>
            <tbody>
              {data.trades.map((t) => (
                <tr key={t.id} className="border-t hairline hover:bg-surface2/40">
                  <td className="px-3 py-1.5 text-dim">{shortTime(t.time)}</td>
                  <td className="px-3 py-1.5 font-semibold text-text">{prettyPair(t.instrument)}</td>
                  <td className="px-3 py-1.5">
                    <span style={{ color: t.side === "BUY" ? "#2bd17e" : "#ff4d6d" }}>{t.side}</span>
                  </td>
                  <td className="px-3 py-1.5 text-right text-dim">{fmtUnits(t.units)}</td>
                  <td className="px-3 py-1.5 text-right text-dim">{fmtPrice(t.price, t.instrument)}</td>
                  <td className={`px-3 py-1.5 text-right ${pnlClass(t.pl)}`}>{t.pl === 0 ? "—" : fmtSigned(t.pl)}</td>
                  <td className="px-3 py-1.5 text-right text-faint">{t.balance ? fmtMoney(t.balance) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
