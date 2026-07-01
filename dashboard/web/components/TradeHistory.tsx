"use client";
import { usePoll } from "@/lib/api";
import type { Trade, Trades } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading } from "./ui";
import { fmtSigned, pnlClass, fmtUnits, fmtPrice, fmtMoney, shortTime, prettyPair } from "@/lib/format";

function statusColor(status: Trade["status"]): string {
  if (status === "FILLED" || status === "POSTED") return "#2bd17e";
  if (status === "ACTIVE") return "#22d3ee";
  if (status === "CANCELLED") return "#8b98a9";
  if (status === "REJECTED") return "#ff4d6d";
  return "#f6b44b";
}

function txLabel(type: string): string {
  return type.replaceAll("_", " ");
}

function txNote(t: Trade): string | null {
  if (t.reject_reason) return t.reject_reason;
  if (t.fill_kind) return t.fill_kind;
  return t.reason;
}

function linkLabel(t: Trade): string {
  if (t.type === "ORDER_FILL" && t.order_id) return `order #${t.order_id}`;
  if (t.linked_fill_id) return `fill #${t.linked_fill_id}`;
  if (t.order_id) return `order #${t.order_id}`;
  return "—";
}

export function TradeHistory() {
  const { data, loading } = usePoll<Trades>("/api/trades?limit=200", 15000);
  const right = data
    ? <span className="font-mono text-[11px] text-faint tnum">{data.count} tx · {data.fill_count} fills</span>
    : undefined;

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle right={right}>
        Order Ledger · OANDA transaction audit
      </SectionTitle>
      {loading && !data ? (
        <Loading label="Loading ledger…" />
      ) : !data?.trades?.length ? (
        <NotConnected label="No ledger rows yet" hint="Reads trained_data/oanda/transactions.jsonl - the bot's audit trail." />
      ) : (
        <div className="scroll-thin max-h-[340px] overflow-auto">
          <table className="data-table min-w-[980px] w-full font-mono text-[11.5px] tnum">
            <thead className="sticky top-0 bg-surface">
              <tr className="eyebrow text-left">
                <th className="px-3 py-2 font-normal">Time (UTC)</th>
                <th className="px-3 py-2 font-normal">Txn</th>
                <th className="px-3 py-2 font-normal">Status</th>
                <th className="px-3 py-2 font-normal">Pair</th>
                <th className="px-3 py-2 font-normal">Side</th>
                <th className="px-3 py-2 text-right font-normal">Units</th>
                <th className="px-3 py-2 text-right font-normal">Exec</th>
                <th className="px-3 py-2 text-right font-normal">Realized P&L</th>
                <th className="px-3 py-2 text-right font-normal">Balance</th>
                <th className="px-3 py-2 text-right font-normal">Link</th>
              </tr>
            </thead>
            <tbody>
              {data.trades.map((t) => {
                const note = txNote(t);
                const hasUnits = Boolean(t.instrument) && t.units !== 0;
                return (
                  <tr key={t.id} className="border-t hairline">
                    <td className="whitespace-nowrap px-3 py-1.5 text-dim">{shortTime(t.time)}</td>
                    <td className="px-3 py-1.5">
                      <div className="whitespace-nowrap font-semibold text-text">{txLabel(t.type)}</div>
                      {note && <div className="max-w-[190px] truncate text-[10px] text-faint">{note}</div>}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5">
                      <span className="font-semibold" style={{ color: statusColor(t.status) }}>{t.status}</span>
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 font-semibold text-text">
                      {t.instrument ? prettyPair(t.instrument) : "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5">
                      {hasUnits ? (
                        <span style={{ color: t.side === "BUY" ? "#2bd17e" : "#ff4d6d" }}>{t.side}</span>
                      ) : "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-dim">{hasUnits ? fmtUnits(t.units) : "—"}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-dim">
                      {fmtPrice(t.price ?? t.linked_fill_price, t.instrument ?? undefined)}
                    </td>
                    <td className={`whitespace-nowrap px-3 py-1.5 text-right ${pnlClass(t.pl)}`}>
                      {t.type === "ORDER_FILL" && t.pl !== 0 ? fmtSigned(t.pl) : "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-faint">{t.balance ? fmtMoney(t.balance) : "—"}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-faint">{linkLabel(t)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
