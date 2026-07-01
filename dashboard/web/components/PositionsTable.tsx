"use client";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { Strategy, Position } from "@/lib/types";
import { Card, SectionTitle, NotConnected } from "./ui";
import { fmtSigned, pnlClass, fmtNum, fmtPrice, fmtPct, ago, prettyPair } from "@/lib/format";

function ageSeconds(openedAt?: string | null): number | null {
  if (!openedAt) return null;
  const t = new Date(openedAt).getTime();
  if (Number.isNaN(t)) return null;
  return (Date.now() - t) / 1000;
}

export function PositionsTable() {
  const { payload } = useStream();
  const { data: strat } = usePoll<Strategy>("/api/strategy", 60000);
  const positions: Position[] = payload?.account?.positions ?? [];
  const prices = payload?.prices?.prices ?? {};
  const onSet = new Set(strat?.on ?? []);

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={<span className="font-mono text-[11px] text-faint tnum">⋮⋮</span>}
      >
        Positions ({positions.length})
      </SectionTitle>
      {positions.length === 0 ? (
        <NotConnected label="No open positions" hint="The trend lane is flat across the universe, or awaiting its first cycle." />
      ) : (
        <div className="scroll-thin overflow-auto px-2 pb-2">
          <table className="data-table min-w-[980px] w-full font-mono text-[12px] tnum">
            <thead>
              <tr className="eyebrow text-left">
                <th className="px-3 py-2 font-normal">Symbol</th>
                <th className="px-3 py-2 font-normal">Side</th>
                <th className="px-3 py-2 text-right font-normal">Size</th>
                <th className="px-3 py-2 text-right font-normal">Entry</th>
                <th className="px-3 py-2 text-right font-normal">Market</th>
                <th className="px-3 py-2 text-right font-normal">P&L (USD)</th>
                <th className="px-3 py-2 text-right font-normal">P&L (%)</th>
                <th className="px-3 py-2 text-right font-normal">Stop</th>
                <th className="px-3 py-2 text-right font-normal">Target</th>
                <th className="px-3 py-2 text-right font-normal">Age</th>
                <th className="px-3 py-2 text-right font-normal">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => {
                const market = prices[p.instrument]?.mid ?? null;
                const entry = p.entry_price ?? null;
                const side = p.side ?? (p.net_units > 0 ? "LONG" : p.net_units < 0 ? "SHORT" : "FLAT");
                const dirSign = side === "SHORT" ? -1 : 1;
                const pctReturn = entry && market ? ((market - entry) / entry) * 100 * dirSign : null;
                const age = ageSeconds(p.opened_at);
                const trendOn = onSet.has(p.instrument);
                return (
                  <tr key={p.instrument} className="border-t hairline">
                    <td className="px-3 py-2 font-semibold text-text">{prettyPair(p.instrument)}</td>
                    <td className="px-3 py-2 font-semibold" style={{ color: side === "SHORT" ? "#ff4d6d" : "#2bd17e" }}>
                      {side}
                    </td>
                    <td className="px-3 py-2 text-right text-dim">{fmtNum(Math.abs(p.net_units), 0)}</td>
                    <td className="px-3 py-2 text-right text-dim">{entry != null ? fmtPrice(entry, p.instrument) : "—"}</td>
                    <td className="px-3 py-2 text-right text-dim">{market != null ? fmtPrice(market, p.instrument) : "—"}</td>
                    <td className={`px-3 py-2 text-right font-semibold ${pnlClass(p.unrealized_pl)}`}>
                      {fmtSigned(p.unrealized_pl)}
                    </td>
                    <td className={`px-3 py-2 text-right ${pctReturn == null ? "text-faint" : pnlClass(pctReturn)}`}>
                      {pctReturn == null ? "—" : `${pctReturn >= 0 ? "+" : ""}${fmtPct(pctReturn, 2)}`}
                    </td>
                    <td className="px-3 py-2 text-right" style={{ color: p.stop_loss != null ? "#ff4d6d" : undefined }}>
                      {p.stop_loss != null ? fmtPrice(p.stop_loss, p.instrument) : <span className="text-faint">—</span>}
                    </td>
                    <td className="px-3 py-2 text-right" style={{ color: p.take_profit != null ? "#2bd17e" : undefined }}>
                      {p.take_profit != null ? fmtPrice(p.take_profit, p.instrument) : <span className="text-faint">—</span>}
                    </td>
                    <td className="px-3 py-2 text-right text-faint">{age != null ? ago(age) : "—"}</td>
                    <td className="px-3 py-2 text-right">
                      <span className="text-dim" title="Long-or-flat SMA(100) trend rule">{p.strategy ?? "trend_sma100"}</span>
                      <span className="ml-2" style={{ color: trendOn ? "#34e5a1" : "#8b98a9" }}>{trendOn ? "●" : "○"}</span>
                    </td>
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
