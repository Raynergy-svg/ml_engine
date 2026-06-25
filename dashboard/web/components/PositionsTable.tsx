"use client";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { Strategy } from "@/lib/types";
import { Card, SectionTitle, NotConnected } from "./ui";
import { fmtSigned, pnlClass, fmtUnits, prettyPair } from "@/lib/format";

export function PositionsTable() {
  const { payload } = useStream();
  const { data: strat } = usePoll<Strategy>("/api/strategy", 60000);
  const positions = payload?.account?.positions ?? [];
  const onSet = new Set(strat?.on ?? []);

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle right={<span className="font-mono text-[11px] text-faint tnum">{positions.length} open</span>}>
        Open Positions · live uPL
      </SectionTitle>
      {positions.length === 0 ? (
        <NotConnected label="No open positions" hint="The trend lane is flat across the universe, or awaiting its first cycle." />
      ) : (
        <div className="scroll-thin overflow-auto px-2 pb-2">
          <table className="w-full font-mono text-[12px] tnum">
            <thead>
              <tr className="eyebrow text-left">
                <th className="px-2 py-1.5 font-normal">Pair</th>
                <th className="px-2 py-1.5 text-right font-normal">Net units</th>
                <th className="px-2 py-1.5 text-right font-normal">Unrealized</th>
                <th className="px-2 py-1.5 text-right font-normal">Trend</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.instrument} className="border-t hairline">
                  <td className="px-2 py-2 font-semibold text-text">{prettyPair(p.instrument)}</td>
                  <td className="px-2 py-2 text-right text-dim">{fmtUnits(p.net_units)}</td>
                  <td className={`px-2 py-2 text-right font-semibold ${pnlClass(p.unrealized_pl)}`}>
                    {fmtSigned(p.unrealized_pl)}
                  </td>
                  <td className="px-2 py-2 text-right">
                    <span style={{ color: onSet.has(p.instrument) ? "#34e5a1" : "#8b98a9" }}>
                      {onSet.has(p.instrument) ? "LONG" : "FLAT"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
