"use client";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { Strategy, Position } from "@/lib/types";
import { Card, SectionTitle, NotConnected } from "./ui";
import { fmtSigned, pnlClass, fmtUnits, fmtPrice, prettyPair } from "@/lib/format";

function pipSize(instrument: string): number {
  return instrument.endsWith("_JPY") ? 0.01 : 0.0001;
}

/** TP/SL cell: price level + distance in pips from the live mid, or an honest "—". */
function BracketCell({ level, mid, instrument, color }: {
  level: number | null | undefined; mid: number | undefined; instrument: string; color: string;
}) {
  if (level == null) {
    return <td className="px-2 py-2 text-right text-faint" title="no bracket order on this position">—</td>;
  }
  const pips = mid != null ? Math.abs(level - mid) / pipSize(instrument) : null;
  return (
    <td className="px-2 py-2 text-right">
      <span style={{ color }}>{fmtPrice(level, instrument)}</span>
      {pips != null && <span className="block text-[10px] text-faint tnum">{pips.toFixed(1)}p</span>}
    </td>
  );
}

export function PositionsTable() {
  const { payload } = useStream();
  const { data: strat } = usePoll<Strategy>("/api/strategy", 60000);
  const positions: Position[] = payload?.account?.positions ?? [];
  const prices = payload?.prices?.prices ?? {};
  const onSet = new Set(strat?.on ?? []);
  // TP/SL are present only once the bot writes them into account_state.json (additive).
  const anyBracket = positions.some((p) => p.take_profit != null || p.stop_loss != null);

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={
          <span className="font-mono text-[11px] text-faint tnum">
            {positions.length} open{!anyBracket && positions.length > 0 ? " · no TP/SL yet" : ""}
          </span>
        }
      >
        Open Positions · live uPL · TP/SL
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
                <th className="px-2 py-1.5 text-right font-normal">SL</th>
                <th className="px-2 py-1.5 text-right font-normal">TP</th>
                <th className="px-2 py-1.5 text-right font-normal">Trend</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => {
                const mid = prices[p.instrument]?.mid;
                return (
                  <tr key={p.instrument} className="border-t hairline">
                    <td className="px-2 py-2 font-semibold text-text">{prettyPair(p.instrument)}</td>
                    <td className="px-2 py-2 text-right text-dim">{fmtUnits(p.net_units)}</td>
                    <td className={`px-2 py-2 text-right font-semibold ${pnlClass(p.unrealized_pl)}`}>
                      {fmtSigned(p.unrealized_pl)}
                    </td>
                    <BracketCell level={p.stop_loss} mid={mid} instrument={p.instrument} color="#ff4d6d" />
                    <BracketCell level={p.take_profit} mid={mid} instrument={p.instrument} color="#2bd17e" />
                    <td className="px-2 py-2 text-right">
                      <span style={{ color: onSet.has(p.instrument) ? "#34e5a1" : "#8b98a9" }}>
                        {onSet.has(p.instrument) ? "LONG" : "FLAT"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!anyBracket && (
            <p className="px-2 pt-2 text-[10px] text-faint">
              TP/SL columns populate once the bot writes bracket levels into account_state.json
              (pending backend contract).
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
