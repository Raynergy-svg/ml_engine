"use client";
import { usePoll } from "@/lib/api";
import type { Strategy } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Loading, Badge } from "./ui";
import { fmtNum, prettyPair } from "@/lib/format";

export function StrategyPanel({
  selected, onSelect,
}: { selected: string; onSelect: (i: string) => void }) {
  const { data, loading } = usePoll<Strategy>("/api/strategy", 60000);

  const gross = data ? Object.values(data.targets).reduce((a, b) => a + b, 0) : 0;
  const universe = data?.universe ?? [];
  const onSet = new Set(data?.on ?? []);

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={
          data ? (
            <span className="font-mono text-[11px] text-faint tnum">
              {data.on.length}/{universe.length} long · {fmtNum(gross * 100, 0)}% gross
            </span>
          ) : undefined
        }
      >
        Trend Strategy · price vs SMA(100) daily
      </SectionTitle>

      {loading && !data ? (
        <Loading label="Computing trend…" />
      ) : !data?.connected ? (
        <NotConnected
          label="Signal needs live candles"
          hint="OANDA practice candle endpoint unreachable. Strategy targets recompute from daily candles."
        />
      ) : (
        <div className="flex flex-1 flex-col gap-3 p-3">
          <div className="grid grid-cols-2 gap-1.5">
            {universe.map((inst) => {
              const on = onSet.has(inst);
              const w = data.targets[inst] ?? 0;
              return (
                <button
                  key={inst}
                  onClick={() => onSelect(inst)}
                  className={`flex items-center justify-between rounded-md border px-2.5 py-2 font-mono text-[12px] transition-colors ${
                    selected === inst ? "border-[var(--color-cyan)]" : "hairline hover:border-[var(--color-faint)]"
                  }`}
                >
                  <span className="font-semibold text-text">{prettyPair(inst)}</span>
                  <span
                    className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
                    style={{
                      color: on ? "#07090d" : "#8b98a9",
                      background: on ? "var(--color-emerald)" : "transparent",
                      border: on ? "none" : "1px solid var(--color-border)",
                    }}
                  >
                    {on ? `LONG ${fmtNum(w * 100, 0)}%` : "FLAT"}
                  </span>
                </button>
              );
            })}
          </div>
          <div className="mt-auto flex flex-wrap gap-2 border-t pt-3 hairline">
            <Badge color="#22d3ee">long-or-flat · managed futures</Badge>
            <Badge color="#8b98a9">shift-1 causal</Badge>
            <Badge color="#8b98a9">0.5× gross cap</Badge>
            {data.status?.halted === false ? (
              <Badge color="#2bd17e" dot pulse>not halted</Badge>
            ) : (
              <Badge color="#ff4d6d" dot>halted</Badge>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
