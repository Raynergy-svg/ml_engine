"use client";
import { usePoll } from "@/lib/api";
import type { Sentiment } from "@/lib/types";
import { Card, SectionTitle, NotConnected, Badge } from "./ui";
import { prettyPair } from "@/lib/format";

export function SentimentPlaceholder() {
  const { data } = usePoll<Sentiment>("/api/sentiment", 60000);
  const agentLabel = data?.strategy_agent ?? "order_flow";

  return (
    <Card className="flex h-full flex-col">
      <SectionTitle
        right={<Badge color={data?.wired_into_strategy ? "#2bd17e" : "#e84ff0"} dot={data?.wired_into_strategy}>
          {data?.wired_into_strategy ? `${agentLabel} · ${data.agent_weight?.toFixed(2) ?? "—"}w` : "DATA-ONLY"}
        </Badge>}
      >
        Order-Book Sentiment
      </SectionTitle>
      <div className="flex-1 p-3">
        <p className="mb-3 text-[11.5px] leading-relaxed text-dim">
          Retail position-book crowding from OANDA v20 feeds the scanner&apos;s
          <span className="text-magenta"> order_flow</span> contrarian signal when enabled.
        </p>
        {!data?.connected ? (
          <NotConnected label="No sentiment snapshot" />
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {Object.entries(data.books).map(([inst, buckets]) => {
              const longPct = buckets.reduce((a, b) => a + parseFloat(b.longCountPercent || "0"), 0);
              const shortPct = buckets.reduce((a, b) => a + parseFloat(b.shortCountPercent || "0"), 0);
              const total = longPct + shortPct || 1;
              const lp = (longPct / total) * 100;
              const features = data.features?.[inst] ?? {};
              const totalImbalance = Number(features.pb_total_imbalance ?? 0);
              const nearImbalance = Number(features.pb_near_imbalance ?? 0);
              return (
                <div key={inst} className="rounded-md border p-2.5 hairline">
                  <div className="mb-1.5 flex items-center justify-between font-mono text-[12px]">
                    <span className="font-semibold text-text">{prettyPair(inst)}</span>
                    <span className="text-faint tnum">{buckets.length} buckets · pb</span>
                  </div>
                  <div className="flex h-2 overflow-hidden rounded-full bg-surface2">
                    <div style={{ width: `${lp}%`, background: "#2bd17e" }} />
                    <div style={{ width: `${100 - lp}%`, background: "#ff4d6d" }} />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[10px] text-faint tnum">
                    <span style={{ color: "#2bd17e" }}>{lp.toFixed(0)}% long</span>
                    <span style={{ color: "#ff4d6d" }}>{(100 - lp).toFixed(0)}% short</span>
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-1 font-mono text-[10px] text-faint tnum">
                    <span>imb {totalImbalance >= 0 ? "+" : ""}{totalImbalance.toFixed(2)}</span>
                    <span>near {nearImbalance >= 0 ? "+" : ""}{nearImbalance.toFixed(2)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}
