import type { ShadowLaneFreshness } from "@/lib/types";
import { ago, shortTime } from "@/lib/format";
import { Badge } from "./ui";

function cadence(seconds: number): string {
  if (seconds % 86_400 === 0) return `${seconds / 86_400}d`;
  if (seconds % 3_600 === 0) return `${seconds / 3_600}h`;
  return `${seconds}s`;
}

export function ShadowFreshnessBadge({
  freshness,
}: {
  freshness: ShadowLaneFreshness;
}) {
  const color =
    freshness.status === "fresh"
      ? "#2bd17e"
      : freshness.status === "stale"
        ? "#f5b14c"
        : "#5a6677";
  const label =
    freshness.status === "fresh"
      ? `FRESH · ${ago(freshness.age_s)}`
      : freshness.status === "stale"
        ? `STALE · ${ago(freshness.age_s)}`
        : "NO READBACK";
  return (
    <Badge
      color={color}
      dot
      title="Freshness derives from the ledger writer's cycle timestamp"
    >
      {label}
    </Badge>
  );
}

export function ShadowLaneReadback({
  freshness,
  source,
}: {
  freshness: ShadowLaneFreshness;
  source: Record<string, string>;
}) {
  return (
    <div className="mt-2 border-t pt-2 font-mono text-[10px] text-faint hairline">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <ShadowFreshnessBadge freshness={freshness} />
        <span>last evaluated {shortTime(freshness.last_evaluated_at)}</span>
        <span>· expected every {cadence(freshness.expected_interval_s)}</span>
      </div>
      <div className="mt-1 break-all">
        ledger readback: {source.ledger ?? "unavailable"}
      </div>
    </div>
  );
}
