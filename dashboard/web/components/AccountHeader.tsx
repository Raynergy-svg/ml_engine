"use client";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { ApiHealth, Trades } from "@/lib/types";
import { Logo } from "./Logo";
import { Badge, StatusDot } from "./ui";
import { fmtMoney, fmtSigned, pnlClass, fmtNum, ago } from "@/lib/format";

function Stat({ label, children, sub }: { label: string; children: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 px-5">
      <span className="eyebrow">{label}</span>
      <span className="font-mono text-[22px] font-semibold leading-tight tnum">{children}</span>
      {sub && <span className="font-mono text-[11px] text-dim tnum">{sub}</span>}
    </div>
  );
}

export function AccountHeader() {
  const { payload, connected } = useStream();
  const { data: trades } = usePoll<Trades>("/api/trades?limit=500", 15000);
  const { data: apiHealth } = usePoll<ApiHealth>("/api/health", 10000);
  const acct = payload?.account;
  const status = payload?.status;

  // Day P&L = P&L realized since today's UTC midnight + current unrealized.
  // Gated on BOTH sources loaded — otherwise we'd transiently show unrealized-only
  // as if it were the settled day figure (an understated number that looks real).
  let realizedToday = 0;
  if (trades?.trades) {
    const startOfDay = new Date();
    startOfDay.setUTCHours(0, 0, 0, 0);
    for (const t of trades.trades) {
      if (new Date(t.time).getTime() >= startOfDay.getTime()) realizedToday += t.pl;
    }
  }
  const unrealized = acct?.unrealized_pl ?? 0;
  const dayPnl = acct && trades?.trades ? realizedToday + unrealized : null;

  const halted = status?.halted ?? true;
  const running = status?.running ?? false;

  return (
    <header className="card sticky top-0 z-20 flex flex-wrap items-center gap-y-4 rounded-none border-x-0 border-t-0 px-6 py-3.5 backdrop-blur">
      <div className="flex items-center gap-4 pr-6">
        <Logo height={34} />
        <div className="hidden flex-col border-l pl-4 hairline sm:flex">
          <span className="font-mono text-[11px] tracking-wide text-dim">Buddy engine</span>
          <span className="font-mono text-[10px] text-faint">{acct?.account_id ?? "—"}</span>
        </div>
      </div>

      <div className="flex flex-1 flex-wrap items-center gap-y-3 divide-x divide-[var(--color-border)]">
        <Stat label="NAV" sub={acct?.currency ? `${acct.currency}` : undefined}>
          {acct ? fmtMoney(acct.nav) : "—"}
        </Stat>
        <Stat label="Day P&L" sub="realized today (UTC) + unrealized">
          <span className={pnlClass(dayPnl)}>{dayPnl === null ? "—" : fmtSigned(dayPnl)}</span>
        </Stat>
        <Stat label="Unrealized" sub={`${acct?.open_trade_count ?? 0} open trades`}>
          <span className={pnlClass(unrealized)}>{acct ? fmtSigned(unrealized) : "—"}</span>
        </Stat>
        <Stat label="Total Realized" sub="since inception">
          <span className={pnlClass(acct?.realized_pl)}>{acct ? fmtSigned(acct.realized_pl) : "—"}</span>
        </Stat>
        <Stat
          label="Margin"
          sub={acct ? `${fmtNum((acct.margin_used / (acct.nav || 1)) * 100, 1)}% used` : undefined}
        >
          {acct ? fmtMoney(acct.margin_available) : "—"}
        </Stat>
      </div>

      <div className="flex items-center gap-2 pl-4">
        <Badge color="#22d3ee" dot>PRACTICE</Badge>
        <Badge color={apiHealth?.control_enabled ? "#f5b14c" : "#5a6677"} dot>
          {apiHealth?.control_enabled ? "CONTROL" : "READ-ONLY"}
        </Badge>
        {halted ? (
          <Badge color="#ff4d6d" dot>HALTED</Badge>
        ) : running ? (
          <Badge color="#2bd17e" dot pulse>RUNNING</Badge>
        ) : (
          <Badge color="#f5b14c" dot>IDLE</Badge>
        )}
        {acct?.stale && (
          <Badge color="#f5b14c" dot title="Account snapshot is stale — the trend lane may be stopped. Figures are last-known, not live.">
            STALE{acct.snapshot_age_s != null ? ` ${ago(acct.snapshot_age_s)}` : ""}
          </Badge>
        )}
        <div className="flex items-center gap-1.5 pl-1" title="SSE stream to data layer">
          <StatusDot color={connected ? "#34e5a1" : "#5a6677"} pulse={connected} />
          <span className="font-mono text-[10px] text-faint">
            {connected ? "live" : "offline"}
            {status?.account_snapshot_age_s != null && ` · cycle ${ago(status.account_snapshot_age_s)}`}
          </span>
        </div>
        <button
          onClick={async () => {
            await fetch("/auth/logout", { method: "POST" });
            window.location.href = "/login";
          }}
          title="Lock / sign out"
          className="ml-1 rounded-md border px-2 py-1 font-mono text-[10px] text-faint hairline hover:text-dim"
        >
          lock
        </button>
      </div>
    </header>
  );
}
