"use client";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import type { Trades, SystemHealth, ControlState } from "@/lib/types";
import { Logo } from "./Logo";
import { Badge } from "./ui";
import { ShieldCheckIcon, ChevronDown } from "./icons";
import { fmtMoney, fmtSigned, pnlClass, fmtNum, fmtPct } from "@/lib/format";

function TopMetric({ label, children, title, positive }: {
  label: string; children: React.ReactNode; title?: string; positive?: boolean;
}) {
  return (
    <div className="min-w-[92px] px-3 py-2" title={title}>
      <span className="block font-mono text-[10px] uppercase tracking-wide text-dim">{label}</span>
      <span className={`mt-1 block whitespace-nowrap font-mono text-[14px] font-semibold tnum ${positive === true ? "text-pos" : positive === false ? "text-neg" : "text-text"}`}>
        {children}
      </span>
    </div>
  );
}

function StatusBlock({ label, value, ok, unknown }: { label: string; value: string; ok: boolean; unknown?: boolean }) {
  const color = unknown ? "#5a6677" : ok ? "#3ee287" : "#ff6b5e";
  return (
    <div className="hidden min-w-[132px] border-l px-5 py-2 hairline md:block">
      <span className="block font-mono text-[10px] uppercase tracking-wide text-dim">{label}</span>
      <span className="mt-1 flex items-center gap-2 font-mono text-[13px] font-semibold" style={{ color }}>
        <ShieldCheckIcon />
        {value}
      </span>
    </div>
  );
}

export function AccountHeader() {
  const { payload, connected } = useStream();
  const acct = payload?.account;
  const status = payload?.status;
  const { data: trades } = usePoll<Trades>("/api/trades?limit=500", 15000);
  const { data: health } = usePoll<SystemHealth>("/api/system_health", 5000);
  const { data: control, error: controlErr } = usePoll<ControlState>("/api/control/state", 5000);

  // Day P&L: realized pl of today's UTC fills + current unrealized (gated on both
  // sources so it never transiently shows unrealized-only as the settled figure).
  let realizedToday = 0;
  if (trades?.trades) {
    const startOfDay = new Date();
    startOfDay.setUTCHours(0, 0, 0, 0);
    for (const t of trades.trades) {
      if (t.type === "ORDER_FILL" && new Date(t.time).getTime() >= startOfDay.getTime()) realizedToday += t.pl;
    }
  }
  const unrealized = acct?.unrealized_pl ?? 0;
  const dayPnl = acct && trades?.trades ? realizedToday + unrealized : null;

  // Win rate: real fraction of CLOSING fills (pl != 0) that were profitable.
  let winRate: number | null = null;
  if (trades?.trades) {
    let wins = 0, closes = 0;
    for (const t of trades.trades) {
      if (t.type !== "ORDER_FILL" || t.pl === 0) continue;
      closes += 1;
      if (t.pl > 0) wins += 1;
    }
    winRate = closes > 0 ? (wins / closes) * 100 : null;
  }

  const drawdownPct = acct?.drawdown_pct != null ? acct.drawdown_pct * 100 : null;
  const marginUsedPct = acct && acct.nav > 0 ? (acct.margin_used / acct.nav) * 100 : null;
  const leverage = !controlErr ? control?.gross_leverage ?? null : null;

  const laneRunning = health?.lanes?.running ?? status?.running ?? null;
  const halted = status?.halted ?? null;
  const systemLabel = halted === true ? "HALTED" : laneRunning === true ? "RUNNING" : laneRunning === false ? "OFFLINE" : "—";
  const gatesStatus = health?.gates?.available ? health.gates.status : null;
  const gateLabel = gatesStatus === "GREEN" ? "ALL CLEAR" : gatesStatus === "RED" ? "ISSUES" : "—";

  return (
    <header className="mx-auto flex min-h-[70px] w-full max-w-[1680px] flex-wrap items-center gap-3 border-b px-3 py-2 hairline sm:px-4 xl:flex-nowrap">
      <div className="flex min-w-[220px] items-center gap-4 pr-2">
        <Logo height={38} />
        <Badge color="#2bd17e">PRACTICE</Badge>
        <span className="hidden whitespace-nowrap font-mono text-[11px] uppercase tracking-wide text-dim lg:inline">
          Forex engine
        </span>
      </div>

      <StatusBlock label="System status" value={systemLabel} ok={halted !== true && laneRunning === true} unknown={laneRunning === null} />
      <StatusBlock label="Gate status" value={gateLabel} ok={gatesStatus === "GREEN"} unknown={gatesStatus == null} />

      <div className="order-3 grid w-full min-w-0 grid-cols-2 rounded-md border bg-surface/55 hairline sm:grid-cols-3 lg:order-none lg:ml-auto lg:w-auto xl:flex">
        <TopMetric label="Equity">{acct ? fmtMoney(acct.nav) : "—"}</TopMetric>
        <TopMetric label="P&L (Day)" positive={dayPnl == null ? undefined : dayPnl >= 0}>
          {dayPnl == null ? "—" : fmtSigned(dayPnl)}
        </TopMetric>
        <TopMetric label="Win Rate">{winRate == null ? "—" : fmtPct(winRate, 1)}</TopMetric>
        <TopMetric label="Drawdown" positive={drawdownPct == null ? undefined : drawdownPct < 5}>
          {drawdownPct == null ? "—" : fmtPct(drawdownPct, 2)}
        </TopMetric>
        <TopMetric label="Leverage">{leverage == null ? "—" : `${fmtNum(leverage, 1)}x`}</TopMetric>
        <TopMetric label="Margin" title={marginUsedPct != null ? `${fmtNum(marginUsedPct, 1)}% used` : undefined}>
          {acct ? fmtMoney(acct.margin_available) : "—"}
        </TopMetric>
      </div>

      <div className="order-2 ml-auto flex min-w-0 items-center gap-2 xl:order-none xl:ml-0">
        <div className="flex min-w-0 items-center gap-1.5 pl-1" title="SSE stream to data layer">
          <span className={`h-1.5 w-1.5 rounded-full ${connected ? "live-dot" : ""}`} style={{ background: connected ? "#34e5a1" : "#5a6677" }} />
          <span className="min-w-0 font-mono text-[10px] text-faint">
            {connected ? "live" : "offline"}
          </span>
        </div>
        <button
          onClick={async () => {
            await fetch("/auth/logout", { method: "POST" });
            window.location.href = "/login";
          }}
          title="Lock / sign out"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-pos/45 bg-pos/10 font-mono text-[12px] font-semibold text-pos hover:bg-pos/20"
        >
          AX
        </button>
        <ChevronDown className="shrink-0 text-faint" />
      </div>
    </header>
  );
}
