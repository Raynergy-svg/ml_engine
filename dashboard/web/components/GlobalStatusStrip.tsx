"use client";
import { useState } from "react";
import { useStream } from "@/lib/stream";
import { usePoll } from "@/lib/api";
import { control } from "@/lib/control";
import type { ControlState } from "@/lib/types";
import { StatusDot } from "./ui";

const LEVERAGE_PRESETS = [
  { label: "Conservative", leverage: 1 },
  { label: "Normal", leverage: 3 },
  { label: "Elevated", leverage: 7 },
  { label: "Maximum", leverage: 15 },
];

function Field({
  label, children, wide = false, onClick, chevron = true,
}: { label: string; children: React.ReactNode; wide?: boolean; onClick?: () => void; chevron?: boolean }) {
  const Comp = onClick ? "button" : "div";
  return (
    <Comp
      onClick={onClick}
      className={`min-h-[58px] rounded-md border bg-surface/70 px-3 py-2 text-left hairline ${wide ? "min-w-[260px] flex-1" : "min-w-[170px]"} ${onClick ? "hover:bg-surface2/60" : ""}`}
    >
      <span className="block font-mono text-[10px] uppercase tracking-wide text-faint">{label}</span>
      <div className="mt-1 flex items-center justify-between gap-4 font-mono text-[13px] text-text">
        {children}
        {chevron && <span className="text-faint">⌄</span>}
      </div>
    </Comp>
  );
}

export function GlobalStatusStrip({ onOpenPalette }: { onOpenPalette?: () => void }) {
  const { payload } = useStream();
  const { data: controlState } = usePoll<ControlState>("/api/control/state", 5000);
  const halted = payload?.status?.halted ?? null;
  const loops = controlState?.loops;
  const automationOn = !!(loops?.trend?.running || loops?.tier7?.running);
  const [openMenu, setOpenMenu] = useState<"strategy" | "risk" | "automation" | null>(null);
  const [busy, setBusy] = useState(false);

  async function doHalt() {
    if (halted) return;
    if (!window.confirm("Halt trading now?\n\nThis is fail-safe — it only stops new orders on the practice account.")) return;
    const res = await control("halt");
    if (!res.ok) window.alert(`Halt request denied: ${String(res.data.detail ?? res.data.error ?? res.status)}`);
  }

  async function doFlatten() {
    // FLATTEN closes every open position — a real, hard-to-reverse action. A single
    // confirm() is not enough for this one: require the operator to type the word.
    const typed = window.prompt(
      'Type FLATTEN to close ALL open positions on the practice account.\nThis cannot be undone.',
    );
    if (typed !== "FLATTEN") return;
    setBusy(true);
    const res = await control("flatten");
    setBusy(false);
    if (!res.ok) window.alert(`Flatten denied: ${String(res.data.detail ?? res.data.error ?? res.status)}`);
    else window.alert(`Flatten result: ${String(res.data.result)} (${String(res.data.count ?? 0)} positions)`);
  }

  async function setLeverage(x: number) {
    setOpenMenu(null);
    if (!window.confirm(`Set gross leverage to ${x}x?`)) return;
    const res = await control("set_gross_leverage", { gross_leverage: x });
    if (!res.ok) window.alert(`Leverage change denied: ${String(res.data.detail ?? res.data.error ?? res.status)}`);
  }

  async function toggleAutomation() {
    setOpenMenu(null);
    const turnOn = !automationOn;
    if (!window.confirm(`${turnOn ? "Start" : "Stop"} both automation loops (trend + Tier 7)?`)) return;
    const action = turnOn ? "start_loop" : "stop_loop";
    await control(action, { loop: "trend" });
    await control(action, { loop: "tier7" });
  }

  return (
    <div className="relative mx-auto flex w-full max-w-[1680px] flex-wrap items-center gap-2 border-b px-3 py-2 hairline sm:px-4">
      <button
        className="grid h-[58px] w-[58px] shrink-0 place-items-center rounded-md border bg-surface/75 text-dim hairline hover:text-text"
        title="Menu"
      >
        <span className="text-[28px] leading-none">≡</span>
      </button>

      <button
        onClick={onOpenPalette}
        className="min-h-[58px] min-w-[260px] flex-1 rounded-md border bg-surface/70 px-3 py-2 text-left hairline hover:bg-surface2/60"
      >
        <div className="flex items-center justify-between gap-3">
          <span className="truncate font-mono text-[13px] text-dim">Type a command or / for menu...</span>
          <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-faint hairline">⌘K</span>
        </div>
      </button>

      <div className="relative">
        <Field label="Strategy" onClick={() => setOpenMenu(openMenu === "strategy" ? null : "strategy")}>
          <span className="font-semibold" title="Long-or-flat SMA(100) trend rule (src/equity/oanda_trend.py)">trend_sma100</span>
        </Field>
        {openMenu === "strategy" && (
          <div className="card absolute left-0 top-[64px] z-20 min-w-[220px] p-2">
            <div className="rounded px-2 py-1.5 font-mono text-[12px] text-text">trend_sma100</div>
            <div className="px-2 py-1 font-mono text-[10.5px] text-faint">only active strategy</div>
          </div>
        )}
      </div>

      <div className="relative">
        <Field label="Risk Mode" onClick={() => setOpenMenu(openMenu === "risk" ? null : "risk")}>
          <span className="flex items-center gap-2">
            <StatusDot color="#2bd17e" />
            {controlState?.gross_leverage != null ? `${controlState.gross_leverage}x` : "—"}
          </span>
        </Field>
        {openMenu === "risk" && (
          <div className="card absolute left-0 top-[64px] z-20 min-w-[200px] p-1.5">
            {LEVERAGE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setLeverage(p.leverage)}
                className="flex w-full items-center justify-between rounded px-2.5 py-1.5 text-left font-mono text-[12px] text-dim hover:bg-surface2 hover:text-text"
              >
                <span>{p.label}</span>
                <span className="text-faint">{p.leverage}x</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="relative">
        <Field label="Automation" onClick={() => setOpenMenu(openMenu === "automation" ? null : "automation")}>
          <span className="flex items-center gap-2">
            <StatusDot color={automationOn ? "#2bd17e" : "#5a6677"} />
            {automationOn ? "Enabled" : "Paused"}
          </span>
        </Field>
        {openMenu === "automation" && (
          <div className="card absolute left-0 top-[64px] z-20 min-w-[220px] p-1.5">
            <button onClick={toggleAutomation} className="w-full rounded px-2.5 py-1.5 text-left font-mono text-[12px] text-dim hover:bg-surface2 hover:text-text">
              {automationOn ? "Stop trend + Tier 7 loops" : "Start trend + Tier 7 loops"}
            </button>
            <div className="px-2.5 py-1 font-mono text-[10px] text-faint">
              trend: {loops?.trend?.running ? "running" : "offline"} · tier7: {loops?.tier7?.running ? "running" : "offline"}
            </div>
          </div>
        )}
      </div>

      <button
        onClick={doHalt}
        disabled={halted === true}
        className="rounded-md border border-warn/65 px-5 py-2 font-mono text-[12px] font-semibold text-warn hover:bg-warn/10 disabled:opacity-40"
        title="Halt trading (audited, fail-safe)"
      >
        HALT
      </button>
      <button
        onClick={doFlatten}
        disabled={busy}
        className="rounded-md border px-5 py-2 font-mono text-[12px] font-semibold text-text hairline hover:bg-surface2 disabled:opacity-40"
        title="Close ALL open positions (audited; requires typed confirmation)"
      >
        FLATTEN
      </button>
      <button
        onClick={() => window.location.reload()}
        className="rounded-md bg-pos px-5 py-2 font-mono text-[12px] font-semibold text-base hover:brightness-110"
        title="Reload the dashboard view (UI-only, no bot action)"
      >
        NEW SESSION
      </button>
      <button className="px-1 font-mono text-[22px] text-faint hover:text-text" title="More">⋮</button>

      {openMenu && <div className="fixed inset-0 z-10" onClick={() => setOpenMenu(null)} />}
    </div>
  );
}
