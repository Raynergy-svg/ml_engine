"use client";
import { useStream } from "@/lib/stream";
import { control } from "@/lib/control";
import { StatusDot } from "./ui";

function Field({ label, children, wide = false }: { label: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`min-h-[58px] rounded-md border bg-surface/70 px-3 py-2 hairline ${wide ? "min-w-[260px] flex-1" : "min-w-[170px]"}`}>
      <span className="block font-mono text-[10px] uppercase tracking-wide text-faint">{label}</span>
      <div className="mt-1 font-mono text-[13px] text-text">{children}</div>
    </div>
  );
}

export function GlobalStatusStrip({ onOpenControl }: { onOpenControl?: () => void }) {
  const { payload } = useStream();
  const halted = payload?.status?.halted ?? null;
  const automation = payload?.status?.running === true;

  async function doHalt() {
    if (halted) return;
    if (!window.confirm("Halt trading now?\n\nThis is fail-safe — it only stops new orders on the practice account.")) return;
    const res = await control("halt");
    if (!res.ok) window.alert(`Halt request denied: ${String(res.data.detail ?? res.data.error ?? res.status)}`);
  }

  return (
    <div className="mx-auto flex w-full max-w-[1680px] flex-wrap items-center gap-2 border-b px-3 py-2 hairline sm:px-4">
      <button
        className="grid h-[58px] w-[58px] shrink-0 place-items-center rounded-md border bg-surface/75 text-dim hairline hover:text-text"
        title="Menu"
      >
        <span className="text-[28px] leading-none">≡</span>
      </button>

      <Field label="Command" wide>
        <div className="flex items-center justify-between gap-3">
          <span className="truncate text-dim">Type a command or / for menu...</span>
          <span className="rounded border px-1.5 py-0.5 text-[10px] text-faint hairline">⌘K</span>
        </div>
      </Field>

      <Field label="Strategy">
        <div className="flex items-center justify-between gap-4">
          <span className="font-semibold" title="Long-or-flat SMA(100) trend rule (src/equity/oanda_trend.py)">trend_sma100</span>
        </div>
      </Field>

      <Field label="Environment">
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-2 text-pos">
            <StatusDot color="#2bd17e" />
            Practice
          </span>
        </div>
      </Field>

      <Field label="Automation">
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-2">
            <StatusDot color={automation ? "#2bd17e" : "#5a6677"} />
            {automation ? "Enabled" : "Paused"}
          </span>
        </div>
      </Field>

      <div className="ml-auto flex min-h-[58px] flex-wrap items-center gap-2 rounded-md border bg-surface/65 px-3 py-2 hairline">
        <button
          onClick={doHalt}
          disabled={halted === true}
          className="rounded-md border border-warn/65 px-5 py-2 font-mono text-[12px] font-semibold text-warn hover:bg-warn/10 disabled:opacity-40"
          title="Halt trading (audited, fail-safe)"
        >
          HALT
        </button>
        <button
          onClick={onOpenControl}
          className="rounded-md border px-5 py-2 font-mono text-[12px] font-semibold text-text hairline hover:bg-surface2"
          title="Flattening positions is not yet a reviewed control action — opens the audited Control panel"
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
      </div>
    </div>
  );
}
