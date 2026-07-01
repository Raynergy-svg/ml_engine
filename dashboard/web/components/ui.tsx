"use client";
import React from "react";

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`card min-w-0 ${className}`}>{children}</div>;
}

export function SectionTitle({
  children, right,
}: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-b px-4 pb-2.5 pt-3.5 hairline">
      <span className="eyebrow min-w-[140px] flex-1 truncate">{children}</span>
      {right && <div className="scroll-thin max-w-full overflow-x-auto">{right}</div>}
    </div>
  );
}

export function StatusDot({ color, pulse = false }: { color: string; pulse?: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${pulse ? "live-dot" : ""}`}
      style={{ background: color, boxShadow: `0 0 8px ${color}` }}
    />
  );
}

export function Badge({
  children, color = "#8b98a9", dot = false, pulse = false, title,
}: { children: React.ReactNode; color?: string; dot?: boolean; pulse?: boolean; title?: string }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[10.5px] font-semibold tnum"
      style={{ color, borderColor: `${color}55`, background: `${color}1f` }}
    >
      {dot && <StatusDot color={color} pulse={pulse} />}
      {children}
    </span>
  );
}

export function NotConnected({ label = "Not connected", hint }: { label?: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 py-10 text-center">
      <StatusDot color="#5a6677" />
      <span className="font-mono text-xs text-faint">{label}</span>
      {hint && <span className="max-w-xs text-[11px] text-faint/80">{hint}</span>}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-faint">
      <StatusDot color="#22d3ee" pulse />
      <span className="font-mono text-xs">{label}</span>
    </div>
  );
}
