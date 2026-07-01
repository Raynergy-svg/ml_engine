"use client";
import { useEffect, useMemo, useRef, useState } from "react";

export interface CommandItem {
  id: string;
  label: string;
  hint?: string;
  group: string;
  run: () => void;
}

export function CommandPalette({
  open, onClose, items,
}: { open: boolean; onClose: () => void; items: CommandItem[] }) {
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) { setQ(""); setIdx(0); setTimeout(() => inputRef.current?.focus(), 10); }
  }, [open]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((i) => i.label.toLowerCase().includes(needle) || i.group.toLowerCase().includes(needle));
  }, [q, items]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") { onClose(); return; }
      if (e.key === "ArrowDown") { e.preventDefault(); setIdx((i) => Math.min(i + 1, filtered.length - 1)); }
      if (e.key === "ArrowUp") { e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)); }
      if (e.key === "Enter") {
        e.preventDefault();
        const item = filtered[idx];
        if (item) { onClose(); item.run(); }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, filtered, idx, onClose]);

  if (!open) return null;

  let lastGroup = "";
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[12vh]" onClick={onClose}>
      <div
        className="card w-full max-w-lg overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b px-3 py-2.5 hairline">
          <span className="font-mono text-[13px] text-faint">/</span>
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => { setQ(e.target.value); setIdx(0); }}
            placeholder="Type a command or search…"
            className="w-full bg-transparent font-mono text-[13px] text-text outline-none placeholder:text-faint"
          />
          <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-faint hairline">ESC</span>
        </div>
        <div className="scroll-thin max-h-[50vh] overflow-auto py-1.5">
          {filtered.length === 0 && (
            <div className="px-3 py-6 text-center font-mono text-[12px] text-faint">No matching commands</div>
          )}
          {filtered.map((item, i) => {
            const showGroup = item.group !== lastGroup;
            lastGroup = item.group;
            return (
              <div key={item.id}>
                {showGroup && <div className="px-3 pb-1 pt-2 font-mono text-[10px] uppercase tracking-wide text-faint">{item.group}</div>}
                <button
                  onMouseEnter={() => setIdx(i)}
                  onClick={() => { onClose(); item.run(); }}
                  className={`flex w-full items-center justify-between px-3 py-2 text-left font-mono text-[13px] ${
                    i === idx ? "bg-surface2 text-text" : "text-dim"
                  }`}
                >
                  <span>{item.label}</span>
                  {item.hint && <span className="text-[10.5px] text-faint">{item.hint}</span>}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
