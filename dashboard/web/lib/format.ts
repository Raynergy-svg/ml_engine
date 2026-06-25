// Number formatting for a trading terminal — always tabular, P&L always signed.

export function fmtNum(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

export function fmtMoney(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return "$" + fmtNum(n, dp);
}

/** Signed P&L string, e.g. +1,284.40 / −312.18 (true minus glyph). */
export function fmtSigned(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const s = fmtNum(Math.abs(n), dp);
  if (n > 0) return "+" + s;
  if (n < 0) return "−" + s;
  return s;
}

export function pnlClass(n: number | null | undefined): string {
  if (n === null || n === undefined || n === 0) return "text-dim";
  return n > 0 ? "text-pos" : "text-neg";
}

export function fmtPct(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return fmtNum(n, dp) + "%";
}

/** Price with pair-aware precision (JPY crosses show 3dp, others 5dp). */
export function fmtPrice(n: number | null | undefined, instrument?: string): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const dp = instrument && instrument.endsWith("_JPY") ? 3 : 5;
  return fmtNum(n, dp);
}

export function fmtUnits(n: number): string {
  return (n >= 0 ? "+" : "−") + Math.abs(n).toLocaleString("en-US");
}

export function shortTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-US", {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

export function ago(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return `${Math.round(seconds)}s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
  return `${(seconds / 3600).toFixed(1)}h ago`;
}

export function prettyPair(p: string): string {
  return p.replace("_", "/");
}
