"use client";
import { useEffect, useRef, useState } from "react";

// Same-origin: the browser talks ONLY to the authed Next server, which proxies
// `/api/*` server-side to the loopback FastAPI. No direct cross-origin call, no API
// token in the client, and the session cookie rides along automatically (incl. SSE).
export const API_BASE = "";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

// A 404 means the route doesn't exist on the server (e.g. /api/control/* while the
// control router is unmounted). Re-polling it at full cadence just spams the console
// and the server log with 404s, so back off hard — the route can only reappear after
// a server restart, and a 60s retry keeps the UI self-healing without the noise.
const NOT_FOUND_BACKOFF_MS = 60_000;

/**
 * Poll a read-only endpoint on an interval. Returns {data, error, loading}.
 * Honest by design: never invents data — `data` is null until the first real
 * response, and `error` is surfaced so the UI can show a "not connected" state.
 */
export function usePoll<T>(path: string, intervalMs: number): {
  data: T | null;
  error: string | null;
  loading: boolean;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let timer: ReturnType<typeof setTimeout>;
    const run = async () => {
      let delay = intervalMs;
      try {
        const d = await apiGet<T>(path);
        if (alive.current) { setData(d); setError(null); }
      } catch (e) {
        const msg = (e as Error).message;
        if (alive.current) setError(msg);
        // Route absent on the server (e.g. control router unmounted): back off hard
        // instead of hammering 404s at the poll cadence. See NOT_FOUND_BACKOFF_MS.
        if (msg.endsWith("-> 404")) delay = Math.max(intervalMs, NOT_FOUND_BACKOFF_MS);
      } finally {
        if (alive.current) { setLoading(false); timer = setTimeout(run, delay); }
      }
    };
    run();
    return () => { alive.current = false; clearTimeout(timer); };
  }, [path, intervalMs]);

  return { data, error, loading };
}
