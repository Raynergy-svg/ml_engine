"use client";
import { useEffect, useRef, useState } from "react";

export const API_BASE =
  process.env.NEXT_PUBLIC_AXIOM_API_URL || "http://localhost:8888";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

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
      try {
        const d = await apiGet<T>(path);
        if (alive.current) { setData(d); setError(null); }
      } catch (e) {
        if (alive.current) setError((e as Error).message);
      } finally {
        if (alive.current) { setLoading(false); timer = setTimeout(run, intervalMs); }
      }
    };
    run();
    return () => { alive.current = false; clearTimeout(timer); };
  }, [path, intervalMs]);

  return { data, error, loading };
}
