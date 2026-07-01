"use client";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { API_BASE } from "./api";
import type { StreamPayload } from "./types";

interface StreamState {
  payload: StreamPayload | null;
  connected: boolean;
}

const StreamCtx = createContext<StreamState>({ payload: null, connected: false });

/**
 * Single SSE subscription to the read-only data layer. EventSource is GET-only
 * and auto-reconnects, so the dashboard cannot push anything to the bot — the
 * channel is structurally one-directional (server -> browser).
 */
export function StreamProvider({ children }: { children: React.ReactNode }) {
  const [payload, setPayload] = useState<StreamPayload | null>(null);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let stopped = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let backoff = 1000; // ms, capped — reset on a healthy message

    const connect = () => {
      if (stopped) return;
      const es = new EventSource(`${API_BASE}/api/stream`);
      esRef.current = es;
      es.onopen = () => setConnected(true);
      es.onmessage = (ev) => {
        try {
          setPayload(JSON.parse(ev.data) as StreamPayload);
          setConnected(true);
          backoff = 1000; // healthy frame → reset backoff
        } catch {
          /* ignore malformed frame */
        }
      };
      es.onerror = () => {
        setConnected(false);
        // EventSource auto-retries only on transient network errors. On a CLOSED
        // state (clean server close, proxy 502, backend restart) it does NOT —
        // so reconnect manually with capped backoff instead of showing a permanent
        // false "offline" until a page reload (F-M1).
        if (es.readyState === EventSource.CLOSED && !stopped) {
          es.close();
          if (retryTimer) clearTimeout(retryTimer);
          retryTimer = setTimeout(connect, backoff);
          backoff = Math.min(backoff * 2, 15000);
        }
      };
    };

    connect();
    return () => {
      stopped = true;
      if (retryTimer) clearTimeout(retryTimer);
      esRef.current?.close();
    };
  }, []);

  return <StreamCtx.Provider value={{ payload, connected }}>{children}</StreamCtx.Provider>;
}

export function useStream(): StreamState {
  return useContext(StreamCtx);
}
