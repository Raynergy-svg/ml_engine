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
    const es = new EventSource(`${API_BASE}/api/stream`);
    esRef.current = es;
    es.onopen = () => setConnected(true);
    es.onmessage = (ev) => {
      try {
        setPayload(JSON.parse(ev.data) as StreamPayload);
        setConnected(true);
      } catch {
        /* ignore malformed frame */
      }
    };
    es.onerror = () => setConnected(false); // EventSource retries automatically
    return () => es.close();
  }, []);

  return <StreamCtx.Provider value={{ payload, connected }}>{children}</StreamCtx.Provider>;
}

export function useStream(): StreamState {
  return useContext(StreamCtx);
}
