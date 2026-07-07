"use client";

import type { AxiomOperatorRunResult } from "./types";

function actorId(): string {
  if (typeof window === "undefined") return "dashboard-server";
  const key = "axiom_actor_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const generated = `dashboard-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  window.localStorage.setItem(key, generated);
  return generated;
}

// Backend epoch can run up to AXIOM_OPERATOR_TIMEOUT_SECONDS (120s default).
// Without a client-side timeout, a hung network path leaves the caller
// awaiting the promise indefinitely with no way to recover or surface an
// error — 130s gives the server's own timeout a chance to fire first.
const RUN_TIMEOUT_MS = 130_000;

export async function runAxiomOperator(): Promise<{ status: number; ok: boolean; data: AxiomOperatorRunResult | Record<string, unknown> }> {
  let res: Response;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), RUN_TIMEOUT_MS);
  try {
    res = await fetch("/api/axiom_operator/run", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-axiom-actor": actorId(),
      },
      body: "{}",
      signal: controller.signal,
    });
  } catch (err) {
    const isAbort = err instanceof DOMException && err.name === "AbortError";
    return { status: 0, ok: false, data: { error: isAbort ? "client_timeout" : "network_error" } };
  } finally {
    clearTimeout(timeout);
  }
  const data = (await res.json().catch(() => ({}))) as AxiomOperatorRunResult | Record<string, unknown>;
  return { status: res.status, ok: res.ok, data };
}
