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

export async function runAxiomOperator(): Promise<{ status: number; ok: boolean; data: AxiomOperatorRunResult | Record<string, unknown> }> {
  let res: Response;
  try {
    res = await fetch("/api/axiom_operator/run", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-axiom-actor": actorId(),
      },
      body: "{}",
    });
  } catch {
    return { status: 0, ok: false, data: { error: "network_error" } };
  }
  const data = (await res.json().catch(() => ({}))) as AxiomOperatorRunResult | Record<string, unknown>;
  return { status: res.status, ok: res.ok, data };
}
