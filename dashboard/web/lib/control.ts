"use client";

import type { ControlState } from "./types";

// Operator control client. Each call POSTs to the bounded control endpoint with the
// per-action confirm header the bot's control layer requires. The server re-enforces
// every immutable (practice-pin, leverage cap, eligibility) regardless of this client.
export interface ControlResult {
  status: number;
  ok: boolean;
  data: Record<string, unknown> & { state?: ControlState };
}

function actorId(): string {
  if (typeof window === "undefined") return "dashboard-server";
  const key = "axiom_actor_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const generated = `dashboard-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  window.localStorage.setItem(key, generated);
  return generated;
}

export async function control(
  action: string,
  params: Record<string, unknown> = {},
): Promise<ControlResult> {
  let res: Response;
  try {
    res = await fetch(`/api/control/${action}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-axiom-confirm": action,
        "x-axiom-actor": actorId(),
      },
      body: JSON.stringify({ params }),
    });
  } catch {
    return { status: 0, ok: false, data: { error: "network_error" } };
  }
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  return { status: res.status, ok: res.ok, data };
}
