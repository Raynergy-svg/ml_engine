// In-memory per-IP fixed-window rate limiter for the login endpoint.
//
// AXIOM runs as a SINGLE instance (one web container / one Fly machine), so an
// in-process limiter is sufficient and needs no external store. It is the documented
// remaining auth gate: it caps password-guessing on the one unauthenticated route.
//
// If AXIOM is ever scaled to multiple web instances, replace this with a shared store
// (Redis/Upstash) — an in-memory limiter does not coordinate across instances.

interface Window {
  count: number;
  resetAt: number; // epoch ms
}

const WINDOW_MS = 15 * 60 * 1000; // 15 minutes
const MAX_ATTEMPTS = 10; // per IP per window
const buckets = new Map<string, Window>();
let lastSweep = 0;

/** Best-effort client IP from proxy headers (Fly/most hosts set x-forwarded-for). */
export function clientIp(headers: Headers): string {
  const xff = headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  return headers.get("x-real-ip") || "unknown";
}

/**
 * Record an attempt for `ip`. Returns whether it is allowed plus retry metadata.
 * Counts ALL attempts (success or fail) — a correct password still consumes a slot,
 * which is fine for a single operator and prevents enumeration via timing of the cap.
 */
export function rateLimit(ip: string, now: number = Date.now()): {
  allowed: boolean;
  remaining: number;
  retryAfterSec: number;
} {
  // Opportunistic sweep of expired buckets so the map can't grow unbounded.
  if (now - lastSweep > WINDOW_MS) {
    for (const [k, w] of buckets) if (w.resetAt <= now) buckets.delete(k);
    lastSweep = now;
  }

  let w = buckets.get(ip);
  if (!w || w.resetAt <= now) {
    w = { count: 0, resetAt: now + WINDOW_MS };
    buckets.set(ip, w);
  }
  w.count += 1;
  const allowed = w.count <= MAX_ATTEMPTS;
  return {
    allowed,
    remaining: Math.max(0, MAX_ATTEMPTS - w.count),
    retryAfterSec: allowed ? 0 : Math.ceil((w.resetAt - now) / 1000),
  };
}

export const RATE_LIMIT = { WINDOW_MS, MAX_ATTEMPTS };
