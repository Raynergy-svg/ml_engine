// Single-user session auth for AXIOM. Edge-compatible (Web Crypto only — no node:crypto)
// so it runs in both middleware (edge) and route handlers (node).
//
// Session token = `${exp}.${HMAC_SHA256(secret, exp)}` (base64url sig). Stateless,
// signed, expiring. The login secret (AXIOM_PASSWORD) and signing key
// (AXIOM_AUTH_SECRET) live ONLY in server env (gitignored) — never sent to the client.
//
// FAIL-CLOSED: if either secret is unconfigured, auth denies everything. Unconfigured
// is locked, never open.

export const SESSION_COOKIE = "axiom_session";
const TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

const enc = new TextEncoder();

function b64url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function hmac(secret: string, msg: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  return b64url(await crypto.subtle.sign("HMAC", key, enc.encode(msg)));
}

/** Constant-time string compare (avoids timing leaks on the signature/password). */
export function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export function authConfigured(): boolean {
  return !!(process.env.AXIOM_AUTH_SECRET && process.env.AXIOM_PASSWORD);
}

// Known dev/weak values that must NEVER guard a production deployment.
const WEAK_PASSWORDS = new Set([
  "axiom-dev-login", "password", "changeme", "admin", "secret", "test", "axiom",
]);

/**
 * Production-grade auth config check. In production we refuse to run on the dev
 * password or a short secret — so a cloud deployment with control enabled CANNOT
 * boot behind weak/default auth (the one forbidden combination). In development the
 * dev password is allowed for local convenience.
 *
 * Returns {ok:true} when safe to serve, else {ok:false, reason} (fail-closed).
 */
export function authProductionSafe(): { ok: boolean; reason?: string } {
  if (!authConfigured()) return { ok: false, reason: "AXIOM_AUTH_SECRET + AXIOM_PASSWORD not set" };
  if (process.env.NODE_ENV !== "production") return { ok: true }; // dev: convenience
  const pw = process.env.AXIOM_PASSWORD || "";
  const secret = process.env.AXIOM_AUTH_SECRET || "";
  if (WEAK_PASSWORDS.has(pw.toLowerCase())) return { ok: false, reason: "AXIOM_PASSWORD is a known dev/weak value" };
  if (pw.length < 12) return { ok: false, reason: "AXIOM_PASSWORD too short (need >= 12 chars in production)" };
  if (secret.length < 32) return { ok: false, reason: "AXIOM_AUTH_SECRET too short (need >= 32 chars in production)" };
  return { ok: true };
}

/** True only when auth is configured AND production-safe — the gate to serve anything. */
export function authReady(): boolean {
  return authProductionSafe().ok;
}

export async function createSessionToken(): Promise<string> {
  const secret = process.env.AXIOM_AUTH_SECRET;
  if (!secret) throw new Error("AXIOM_AUTH_SECRET not configured");
  const exp = String(Date.now() + TTL_MS);
  return `${exp}.${await hmac(secret, exp)}`;
}

export async function verifySessionToken(token: string | undefined | null): Promise<boolean> {
  const secret = process.env.AXIOM_AUTH_SECRET;
  if (!secret || !token) return false;             // fail-closed
  const dot = token.indexOf(".");
  if (dot < 0) return false;
  const exp = token.slice(0, dot);
  const sig = token.slice(dot + 1);
  const expMs = Number(exp);
  if (!Number.isFinite(expMs) || expMs < Date.now()) return false; // expired/garbage
  const expected = await hmac(secret, exp);
  return timingSafeEqual(sig, expected);
}

/** Verify a submitted login password against AXIOM_PASSWORD (constant-time). */
export function checkPassword(submitted: string): boolean {
  const expected = process.env.AXIOM_PASSWORD;
  if (!expected) return false;                      // fail-closed
  return timingSafeEqual(submitted, expected);
}
