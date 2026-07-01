import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE, createSessionToken, checkPassword, authReady } from "@/lib/auth";
import { clientIp, rateLimit } from "@/lib/ratelimit";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  // Fail-closed: unconfigured OR weak/dev auth in production refuses login.
  if (!authReady()) {
    return NextResponse.json({ error: "auth_not_ready" }, { status: 503 });
  }
  // Rate-limit BEFORE checking the password — caps guessing on the one open route.
  const rl = rateLimit(clientIp(req.headers));
  if (!rl.allowed) {
    return NextResponse.json(
      { error: "too_many_attempts", retry_after_sec: rl.retryAfterSec },
      { status: 429, headers: { "retry-after": String(rl.retryAfterSec) } },
    );
  }
  let password = "";
  try {
    const body = await req.json();
    password = typeof body?.password === "string" ? body.password : "";
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  if (!checkPassword(password)) {
    return NextResponse.json({ error: "invalid_credentials" }, { status: 401 });
  }
  const token = await createSessionToken();
  const res = NextResponse.json({ ok: true });
  res.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    // HTTPS-only in production (Tailscale serve provides TLS); relaxed for local http dev.
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 7 * 24 * 60 * 60,
  });
  return res;
}
