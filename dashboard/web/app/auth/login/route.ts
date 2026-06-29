import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE, createSessionToken, checkPassword, authConfigured } from "@/lib/auth";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  if (!authConfigured()) {
    return NextResponse.json({ error: "auth_not_configured" }, { status: 503 });
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
