import { NextRequest, NextResponse } from "next/server";

// Authed, server-side, READ-ONLY proxy: browser → (authed) Next → FastAPI on loopback.
// FastAPI is never exposed to the tunnel/network; only this GET proxy reaches it, and
// only after middleware has validated the session. GET-only by construction — there is
// no POST/PUT/DELETE path here, so the remote view cannot mutate anything.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const API_ORIGIN = process.env.AXIOM_API_ORIGIN || "http://127.0.0.1:8888";

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const sub = (path || []).join("/");
  const search = req.nextUrl.search || "";
  const target = `${API_ORIGIN}/api/${sub}${search}`;

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: "GET",
      headers: { accept: req.headers.get("accept") || "application/json" },
      cache: "no-store",
      // @ts-expect-error - Node fetch streaming flag
      duplex: "half",
    });
  } catch {
    return NextResponse.json({ error: "upstream_unreachable" }, { status: 502 });
  }

  // Pass the body straight through (works for JSON and for the SSE stream).
  const headers = new Headers();
  const ct = upstream.headers.get("content-type");
  if (ct) headers.set("content-type", ct);
  headers.set("cache-control", "no-store");
  if (ct?.includes("text/event-stream")) headers.set("x-accel-buffering", "no");
  return new Response(upstream.body, { status: upstream.status, headers });
}
