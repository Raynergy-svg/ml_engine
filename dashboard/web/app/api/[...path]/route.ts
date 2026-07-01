import { NextRequest, NextResponse } from "next/server";

// Authed, server-side proxy: browser → (authed) Next → FastAPI on loopback. FastAPI is
// never exposed to the tunnel/network; only this proxy reaches it, after the session is
// validated. GET = all read endpoints + SSE. POST = ONLY the bounded control endpoints
// (/api/control/*), which are themselves guarded server-side (practice-pinned immutables,
// per-action confirm, audit). Any other POST path is rejected here (405).
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

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const sub = (path || []).join("/");
  // Only the bounded control endpoints accept POST. Everything else is read-only.
  if (!sub.startsWith("control/")) {
    return NextResponse.json({ error: "method_not_allowed" }, { status: 405 });
  }
  const body = await req.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${API_ORIGIN}/api/${sub}`, {
      method: "POST",
      headers: {
        "content-type": req.headers.get("content-type") || "application/json",
        // forward the per-action confirm token the bot's control layer requires
        "x-axiom-confirm": req.headers.get("x-axiom-confirm") || "",
      },
      body,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ error: "upstream_unreachable" }, { status: 502 });
  }
  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") || "application/json", "cache-control": "no-store" },
  });
}
