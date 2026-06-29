import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE, verifySessionToken, authConfigured } from "@/lib/auth";

// Next 16 request proxy (formerly "middleware"). Auth gate for ALL routes: pages,
// the /api/* read-only proxy, and SSE. Everything except the login surface requires
// a valid session. Fail-closed when auth isn't configured.
const PUBLIC_PATHS = ["/login", "/auth/login"];

function isPublic(pathname: string): boolean {
  if (PUBLIC_PATHS.includes(pathname)) return true;
  return (
    pathname.startsWith("/_next/") ||
    pathname === "/favicon.ico" ||
    pathname.startsWith("/icon") ||
    pathname.startsWith("/apple-")
  );
}

export async function proxy(req: NextRequest) {
  const { pathname } = req.nextUrl;
  if (isPublic(pathname)) return NextResponse.next();

  const isApi = pathname.startsWith("/api/") || pathname.startsWith("/auth/");

  // Fail-closed: unconfigured auth denies everything (never default open).
  if (!authConfigured()) {
    if (isApi) {
      return NextResponse.json(
        { error: "auth_not_configured", detail: "Set AXIOM_AUTH_SECRET + AXIOM_PASSWORD in the server env." },
        { status: 503 },
      );
    }
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("err", "unconfigured");
    return NextResponse.redirect(url);
  }

  const ok = await verifySessionToken(req.cookies.get(SESSION_COOKIE)?.value);
  if (ok) return NextResponse.next();

  if (isApi) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.searchParams.set("next", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
