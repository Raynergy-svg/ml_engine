# AXIOM — secure remote access (Phase 1: authed read-only view)

Goal: view AXIOM **from your phone**, securely, without exposing anything to the public
internet. Two layers of defense:

1. **Tailscale** — a private mesh between your Mac and phone. The dashboard is reachable
   only by devices on *your* tailnet, over HTTPS. Nothing is public.
2. **AXIOM login** — a single-user access key (session cookie). Even on your tailnet, no
   unauthenticated request reaches any data.

```
phone (your tailnet) ──HTTPS──> Tailscale ──> Next.js :3000 (authed)
                                                  │  server-side proxy (loopback only)
                                                  ▼
                                          FastAPI :8888  ──read-only──> OANDA fxPractice
```
FastAPI binds `127.0.0.1` and is **never** tunneled. The browser only ever talks to the
authed Next origin; the OANDA token never leaves the Mac.

---

## Step 0 — set the secrets (one time; never commit them)

Edit `dashboard/web/.env.local` (gitignored) and replace the dev values:

```bash
# a strong random signing key for sessions:
openssl rand -hex 32          # paste as AXIOM_AUTH_SECRET
```
```ini
AXIOM_AUTH_SECRET=<the 64-char hex from openssl>
AXIOM_PASSWORD=<a long passphrase you'll type on the phone>
AXIOM_API_ORIGIN=http://127.0.0.1:8888
```
The OANDA **practice** token stays where it already is — repo-root `.env.local`
(`OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID`). Don't move or commit it.

## Step 1 — run the two local servers (on the Mac)

```bash
# 1) data layer (read-only), loopback only
cd /Users/buddy/Documents/ml_engine
python -m uvicorn dashboard.server.app:app --host 127.0.0.1 --port 8888

# 2) the web app — production build, bound to loopback (Tailscale fronts it)
cd dashboard/web
npm run build
npx next start -H 127.0.0.1 -p 3000
```
Local sanity check: open `http://localhost:3000` → you should hit the **AXIOM login**.

## Step 2 — Tailscale (operator-side; like the broker token, this part is yours)

**Install + sign in (both devices, same account/tailnet):**
- **Mac:** install Tailscale (`brew install --cask tailscale` or the Mac App Store), open it,
  sign in. Ensure the CLI is available (`tailscale version`; App Store build: enable the CLI
  from the app's menu, or use `/Applications/Tailscale.app/Contents/MacOS/Tailscale`).
- **iPhone:** install **Tailscale** from the App Store, sign into the **same** account, toggle it ON.

**Serve the dashboard privately over HTTPS (Mac):**
```bash
tailscale serve --bg 3000        # serves localhost:3000 over HTTPS on your tailnet
tailscale serve status           # prints the URL, e.g. https://<your-mac>.<tailnet>.ts.net
```
`serve` is **private to your tailnet** and provisions a valid HTTPS cert automatically.

**On the phone:** with Tailscale ON, open the `https://<your-mac>.<tailnet>.ts.net` URL →
AXIOM login → enter your `AXIOM_PASSWORD`. You now have the live read-only terminal. Tap
**lock** (top-right) to sign out.

**Stop sharing when you want:**
```bash
tailscale serve reset            # remove the serve config
```

> ⚠️ Do **NOT** use `tailscale funnel` — that exposes it to the *public* internet. We use
> `serve` (tailnet-private) on purpose.

---

## Security properties (Phase 1)
- No unauthenticated access to any endpoint (enforced in `proxy.ts`, fail-closed if the
  secrets are unset).
- FastAPI is loopback-only; reachable solely via the authed Next proxy (GET-only).
- Session cookie is `httpOnly` + `secure` (HTTPS) + `sameSite=lax`; signed (HMAC), 7-day expiry.
- The OANDA token and bot internals never reach the browser.
- **Read-only** — there is no write/control path in Phase 1. (Phase 2 control is built
  disabled behind a flag and is NOT exposed; see `dashboard/CONTROL_DESIGN.md`.)

## Alternative tunnel (if you prefer): Cloudflare Tunnel + Access
`cloudflared tunnel` to `127.0.0.1:3000` + a Cloudflare **Access** policy (email OTP / SSO)
in front. More moving parts (a domain + Cloudflare account) than Tailscale; only worth it if
you already live in Cloudflare. Tailscale `serve` is the simpler, fully-private default.
