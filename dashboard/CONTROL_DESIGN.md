# AXIOM Phase 2 — bounded operator control (ENABLED 2026-06-30)

> **STATUS: ENABLED (localhost).** Per the 2026-06-30 mandate, AXIOM is the primary
> Buddy client and operates the practice bot. The control router is mounted when
> `AXIOM_CONTROL_ENABLED=1` (still default OFF → 404 if unset). All actions are
> functional; the separate safety verifier returned **PASS** (all 7 escape questions →
> NO; see verdict below). The Control tab in the UI drives these via the authed proxy.
>
> ⚠️ **AUTH BEFORE REMOTE.** Control is safe on **localhost** (loopback FastAPI + authed
> Next proxy). It is **NOT** safe to expose remotely (Tailscale/phone) until strong auth
> is in front — a control-enabled endpoint reachable without auth would let anyone
> halt/start/leverage the bot. Keep `tailscale serve` + a strong `AXIOM_PASSWORD` (and
> rotate `AXIOM_AUTH_SECRET`) before any remote use. This round is localhost-only.

## Proposed action set (CONFIRM before enabling)
All POST, **auth-gated** (Phase-1 session) + **explicit per-action confirm** + **audit-logged**:

| Action | Effect | Risk | Notes |
|---|---|---|---|
| `halt` | `StateEngine.set_halted(True)` | none (fail-safe) | always allowed; stops new trades |
| `unhalt` | `StateEngine.set_halted(False)` | medium | re-asserts `env==practice` + rails intact first |
| `set_gross_leverage` | set the OANDA gross-exposure dial/cap | medium | **clamped to [0, 15]**; out-of-range rejected |
| `start_loop` / `stop_loop` | start/stop a **named** loop (`trend` \| `tier7`) | medium | whitelist only; **no arbitrary command exec** |

Nothing else. No threshold edits, no gate edits, no model promotion, no env/account changes.

## Server-side IMMUTABLES (enforced at the API layer, regardless of any request)
These are structural — not UI-hidden. A crafted/malicious request cannot reach them.

- **I1 — Practice only.** `control_safety.assert_practice()` re-derives `oanda_environment`
  from a fresh `ScannerConfig()` on **every** action and refuses unless `"practice"`
  (config default `config.py:742`). The control layer accepts **no** `environment` / `url` /
  `account` parameter at all — there is no field to set live. Live is unreachable by design.
- **I2 — No real-money endpoint.** The control layer never constructs a URL and only ever
  uses the practice-pinned `OandaPracticeClient` (base URL `api-fxpractice`, no live constant
  exists in it). It cannot talk to `api-fxtrade`.
- **I3 — No Hard-NO relaxation.** The only mutations are the four bounded actions. Leverage
  is clamped to the 15× cap server-side. `unhalt` is refused unless env is practice. There is
  no code path to edit gates, thresholds, the ship gate, or `oanda_environment`.
- **I4 — No promotion.** There is **no promote action**. Ship-gate promotion stays entirely
  in the bot's gated training path; AXIOM physically lacks the capability to promote a model
  (gate-failing or otherwise).

## Enforcement architecture
1. **Flag gate (mount-time):** `app.py` mounts the control router **only if**
   `AXIOM_CONTROL_ENABLED` is truthy. Default → not mounted → `404` on every control route.
2. **Auth gate:** reached only through the Phase-1 authed Next proxy (session cookie). The
   FastAPI control routes also require a confirm header.
3. **Confirm gate:** each request must carry `X-AXIOM-Confirm: <action>` matching the action
   (an explicit, per-action acknowledgement — not a blanket session grant).
4. **Guard gate (pre-effect):** `control_safety` runs BEFORE any mutation:
   `assert_practice()` → `validate_action()` → `clamp/whitelist params`. A guard failure →
   `403`, audited, no effect.
5. **Audit (always):** every attempt — allowed **and** denied — appends an atomic JSONL line
   to `trained_data/axiom/control_audit.jsonl`: `{ts, action, params, allowed, reason, result}`.

## Threat model — crafted requests must all resolve to "no breach"
| Crafted request | Outcome |
|---|---|
| `{"environment":"live"}` / `{"url":"...api-fxtrade..."}` | param ignored (not accepted); `assert_practice` still practice → **no live** |
| `{"action":"set_gross_leverage","x":999}` | clamped/rejected to ≤ 15 → **no over-leverage** |
| `{"action":"promote_model"}` / any unknown action | unknown → `404/403` → **no promotion** |
| `unhalt` while env somehow ≠ practice | refused by `assert_practice` → **stays halted** |
| Any control call with flag OFF | router unmounted → `404` → **no surface** |
| Any control call without session / confirm | `401` / `403` → **no unauth action** |

## Implementation status — ALL FUNCTIONAL (2026-06-30)
- `halt` — functional (StateEngine.set_halted(True)); fail-safe.
- `unhalt` — functional via `assert_unhalt_eligible`: refuses unless **practice** + **NAV
  drawdown < 20%** + **verify_gate GREEN**. Model age is surfaced but INFORMATIONAL (the live
  lane is the non-ML trend strategy; FX/ML is retired/L-016, so blocking on its permanent
  staleness would deadlock the unhalt→start-loop resume flow). Verified round-trip on practice.
- `set_gross_leverage` — functional: writes a clamped `[0,15]` override to
  `trained_data/axiom/control_overrides.json`; `run_oanda_trend.py` re-reads it each cycle
  and sizes the trend book to the dial, while OANDA-backed scanner execution re-reads it
  before each broker order and enforces it as a total gross-notional cap. Both paths
  re-cap/refuse values outside the 15× server-side bound.
- `start_loop` / `stop_loop` — functional via fixed-whitelist subprocess (`trend`→run_oanda_trend.py
  --loop, `tier7`→run_tier7_loop.py); stop by pgrep-needle + SIGTERM. No shell, no user input in argv.

## Separate safety verifier — VERDICT: PASS (re-run 2026-06-30, control FUNCTIONAL+ENABLED)
Independent Security Engineer re-audit of the functional control: all 7 escape questions → **NO** —
a crafted request (via the control endpoints OR the Next POST proxy) cannot reach live / real-money
/ api-fxtrade / relax a Hard-NO / promote a ship-gate-failing artifact / over-leverage >15× (clamped
on BOTH dashboard + loop) / arbitrary-exec (argv fixed by whitelist, no shell) / skip the unhalt gate
/ POST a non-control endpoint. No HIGH findings. MED-1 (unhalt freshness) addressed: lane snapshot
age now surfaced (informational, non-blocking by design). LOW items: audit-write failure surfacing,
no secret on the control router (relies on loopback + authed proxy — hence the auth-before-remote rule).

## Exposure checklist
- [x] Separate safety verifier: **PASS** (functional + enabled re-run).
- [x] Operator confirmed the action set (2026-06-30 mandate) + 15× cap.
- [x] `unhalt` eligibility gate wired (practice + drawdown + gates GREEN).
- [x] `set_gross_leverage` / loop-control effect mechanisms wired.
- [x] UI Control tab with per-action confirm + read-only audit-log view.
- [x] Enabled on **localhost** (`AXIOM_CONTROL_ENABLED=1`); halt→unhalt round-trip verified on practice.
- [ ] **REMOTE GATE (still required):** strong auth (Tailscale + strong `AXIOM_PASSWORD`) +
      login rate-limit before ANY remote/phone exposure. Localhost-only until then.
