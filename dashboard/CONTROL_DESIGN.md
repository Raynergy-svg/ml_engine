# AXIOM Phase 2 — bounded remote control (DESIGN; built DISABLED)

> **STATUS: NOT EXPOSED.** This layer is built behind a feature flag that defaults
> **OFF** (`AXIOM_CONTROL_ENABLED` unset → the control router is not even mounted →
> every control route 404s). It will not be turned on until **(a)** the separate safety
> verifier returns PASS ("can a crafted request reach live / real-money / a Hard-NO? →
> NO") **and (b)** the operator confirms the action set below. The UI has no control
> affordance yet. This is AXIOM's first-ever write path — treated as a major safety surface.

## Proposed action set (CONFIRM before enabling)
All POST, **auth-gated** (Phase-1 session) + **explicit per-action confirm** + **audit-logged**:

| Action | Effect | Risk | Notes |
|---|---|---|---|
| `halt` | `StateEngine.set_halted(True)` | none (fail-safe) | always allowed; stops new trades |
| `unhalt` | `StateEngine.set_halted(False)` | medium | re-asserts `env==practice` + rails intact first |
| `set_gross_leverage` | set the trend lane's gross-leverage dial | medium | **clamped to [0, 15]**; out-of-range rejected |
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

## Exposure checklist (all required before flipping the flag)
- [ ] Separate safety verifier: **PASS** (no crafted request reaches live/real-money/Hard-NO).
- [ ] Operator confirms the action set above (and the leverage cap value).
- [ ] Wire UI with explicit confirm modals per action (and surface the audit log read-only).
- [ ] Then, and only then, set `AXIOM_CONTROL_ENABLED=1`.

Until every box is checked, control stays **disabled** and unexposed.
