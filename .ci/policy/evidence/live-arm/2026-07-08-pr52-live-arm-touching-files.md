# Live/arm review evidence

- **PR:** https://github.com/Raynergy-svg/ml_engine/pull/52
- **Files touched (in this PR's diff vs `main`) that trigger `policy.live_arm`:**
  - `src/agent_runtime/policy.py` — authored/modified in this PR (CodeRabbit review-response
    round 3): added a catch-all `except Exception` around the preflight block in
    `PolicyEngine.submit()` so a bug in a custom `spec.preflight` (or in
    `_preflight_practice_pin` itself) is still recorded to the audit trail before re-raising,
    matching the symmetric catch-all the `execute()` branch already had. Does **not** change
    which actions are OPERATIONAL/DEESCALATION/ESCALATION, does not add an `execute` path to
    any ESCALATION action, does not touch `_preflight_practice_pin`'s actual practice-pin logic.
  - `src/crypto/crypto_live_gate.py`, `src/crypto/crypto_carry_live_gate.py`,
    `src/equity/track_b_live_gate.py` — new files, authored elsewhere in this branch's history
    (not by this review-response session). Reviewed here: all three are thin re-export wrappers
    around `src.equity.live_gate.LiveGate`, scoping only the state/audit file paths per lane.
    None call `.arm()` themselves; every constructor starts disarmed
    (`LiveGate(config, state_path=..., audit_path=...)` with no `arm()` call in the module).
    None of the three lanes has a `SHIP_GATE.json` with `gate_pass=True` on disk, so
    `enforce_ship_gate` would refuse an arm attempt even if one were made.
  - `src/scanner/automation/state_engine.py` — modified elsewhere in this branch's history (not
    by this review-response session), adding `crypto_carry` to `KNOWN_LANES` and the
    `get_halted_strict()` fail-closed lane check. Reviewed here: the legacy global
    `halted=True` still force-halts every lane (fail-safe OR, unchanged); `get_halted_strict`
    fails closed (returns `True`/halted) on a missing file, unreadable JSON, non-dict payload,
    missing `halted_lanes`, or a missing/non-bool per-lane entry — never assumes a lane is safe
    by omission.

- **What changed and why:** This PR is a multi-round CodeRabbit-review response (security,
  correctness, stability fixes across `brain_loop`, `agent_runtime`, `axiom_operator`, and
  several shadow-lane driver scripts) plus, separately on the same shared branch, ongoing
  feature work (crypto XS-momentum, cash-and-carry, and Track B shadow lanes with their own
  per-lane `LiveGate` wrappers and per-lane halt support). None of the live/arm-triggering
  files in this diff call `.arm()`, flip `oanda_environment` away from `"practice"`, or relax
  halt semantics — every new lane's `LiveGate` starts disarmed with no ship-gate artifact on
  disk to pass, and the state-engine changes strictly narrow (add fail-closed checks), not
  loosen, halt evaluation.
- **Confirms `oanda_environment` remains `"practice"`:** yes — `grep -rn 'oanda_environment\s*=\s*"live"' src/ scripts/` returns no matches in this diff; `.claude/state.json`'s `oanda_environment` field is untouched by any file in this evidence list.
- **Confirms `halted` semantics are unchanged or explicitly intended:** yes, explicitly intended — the `state_engine.py` diff only ADDS a stricter fail-closed path (`get_halted_strict`) and a new lane name to the existing allowlist; the pre-existing fail-safe-OR (global halt beats every lane) is unchanged. The three new `LiveGate` wrappers are structurally incapable of arming (no `arm()` call site, no passing ship-gate artifact).
- **Reviewer:** Claude (Sonnet 5), automated CodeRabbit-review-response session
- **Reviewer sign-off notes:** This is an automated agent's evidence file, not a human sign-off — per this policy's own documented limit ("Not a substitute for code review... checks that an evidence artifact was added, not that a human actually reviewed it carefully"), the operator should still review this PR's live/arm-surface diffs before merge if that hasn't already happened. This file only attests to what was mechanically verified above (no `arm()` calls, no live-environment string literals, fail-closed halt semantics preserved).
