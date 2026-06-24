# Managed trust anchor — install (one privileged step)

This closes the *root* of two floors the in-repo gates could only make git-visible:
- **#1 gate-tamper / #5 hook-neuter (disable dimension):** the worker can edit `.claude/settings.json`
  to remove or re-point the Stop hook. The in-repo gates can't stop that — it's the worker's own file.
- **Fix:** pin the Stop hook in **enterprise managed settings**, installed to a **root-owned system
  dir outside the repo**. Managed settings are the highest precedence and *"can't be overridden by
  anything"* (verified: code.claude.com/docs/en/settings, 2026-06-23). Claude Code's write access is
  confined to the project tree, so the worker **cannot edit or remove** the managed file/hook without
  admin — which **you** control.

**Honest framing:** this does not make tampering *impossible* in the abstract; it turns *"the worker
can disable the gate"* into *"the worker can't disable the gate without admin write to a root-owned
system path,"* and admin is the operator's. That is the real, and correct, win.

## The one privileged step (you run this; the agent does NOT)

The agent prepared three files in this directory. Install them to the system managed dir:

| OS | Managed dir |
|----|-------------|
| **macOS** (primary) | `/Library/Application Support/ClaudeCode/` |
| **Linux / WSL** | `/etc/claude-code/` |
| **Windows** | `C:\Program Files\ClaudeCode\`  ← **not** `C:\ProgramData\ClaudeCode\` (deprecated as of v2.1.75) |

Easiest (macOS/Linux), idempotent:

```
sudo bash .claude/loop/managed/install_managed_anchor.sh
```

(Run it *without* `sudo` first to see the exact commands and a dry run — it changes nothing as
non-root.) Then verify (no sudo needed):

```
python3 .claude/loop/managed/verify_managed_anchor.py
```

A green result confirms from disk: the managed file is present, wires the wrapper Stop hook, and is
**not writable by your user** (root-owned) — the property that makes it un-overridable.

## What gets installed

- **`managed-settings.json`** — pins a `Stop` hook to the wrapper. Merges with (and cannot be removed
  by) local settings.
- **`ml_engine_gate_wrapper.py`** — the root-owned Stop-hook gate. It is **self-contained**: it
  re-derives the gate-script SHA-256 hashes itself (so a *neutered in-repo checker* is caught), bakes
  the canonical pinned-script list (so manifest *entry-drop* is caught), is scoped to this repo (a
  no-op elsewhere), and only then runs the in-repo `risk_monitor`. `TARGET_REPO` is baked, not
  env-overridable, so it can't be pointed at a clean fixture. On any Hard-NO/integrity failure it
  exits 2 → blocks the turn from ending.

## Re-install when the pinned set changes

The wrapper bakes `GATE_SCRIPTS` (in lockstep with `.claude/loop/_integrity.py`). If you ever add a
gate script, update both and **re-run the install step** so the root-owned wrapper learns the new pin.

## Optional stronger lockdown — `allowManagedHooksOnly`

Add `"allowManagedHooksOnly": true` to `managed-settings.json` to load **only** managed/SDK/managed-
plugin hooks and block all user/project hooks. This prevents the worker from adding *any* hook, but
it also blocks the local SessionStart context-boot + tmux hooks — so if you enable it, also move the
SessionStart boot hook into `managed-settings.json`. Left off by default to avoid disrupting your
local workflow.

## Optional (EXPERIMENTAL) — agent-based hook for verifier dispatch

The docs define a hook `type: "agent"` that spawns a subagent to verify a condition — **"experimental
and may change."** Added to managed settings, it moves the *verifier dispatch into the harness* (the
worker can't skip it), which hardens floor #5's **lazy** dimension. Snippet (Stop event):

```json
{ "hooks": { "Stop": [ { "hooks": [
  { "type": "agent",
    "prompt": "Hook input: $ARGUMENTS. You are an independent verifier. Read .claude/loop/state.json, .claude/loop/agent_verdict.json, and .claude/loop/verdict.json in /Users/buddy/Documents/ml_engine. If the loop claims a stopping point but the deterministic gate state looks inconsistent (verify_gate verdict not PASS, or open questions remain), return decision=block with a reason. Otherwise allow.",
    "model": "claude-haiku-4-5-20251001" } ] } ] } }
```

**Honest limit (don't over-trust it):** the subagent can only **read** repo files, which the worker
controls — so it does **not** produce a *non-forgeable* verdict; and it is experimental. It hardens
"did the worker skip verification" (harness-spawned, can't skip), **not** "did the worker lie." The
**enforced** floor is the deterministic managed wrapper above. Treat the agent hook as defense-in-
depth, not closure.

## Residual floors after install (see NOTES "Irreducible floors")

- The wrapper trusts the **in-repo manifest**; editing a gate script *and* its manifest together is
  still git-visible (not blocked). Closing that: bake the manifest hash into the root-owned wrapper
  (re-install on every gate change) — heavier, available if you want it.
- The agent half of verification (semantic judgment) remains advisory (L-011). The deterministic half
  is enforced; human review backstops the rest.
- **Scope/cwd assumption:** the wrapper no-ops when the Stop-hook input `cwd` is outside this repo (so
  it doesn't interfere with your other projects). That `cwd` is supplied by Claude Code in the hook
  payload, not freely writable by the worker mid-session — the same harness-integrity assumption the
  in-repo `stop_gate.sh` already relies on. If that payload could be forged, scoping would be
  bypassable; that's a harness assumption, not a defect in these files. Set `allowManagedHooksOnly`
  and/or remove the scope check (always-on) if you accept system-wide interference for stricter scope.
