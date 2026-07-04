# AXIOM / Buddy LaunchAgents

Daemonizes the four background processes so a **reboot or logout never silently
takes the whole stack down again** (root cause of the 2026-06-30 11:35 UTC outage:
a machine reboot reaped four plain detached processes that nothing relaunched).

| LaunchAgent | Process | Port | Notes |
|---|---|---|---|
| `com.axiom.api`   | FastAPI read/control data layer | `:8888` | loopback-only, `AXIOM_CONTROL_ENABLED=1` |
| `com.axiom.web`   | Next.js operator cockpit | `:51999` | dev server |
| `com.buddy.trend` | OANDA daily-trend trading loop | — | **PRACTICE only** |
| `com.buddy.tier7` | Tier 7 bounded self-heal loop | — | L5-clamped |
All four: `RunAtLoad=true` (start at login) + `KeepAlive=true` (restart on exit).
`WorkingDirectory` is pinned to the repo root so `.env.local` (OANDA practice creds)
is found by the dotenv walk — the daemons authenticate without shell env.

## Fifth agent: com.buddy.learning_loop (NOT installed by load.sh)

| LaunchAgent | Process | Port | Notes |
|---|---|---|---|
| `com.buddy.learning_loop` | Market-closed continual-learning batch (`scripts/offline_learning_cycle.py`) | — | scheduled only (Saturday), `RunAtLoad=false`, `KeepAlive=false` — not resident |

`com.buddy.learning_loop.plist` exists on disk but is deliberately **excluded** from
`load.sh`'s `LABELS` list. It's a scheduled batch job, not a resident daemon — bundling
it with the other four would mean every routine `load.sh` re-run (e.g. after a reboot)
silently installs/enables it too, with no separate decision point. Never touches OANDA,
halt/arm state, or live execution (see `scripts/offline_learning_cycle.py` docstring) —
this exclusion is about install *hygiene*, not safety.

### Enabling the learning-loop schedule

```bash
cp scripts/axiom_launchd/com.buddy.learning_loop.plist ~/Library/LaunchAgents/
launchctl bootout   "gui/$(id -u)/com.buddy.learning_loop" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.buddy.learning_loop.plist
launchctl enable    "gui/$(id -u)/com.buddy.learning_loop"
```

To pause it again: `launchctl bootout gui/$(id -u)/com.buddy.learning_loop`.

## SAFETY INVARIANT — KeepAlive is NOT a halt bypass

`KeepAlive` only keeps the **process** alive. It does **not** override trading safety:

- The trading loops re-read `.claude/state.json` **every cycle**. If `halted=true`,
  a respawned loop **refuses to trade** (enforced at `src/equity/oanda_trend.py`
  halt check + `src/scanner/execution.py:2081`). A reboot therefore restores the
  operator's **last** state — it never resumes trading the operator had halted.
- `oanda_environment` is re-derived and pinned to `practice` every cycle
  (`src/scanner/config.py` default + `assert_practice`). KeepAlive cannot flip env.
- Leverage is re-clamped to ≤15× every cycle regardless of any override.

So daemonization restores **availability**, never **authority**. The Hard NOs
(practice-only, respect halt, no real money, ≤15× leverage) hold across respawns.

**If the bot ever moves to real-money / live trading, `KeepAlive` on
`com.buddy.trend` MUST be reconsidered / human-gated.** Auto-respawning a *live*
money-moving loop on every boot is not acceptable; practice-only is why it is safe
today.

## Usage

```bash
bash scripts/axiom_launchd/load.sh        # install + (re)load all four
launchctl print gui/$(id -u)/com.axiom.api   # inspect one service
launchctl bootout gui/$(id -u)/com.buddy.trend   # stop + unload one (e.g. to pause trading)
```

To **stop the bot from trading**, prefer setting `state.json halted=true` (or the
AXIOM control panel HALT) over unloading the agent — that keeps the supervisor up
and the halt auditable, rather than just killing the process.
