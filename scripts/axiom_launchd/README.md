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
