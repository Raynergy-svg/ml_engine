# Hermes Watchdog + Daily Brief — Design Spec

**Date:** 2026-05-12
**Status:** spec written, awaiting operator review before plan generation
**Parent spec:** `docs/superpowers/specs/2026-05-12-hermes-pattern-absorption-design.md`
**Predecessor:** Tier 1 cherry-picks (T1–T6) — landed in main at `047f0e3`, operator-confirmed rendering 2026-05-12

## Goal

Give Buddy a continuous, structured narration of its own state — written by Buddy, into a single bounded brain file — so that:

1. Future-Claude (any session) opens `.claude/brain/hermes_watchdog.md` and sees what happened while away.
2. Hermes (configured separately as a Telegram concierge) reads the same file to answer operator status questions and surface escalations from a phone.
3. The operator gets a daily structured brief at 07:00 UTC summarizing the prior 24h.

This spec covers **Buddy's writer side only**. Hermes-as-Telegram-concierge is a follow-up spec; this one fixes the contract so Hermes can be built against it independently.

## Scope

**In scope:**
- A `hermes_watchdog` Python script that runs every 30 min via `ScheduledJobsRegistry` (T1), reads state files, and appends `watch` / `silent` entries to a digest file when warranted.
- A `hermes_daily_brief` Python script that runs daily at 07:00 UTC, composes a fixed-schema summary, and appends a `brief` block to the same digest.
- A new brain-file `hermes_watchdog.md` participating in `brain_caps` enforcement, with weekly rotation to `.claude/brain/.archive/`.
- Two new `.claude/jobs.json` entries.
- Test coverage per CLAUDE.md no-mocks rule (real disk via `tmp_path`).

**Out of scope (future specs):**
- Hermes-side Telegram bot setup (bot token, DM pairing, allowlist).
- Hermes-side skills that read `hermes_watchdog.md` and push to chat.
- Operator-actioned escalations (halt/unhalt from chat, retrain triggers, Claude Code dispatch).

## Architecture

```
EVERY 30 MIN (via T1 ScheduledJobsRegistry → daemon thread)
    python -m src.scanner.automation.hermes_watchdog
        reads:  .claude/heartbeat.json
                .claude/state.json
                .claude/alert_state.json
                trained_data/trade_journal_rl.json
                trained_data/virtual_trades.jsonl
                trained_data/jobs_runtime_state.json   (T1's last_status / last_error per job)
        decides: alert / silent / no-op
        writes:  append entry to .claude/brain/hermes_watchdog.md
                 (atomic temp+rename via safe_json_write pattern adapted for markdown)

DAILY 07:00 UTC (via T1 ScheduledJobsRegistry → daemon thread)
    python -m src.scanner.automation.hermes_daily_brief
        reads:  same files + trained_data/models/<PAIR>/transformer_direction.meta.pkl
                .claude/brain/hermes_watchdog.md   (for "alerts last 24h" rollup)
        composes: fixed-schema brief block (no LLM)
        writes:  append brief block to .claude/brain/hermes_watchdog.md

ON BOOT (via existing brain_caps check)
    check_brain_caps() warns if hermes_watchdog.md exceeds cap.
    A small rotation helper (new) moves it to .archive/hermes_watchdog_YYYY_WW.md
    when over hard_cap * 1.5, starting a fresh file.
```

Two crucial properties:
- **Both scripts run as subprocesses fired by the T1 registry's `subprocess.Popen(shlex.split(cmd))` pattern.** They have no in-process access to the running scanner's `error_banner`, `counters`, or `phase_state` — only file-based state. This is by design: Claude-free in the hot path, file contracts are the boundary.
- **Both append-only.** No mutation of past entries. Rotation = full-file move, never partial.

## File contracts

### `.claude/brain/hermes_watchdog.md`

Append-only digest. Operator can read. Future-Claude reads at session start. Hermes-Telegram (future) reads on operator request.

**Entry format:**

```markdown
# Hermes Watchdog Digest

> Auto-written by `hermes_watchdog` (every 30 min) and `hermes_daily_brief` (daily 07:00 UTC).
> Operator-readable. Rotated weekly into `.claude/brain/.archive/`.

## 2026-05-12

### 07:00Z — brief
- halted: true · mode: live · cycles_today: 0 (process restarted at 06:42Z)
- trades_24h: 0 trades · P&L $0
- model ages: EUR_USD 14.2d · GBP_USD 9.0d (M15)
- jobs: 2 active, 0 paused, 0 failures in last 24h
- alerts_24h: 0 watch entries
- notable: scanner in HALTED state for 18d; operator un-halt required to resume

### 14:30Z — watch
- trigger: consecutive_losses=3 (alert_state.json)
- recent closes: EUR_USD SHORT -$54.20 · USD_CHF SHORT -$57.94 · GBP_USD SHORT -$72.10
- heartbeat fresh (4s ago); scanner_alive=true

### 14:35Z — watch
- trigger: scheduled_jobs failure
- job_id: nightly_audit · last_status=failure · last_error="exit 1: disk full at /tmp"
- recommendation: free disk before next 22:00Z fire

### 21:00Z — silent
- (no alert; logged for completeness — last silent was 16:00Z)
```

**Entry header pattern:** `### HH:MMZ — {brief|watch|silent}`

- `brief` — daily structured summary (one per day, at 07:00 UTC).
- `watch` — anomaly detected; details below.
- `silent` — routine no-op; logged at most once every 4h to prove the watchdog is alive without spamming.

**Day header:** `## YYYY-MM-DD` — inserted by the first entry of a new UTC day. Watchdog/brief scripts check the last day-header in the file and add a new one if the current UTC date differs.

### `.claude/jobs.json`

Two new entries (both `enabled: true` by default — this is the operator-visible default we're shipping).

```json
{
  "jobs": [
    {
      "job_id": "homework_weekly",
      "name": "Weekly homework batch",
      "schedule": "weekly_SUN_02:00",
      "command": "python buddy_scanner.py homework --generate-batch --last 100",
      "enabled": false,
      "description": "Regenerate the last 100 trade homework entries every Sunday 02:00 UTC."
    },
    {
      "job_id": "hermes_watchdog",
      "name": "Hermes Watchdog — anomaly detection",
      "schedule": "every_30_minutes",
      "command": "python -m src.scanner.automation.hermes_watchdog",
      "enabled": true,
      "description": "Reads scanner state files every 30 min; writes structured observations to .claude/brain/hermes_watchdog.md when anomalies detected."
    },
    {
      "job_id": "hermes_daily_brief",
      "name": "Hermes Daily Brief — 24h structured summary",
      "schedule": "daily_07:00",
      "command": "python -m src.scanner.automation.hermes_daily_brief",
      "enabled": true,
      "description": "Composes daily fixed-schema summary; appends to .claude/brain/hermes_watchdog.md at 07:00 UTC."
    }
  ]
}
```

If `.claude/jobs.json` doesn't exist at boot, the orchestrator's `default_jobs()` returns the `homework_weekly` entry only. **This spec extends `default_jobs()` to also include the two Hermes entries.** Operator-edited `jobs.json` is preferred when present; defaults seed fresh installs.

### `.claude/brain/.archive/hermes_watchdog_YYYY_WW.md`

Weekly rotation target. The active digest rotates here when:
- on watchdog tick, the digest file size exceeds `brain_caps[hermes_watchdog.md].hard_cap * 1.5` (i.e., the high-severity threshold from `brain_caps.py`), OR
- on Sunday at the 00:00Z silent-tick (rotation is opportunistic, not on a separate cron).

`YYYY_WW` = ISO year + week number (e.g., `2026_19`).

### `brain_caps.py` extension

Add one entry:

```python
_DEFAULT_CAPS: dict[str, tuple[int, float]] = {
    "briefing.md":         (3_000, 1.20),
    "session_handoff.md":  (2_000, 1.20),
    "open_questions.md":   (1_500, 1.20),
    "strategic_log.md":    (8_000, 1.15),
    "trade_narrative.md":  (5_000, 1.15),
    "hermes_watchdog.md":  (8_000, 1.15),  # NEW: same shape as strategic_log
}
```

Same hard cap as `strategic_log.md` (digest-style file). Warn at 9.2K (1.15×). Rotation triggers at 12K (1.5×).

## Watchdog decision tree

Inputs read fresh each tick (no cross-tick state except an in-memory cache of the last alert hashes for dedup within a single tick chain):

| File | Fields read | Used for |
|---|---|---|
| `.claude/heartbeat.json` | `ts_iso`, `cycle_count`, `scanner_alive` | Process-death detection |
| `.claude/state.json` | `halted`, `mode` | Annotate every entry |
| `.claude/alert_state.json` | `consecutive_losses`, `drawdown_pct`, `win_rate_drop` | Primary alert signals |
| `trained_data/trade_journal_rl.json` | Last 5 closed trades (`pair`, `direction`, `pnl`, `closed_at`) | Loss-streak detail |
| `trained_data/virtual_trades.jsonl` | Last 20 lines | Gate-rejection rate sanity |
| `trained_data/jobs_runtime_state.json` (T1) | For each job: `state`, `last_status`, `last_error`, `last_status_at` | Job-failure alerts |

**Alert (write a `watch` entry) if ANY trigger fires:**

| Trigger | Condition | Rationale |
|---|---|---|
| `consecutive_losses_high` | `alert_state.consecutive_losses >= 3` | Catches the 14-loss-streak pattern from 2026-04-15 |
| `drawdown_breach` | `alert_state.drawdown_pct >= ScannerConfig.alert_drawdown_threshold` (or fallback `0.05`) | Operator must see this |
| `heartbeat_dead` | `now - heartbeat.ts_iso > 60s` AND last-tick had `scanner_alive=true` | Process died silently |
| `job_failing` | Any job in `jobs_runtime_state` has `last_status="failure"` AND `state="active"` AND `last_status_at` is newer than the most recent `watch` entry that referenced this `job_id` | T1 surface — silent cron failures became visible day 1 |

**Silent (write a `silent` entry) if:**
- No trigger fired AND
- No `watch` entry was written in the last 4 hours AND
- No `silent` entry was written in the last 4 hours.

The last condition rate-limits silents to once every 4h max — proof-of-life without log spam.

**No-op (write nothing) otherwise.** A healthy 30-min tick during normal trading hours typically writes nothing for 8h between scheduled silents.

## Daily brief composition

Fields, all fixed-schema (no LLM):

| Field | Source | Format |
|---|---|---|
| `halted` | `state.json:halted` | `true` / `false` |
| `mode` | `state.json:mode` | `live` / `demo` |
| `cycles_today` | `heartbeat.cycle_count` – yesterday's stored value | int |
| `trades_24h` | `trade_journal_rl.json`, count + sum(pnl) where `closed_at >= midnight UTC` | `N trades · P&L $X.XX` |
| `model_ages` | walk `trained_data/models/<PAIR>/transformer_direction.meta.pkl`, compute days since `trained_at` | `EUR_USD 14.2d · GBP_USD 9.0d (M15)` |
| `jobs` | `jobs_runtime_state.json` (T1) — count active/paused/failed in last 24h | `N active, M paused, K failures in last 24h` |
| `alerts_24h` | grep digest for `### .*— watch` entries in last 24h | int |
| `notable` | hard-coded selection: highest-severity active condition wins; else "all systems nominal" | one sentence |

The `notable` selection priority (first match wins):
1. `halted=true` AND `consecutive_losses >= 3` → `"halted on loss streak — operator review required"`
2. `halted=true` → `"halted; operator un-halt required to resume"`
3. any `job_failing` in last 24h → `"scheduled job <id> failed: <last_error[:80]>"`
4. any model age > 30 days → `"model staleness — <PAIR> is <N>d old"`
5. else → `"all systems nominal"`

### Cross-tick state

The brief needs yesterday's `heartbeat.cycle_count` to compute `cycles_today`. Stored in:

```
.claude/hermes_brief_state.json
{
  "last_brief_at_iso": "2026-05-12T07:00:00+00:00",
  "cycle_count_at_last_brief": 12345
}
```

Atomic write via the temp+rename pattern. If missing on first run, `cycles_today` reports `unknown` and the brief is still composed.

## Forward integration: Hermes-Telegram concierge

**Out of scope for this spec.** Future spec (`docs/superpowers/specs/YYYY-MM-DD-hermes-telegram-concierge-design.md`) will cover:

- Hermes-side bot setup, DM pairing, operator allowlist.
- Skills: `/status` (tail `hermes_watchdog.md` for last 24h watch entries), `/brief` (return last `brief` block).
- Push notifications: Hermes runs its own cron that diffs `hermes_watchdog.md`'s tail every 5 min and forwards new `watch` entries to Telegram.
- Escalations from chat → Claude Code subprocess (e.g., "fix the SHORT bias" → Hermes spawns `claude --workdir=/Users/buddy/Documents/ml_engine -p '...'` in a worktree).

The contract this spec locks in — the **file format and entry schema** of `hermes_watchdog.md` — is the boundary Hermes builds against. Nothing about Hermes affects Buddy.

## Test strategy

Per CLAUDE.md no-mocks rule: real classes, real disk via `tmp_path`, real schedule-grammar calls.

| Test file | Coverage |
|---|---|
| `tests/test_hermes_watchdog.py` | Drive `hermes_watchdog.main()` with crafted state files in `tmp_path`. Assert the expected `watch` / `silent` / no-op decision under each trigger. Assert entry text matches the documented format. Verify dedup logic for repeated job failures. |
| `tests/test_hermes_daily_brief.py` | Drive `hermes_daily_brief.main()` with crafted journal + heartbeat + meta sidecars. Assert each brief field renders correctly. Assert `cycles_today` reads/writes `hermes_brief_state.json` round-trip. Assert `notable` selection priority. |
| `tests/test_hermes_watchdog_rotation.py` | Pre-populate digest beyond hard_cap*1.5 in `tmp_path`. Assert next watchdog tick rotates to `.archive/hermes_watchdog_YYYY_WW.md`, starts fresh file with day header. |
| `tests/test_hermes_jobs_default.py` | Verify `default_jobs()` includes the two Hermes entries; verify `.claude/jobs.json` round-trips them. |

## Open decisions (operator review before plan)

These are the choices I made by default. Each is amendable — flag any you want different before I write the implementation plan.

| # | Decision | Default | Alternatives |
|---|---|---|---|
| D1 | Watchdog frequency | every 30 min | every 15 min (more noise) / every 60 min (slower escalation) |
| D2 | Brief time | 07:00 UTC | 13:00 UTC (US market open) / operator-local timezone (no croniter dep though) |
| D3 | Silent rate-limit | once per 4h | per 1h (more proof-of-life) / never (audit-clean but no liveness) |
| D4 | Both new jobs `enabled: true` default | true | false (operator opts in by editing `.claude/jobs.json`) |
| D5 | Drawdown threshold for `drawdown_breach` | `ScannerConfig.alert_drawdown_threshold` if exists else 0.05 | hardcoded different value |
| D6 | Where rotation happens | inline in watchdog script | separate `python -m hermes_rotate` job |
| D7 | brain_caps entry size | 8K hard, 1.15× warn | 5K (tighter) / 12K (looser) |
| D8 | `notable` priority order | halted-loss-streak > halted > job-failure > model-stale > nominal | reorder |

## Risk + rollback

| Concern | Mitigation |
|---|---|
| Watchdog itself fails → `jobs_runtime_state[hermes_watchdog].last_status="failure"` → next watchdog tick fires "job_failing" alert about itself, recursive | Watchdog explicitly skips its own job_id when scanning `jobs_runtime_state`. Failure surfaces only on T1 Jobs screen. |
| Brief takes >2s on slow disk because of meta sidecar walks | Daemon-thread execution from T1 dispatch — scan loop unaffected. Tests assert <2s budget against `tmp_path` SSD-like timing. |
| Digest grows unbounded between rotations | brain_caps warns at 9.2K (boot banner); rotation triggers at 12K (inline in watchdog). Worst case: one weekly rotation cycle of overflow before next boot. |
| Concurrent appends (brief at 07:00 while watchdog ticks at 07:00) | Both scripts write via temp+rename atomic pattern. Whichever lands second sees the other's content because read-rewrite-rename uses fresh read. No corruption; entries may be reordered by milliseconds. Acceptable. |
| Watchdog reads partially-written journal/heartbeat file mid-write | `safe_json_read` (cloud-branch utility) with try/except. On parse failure, the tick is a no-op for that input; the next tick (30 min later) sees the consistent state. |
| Operator wants to silence the watchdog temporarily | `hermes pause hermes_watchdog` via the T1 Jobs screen (Press P on F9). State flips to `paused`; `due_jobs` skips it; resumes with Press R. |

## Acceptance

Spec is complete when:
1. This document is committed under `docs/superpowers/specs/`.
2. Operator reviews and either approves the 8 default decisions in "Open decisions" or amends them inline.
3. Implementation plan written (separate doc via `superpowers:writing-plans` skill) targeting `docs/superpowers/plans/2026-05-12-hermes-watchdog-brief.md`.

Plan execution itself is gated on:
- Tier 1 (T1–T6) live on main (DONE — `047f0e3`).
- Plan operator-approved.
