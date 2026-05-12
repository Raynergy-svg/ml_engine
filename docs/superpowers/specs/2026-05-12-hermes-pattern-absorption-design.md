# Hermes Pattern Absorption — Design Spec

**Date:** 2026-05-12
**Status:** approved, plans authored
**Forward refs:**
- `docs/superpowers/plans/2026-05-12-tier1-cherry-picks.md`
- `docs/superpowers/plans/2026-05-12-tier2-cherry-picks.md`
- `docs/superpowers/plans/2026-05-12-tier3-cherry-picks.md`

## Goal

Lift Buddy from "TUI that drives an FX bot" toward "production autonomous tool that an operator can trust at-a-glance" by absorbing 13 patterns from Hermes Desktop (`Raynergy-svg/Buddy-Autonomous-loop`, a fork of `fathah/hermes-desktop`). Tier 1 patterns ship as standalone Buddy improvements with no external dependency. Tier 2 patterns build on Tier 1. Tier 3 patterns are speculative and gated on demonstrated need.

## Research provenance

Three `feature-dev:code-explorer` subagents read `/tmp/buddy-autonomous-loop` (shallow clone, 124 TS + 29 TSX files, 0 Python) on 2026-05-12, each on a separate lens:

1. **Memory & state surfacing UX** — how the GUI presents MEMORY.md, USER.md, sessions, profiles, skills, SOUL.md.
2. **Always-on agent IPC + backend contracts** — how the GUI connects to the always-running Hermes Agent at `127.0.0.1:8642`.
3. **Operator transparency** — how the GUI shows token usage, tool progress, costs, model state, errors.

Combined output: 21 patterns. After dedup across overlapping lenses (e.g., "scheduler missing status fields" + "per-job cards with last_error" describe the same gap from backend vs UX angles), 13 unique candidates remain.

Already absorbed by the cloud branch (`origin/claude/cherry-pick-ml-engine-upgrade-hKlIu`, 9 commits ahead of main) — not re-recommended here:
- FTS5 trade-journal search modal
- Vim ':' command palette
- Scheduled jobs registry (`scanner/automation/scheduled_jobs.py`)
- Brain-file char caps (`scanner/automation/brain_caps.py`)
- Live log viewer modal (`tui/log_tailer.py` + modal)
- Skills content-hash lockfile + CI drift workflow
- `safe_json_write` atomic writes consolidated across 14 sites

## Decision

**Sequencing:**

```
NOW   → Land cloud cherry-pick branch (PR pending)
THEN  → Tier 1 (T1–T6): 6 independent PRs, each landable in isolation
THEN  → Hermes watchdog + brief design (separate spec, builds on Tier 1 surfaces)
THEN  → Tier 2 (T7–T10): 4 independent PRs
DEFER → Tier 3 (T11–T13): write specs only when specific operational need surfaces
```

**Rationale for ordering Tier 1 before Hermes integration:**
Once Buddy has `last_error` fields on scheduled jobs, a liveness badge driven by `heartbeat.json`, an inline error banner pattern in F1, and a cumulative work-unit counter, the Hermes-on-Telegram concierge gets **clean structured surfaces to read from and render alerts against**. Without these, Hermes would be grepping raw logs and guessing. Tier 1 is the unblock, not the dependency.

**Why these specific 13:**
Each pattern was scored on three axes by the research agents — leverage for Buddy's autonomy story, transfer confidence to Python+Textual single-process, and implementation effort. The 13 chosen all score HIGH or MEDIUM on transfer confidence. Patterns scoring LOW (SSH connection modes, SSE empty-probe, multi-profile UX) were dropped — see "Drops" section.

## Architecture (post-Tier-1, pre-Hermes)

```
┌──────────────────────────────────────────────────────────────────────┐
│  OPERATOR (Textual TUI — F1 Overview / F2 Inbox / F3 Trades / etc.)  │
│  Sees: liveness badge · cycle counter · phase indicator              │
│         · error banner (T6) · jobs panel (T1 surface)                │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↑ Reactive widgets
                               │ poll/subscribe
┌──────────────────────────────┴───────────────────────────────────────┐
│  EmbeddedScanner (src/tui/embedded_scanner.py)                       │
│    run_one_cycle() emits:                                            │
│       _brain(msg)            existing — rich-markup F1 feed          │
│       _progress_cb(phase)    NEW (T4) — transient phase tag          │
│       _error_banner = msg    NEW (T6) — sticky exception surfacing   │
│       _stats                 NEW (T3) — counters increment           │
│                                                                       │
│  Heartbeat writer (existing) → .claude/heartbeat.json (10s)          │
│                                                                       │
│  ScheduledJobsRegistry (cloud branch, wired in orchestrator)         │
│    JobRuntimeState now includes:                                     │
│       last_run_at, last_status, last_error, run_count   (existing)   │
│       state ∈ {active, paused}                          NEW (T1)     │
│       next_run_at_iso                                   NEW (T1)     │
│       last_status_at                                    NEW (T1)     │
│                                                                       │
│  ConfigAdjuster._load_state()                                        │
│    Now wrapped in 5s TTL cache, invalidate-on-save        NEW (T5)   │
└──────────────────────────────────────────────────────────────────────┘
```

## Tier 1 — Standalone production-quality lifts (T1–T6)

Each ships independently as a small PR. None depends on Hermes. None depends on the others (T3+T4 share a render surface but can land in either order).

| ID | Pattern | Files | Effort | Confidence |
|---|---|---|---|---|
| **T1** | Scheduler observability — `state`/`next_run_at`/`last_status_at` fields + pause/resume/trigger-now + Jobs TUI screen | `scanner/automation/scheduled_jobs.py`, new `tui/screens/jobs_screen.py`, `tui/app.py` binding | S→M | HIGH |
| **T2** | Liveness badge — Textual `Reactive` widget reads `heartbeat.json`, renders LIVE/HALTED/INIT/ERROR | `tui/widgets/liveness_badge.py` (new), `tui/app.py` footer mount | S | HIGH |
| **T3** | Cumulative work-unit counter — accumulating `cycles · pairs · gates · trades` bar | `tui/embedded_scanner.py` (counter dataclass), `tui/widgets/stats_bar.py` (new), `tui/app.py` | S | HIGH |
| **T4** | Transient phase indicator — `_progress_cb(phase, detail)` hook fired at scan/agent/gate/exec boundaries | `tui/embedded_scanner.py` (hook), `tui/widgets/phase_indicator.py` (new), `tui/app.py` | S | HIGH |
| **T5** | TTL cache for ConfigAdjuster reads — 5s in-memory cache, invalidate-on-write | `scanner/automation/config_adjuster.py` | S | HIGH |
| **T6** | Inline error banner with View-Log — scanner sets `error_banner`; F1 renders dismissible notification with deeplink to log viewer | `tui/embedded_scanner.py` (banner field), `tui/app.py` (reactive watcher + notify) | S→M | HIGH |

Full TDD task breakdown in `docs/superpowers/plans/2026-05-12-tier1-cherry-picks.md`.

## Tier 2 — Depth (T7–T10)

Ships after Tier 1 has been live for ≥ 1 week (gives time for Tier 1 regressions to surface). Builds on Tier 1 surfaces.

| ID | Pattern | Files | Effort | Confidence |
|---|---|---|---|---|
| **T7** | Brain section editor — parse `.claude/brain/briefing.md` by `##` headings, per-section CRUD with `brain_caps`-aware pre-write validation, reset-to-template | `tui/widgets/brain_editor.py` (new), `.claude/brain/briefing.md.default` (new), `tui/app.py` | M | HIGH |
| **T8** | Config-write triggers reload — post-write hook in `AdjustmentApprover._save_approved` flips a `config_dirty` flag; `EmbeddedScanner.run_one_cycle` checks and reloads pre-tick | `scanner/automation/adjustment_approver.py`, `tui/embedded_scanner.py` | S→M | HIGH |
| **T9** | Two-tier trade journal cache — `.claude/ui_cache/trades_index.json` with precomputed summary rows; incremental sync via append-offset; Trades screen reads index | `tui/cache/trades_cache.py` (new), `tui/screens/trades_screen.py` (consume) | M | HIGH |
| **T10** | Per-pair model inventory panel — walk `trained_data/models/*/transformer_direction.meta.pkl`; show per-pair card (age, holdout %, gate active, last retrain) | `tui/widgets/model_inventory.py` (new), `tui/screens/diagnostics_screen.py` (mount) | M | MEDIUM |

Full TDD task breakdown in `docs/superpowers/plans/2026-05-12-tier2-cherry-picks.md`.

## Tier 3 — Speculative (T11–T13)

Each requires a concrete operational trigger before being scheduled. Specs and plans below are *drafts* — they describe the design but assume the value case is unconfirmed.

| ID | Pattern | Files | Effort | Confidence |
|---|---|---|---|---|
| T11 | Stepped retrain progress — 3-layer (bar + step label + log) progress display threaded through trainer | `training/trainers/transformer_trainer.py` (emit), new `tui/widgets/retrain_progress.py` | M→L | MEDIUM |
| T12 | Two-pane skills/rules viewer — `.claude/rules/*.md` browsable with active-vs-draft badges | `tui/screens/rules_screen.py` (enhance) | S→M | MEDIUM |
| T13 | Capability inventory — live registry of which ML heads loaded per pair (introspect `GateEvaluator` + meta sidecars) | `tui/widgets/capability_inventory.py` (new), `tui/screens/diagnostics_screen.py` | M | MEDIUM |

Full TDD task breakdown in `docs/superpowers/plans/2026-05-12-tier3-cherry-picks.md` — note: each task is gated on an operator-defined trigger.

## Drops

The following patterns surfaced in research but are intentionally not in scope. Reason recorded for each so a future operator doesn't re-litigate.

| Pattern | Source agent | Why dropped |
|---|---|---|
| Multi-profile UX (literal) | A1.#3 | Buddy has no multi-profile concept. Adapted version is T10 (per-pair model state). |
| Streaming chunk callback interface | A2.#2 | Overlaps with existing `self._brain(...)` pattern; marginal lift. |
| SSE empty-probe fallback | A2.#3 | Requires HTTP boundary Buddy doesn't have. Carry forward if Buddy grows a scanner HTTP API. |
| Connection mode (local/remote/SSH) | A2.#4 | Roadmap pattern — only matters if scanner runs on a different host. Defer until that need surfaces. |
| Slash-command autocomplete with categories | A3.#5 | Cloud branch already shipped `:` palette (pick #4). The autocomplete-with-categories layer is a polish, not a separate cherry-pick. |
| Memory provider plugin registry | A1.#7 | Speculatively adapted to "ML head capability inventory" — captured as T13 instead. |

## Open questions and resolutions

The architecture sketch from 2026-05-12 surfaced 5 open questions for the Hermes watchdog/brief design. Tier 1 resolves most of them by providing structured surfaces:

| Q | Original question | Resolution after Tier 1 |
|---|---|---|
| Q1 | Telegram bot identity (per-operator vs allowlist) | **Defer to Hermes watchdog spec.** Tier 1 doesn't depend on this. |
| Q2 | What goes in the daily brief | **Render the T3 counter values + T1 `last_error` fields + T6 error banner history.** Fixed schema, no LLM-summarized prose in v1. |
| Q3 | Escalation threshold (when does watchdog page operator) | **Triggers: any T6 banner active, any T1 `last_status="failure"` on a non-paused job, heartbeat stale > 60s, consecutive losses ≥ 3.** Hard thresholds — no LLM judgment. |
| Q4 | Where the watchdog Python script lives | **`src/scanner/automation/hermes_watchdog.py`** (parallel to `scheduled_jobs.py`). Confirmed by recon — neighbors stay together. |
| Q5 | Brief delivery channel | **Both Telegram AND append to `.claude/brain/trade_narrative.md` (or new `hermes_watchdog.md`)** so Claude Code sees Hermes' summary at next session start. |

The Hermes watchdog spec will be written as a separate doc after Tier 1 lands.

## Risk and rollback

Per-tier rollback strategy. Each tier-plan PR is independent — partial rollback (revert one PR) is the supported recovery path.

**Tier 1 risk surface:**
- T1: extending `JobRuntimeState` schema is additive — old state JSON loads as old shape with new fields defaulted. Reversible.
- T2: pure-read widget on existing file. Reversible.
- T3, T4: new counter/hook in EmbeddedScanner — if it raises, the scan loop catches via existing exception handler. Set `enable_T3_stats: bool = True` flag in `ScannerConfig` for kill-switch.
- T5: TTL cache wrapper — if it returns stale data, the wrong adjustments are applied for ≤ 5 seconds. Reversible.
- T6: new exception surfacing — if the banner widget itself raises, the scan loop continues (banner is best-effort). Reversible.

**Tier 2 risk surface:**
- T7: brain editor — if a save corrupts `briefing.md`, the existing brain-caps CI catches oversize but a hand-edit could still write malformed content. Mitigate: save through `safe_json_write`-equivalent for markdown (temp + rename), keep `.bak` for one cycle.
- T8: config-reload-on-write — if reload causes a partial-state cycle, scanner might trade on half-applied config. **Mitigation: reload happens AT the beginning of `run_one_cycle()`, never mid-cycle.** This is non-negotiable.
- T9: trade journal cache — corruption in cache produces wrong UI rendering; the journal itself is unaffected. Operator can delete `.claude/ui_cache/trades_index.json` to force rebuild.
- T10: model inventory — read-only on PKL sidecars. Worst case: stale or wrong display. No execution impact.

**Tier 3 risk:** specs are drafts. Operator approves activation case-by-case.

## Testing discipline

Per CLAUDE.md "NO MOCK CODE" rule: all new tests use real classes against real disk via `tmp_path`. Existing tests in the cloud branch (e.g., `test_scheduled_jobs.py`) that use `MagicMock` are **grandfathered** — we don't retroactively rewrite them but never add new mocks. When migrating a test file for other reasons, drop the mocks per CLAUDE.md guidance.

Test naming: per existing `tests/test_*.py` convention.

## Skill / governance trail

- This spec follows the `superpowers:brainstorming` checklist (sections 1–9).
- Per-tier implementation plans follow `superpowers:writing-plans` format (TDD-disciplined steps with code blocks).
- Per CLAUDE.md "Subagents always pick their own skills": engineers/agents executing tier plans may invoke their own debugging/test-driven-development/code-review skills as needed.

## Acceptance

The spec is considered complete when:
1. All three tier plans exist at the documented paths.
2. Operator has reviewed and not requested changes.
3. The first Tier 1 PR (T1: scheduler observability) merges to main.

After (3), the watchdog/brief design spec is unblocked and gets written separately.
