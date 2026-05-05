# Supervisor Console — Operator Runbook

> **Audience:** the operator on duty (including future-you at 2 AM).
> **Scope:** Phase 91 Supervisor Console (US-501 → US-518). FX scanning + OANDA execution.
> **Tone:** operational discipline. If a section feels boring, that's the point.

This runbook is the single source of truth for operating the Supervisor Console. It is referenced from [CLAUDE.md](../CLAUDE.md) under **Claude Brain** and is required reading before any live-mode session.

---

## Table of Contents

1. [Launch Procedure](#1-launch-procedure)
2. [Hotkey Reference](#2-hotkey-reference)
3. [Confirmation Flows](#3-confirmation-flows)
4. [Adjustment Inbox Workflow](#4-adjustment-inbox-workflow)
5. [Kill Switch — What to Do When Hit](#5-kill-switch--what-to-do-when-hit)
6. [Reading the Gate Trace](#6-reading-the-gate-trace)
7. [Reading the Staleness Banner](#7-reading-the-staleness-banner)
8. [Reading the Weight Inspector](#8-reading-the-weight-inspector)
9. [Incident Response — 10-Minute Post-Kill Checklist](#9-incident-response--10-minute-post-kill-checklist)
10. [On-Call Escalation Checklist](#10-on-call-escalation-checklist)
11. [Appendix A — Reality Checker Walkthrough](#appendix-a--reality-checker-walkthrough)
12. [Phase 95 — First Live-Mode Re-Enable Checklist](#12-phase-95--first-live-mode-re-enable-checklist)

---

## 1. Launch Procedure

The console is launched via the `./buddy` script from the repo root.

```bash
cd /Users/buddy/Documents/ml_engine
./buddy
```

**What the launcher does:**
- Sources `.env.local` (loads OANDA credentials).
- Activates the project venv.
- Auto-detects mode: `--live` if `OANDA_API_KEY` and `OANDA_ACCOUNT_ID` are present, else `--demo`.

**Pre-flight checklist (do not skip on live mode):**
- [ ] Confirm tmux session `buddy` is the active session (panes: `tui` / `scanner` / `logs` / `shell`).
- [ ] Confirm NAV banner matches OANDA web dashboard within 0.5%.
- [ ] Confirm `StalenessBanner` is **green** (no model older than 7d) — see §7.
- [ ] Confirm pending inbox count is reviewed (F2) — no stale approval requests > 24h.
- [ ] Confirm the mode indicator in the top strip reads the expected value (`DRY-RUN` for paper, `LIVE` for execution).

If any pre-flight item fails, do **not** flip to live. Resolve first.

---

## 2. Hotkey Reference

Every binding shipped from US-504 onward. Hotkeys are global unless a screen or modal is noted.

### Global navigation (any screen)

| Key      | Action                                        | Source     |
| -------- | --------------------------------------------- | ---------- |
| `F1`     | Switch to **Overview** tab                    | US-504     |
| `F2`     | Switch to **Inbox** tab                       | US-504     |
| `F3`     | Switch to **Trades** tab                      | US-504     |
| `F4`     | Switch to **Agents** tab                      | US-504     |
| `F5`     | Switch to **Journal** tab                     | US-504     |
| `F6`     | Switch to **Config** tab                      | US-504     |
| `F7`     | Switch to **Rules** tab                       | US-504     |
| `F8`     | Switch to **Diagnostics** tab                 | US-504     |
| `Ctrl+F` | Cycle asset class (FX ⇄ Futures ⇄ Hybrid)     | US-504     |
| `Ctrl+R` | State-preserving TUI restart                  | Mythos     |
| `F12`    | State-preserving TUI restart fallback         | Mythos     |
| `c`      | Copy focused panel or full snapshot           | Mythos     |
| `q`      | Quit (clean shutdown, flushes state)          | US-504     |

### Supervisor controls (any screen, destructive)

| Key      | Action                                                | Source |
| -------- | ----------------------------------------------------- | ------ |
| `Space`  | Pause / resume scanner (`scanner_paused` flag)        | US-506 |
| `k`      | **Kill switch** — flatten all open positions          | US-507 |
| `m`      | Toggle scanner mode (DRY-RUN ⇄ LIVE)                  | US-508 |
| `a`      | Abort the highest-priority pending signal             | US-509 |
| `u`      | Health-gated unhalt after auto-halt                   | Mythos |

> Supervisor controls are **disabled** while a kill is in progress (`_kill_in_progress` guard). The status strip displays `KILL IN PROGRESS` and ignores keypresses.

### Inbox screen (F2)

| Key      | Action                                              | Source |
| -------- | --------------------------------------------------- | ------ |
| `a`      | Approve highlighted adjustment                      | US-512 |
| `r`      | Reject highlighted adjustment                       | US-512 |
| `s`      | Snooze highlighted adjustment 24h                   | US-512 |
| `v`      | View detail modal for highlighted adjustment        | US-512 |
| `Esc`    | Close detail modal                                  | US-512 |

### Trades screen (F3)

| Key      | Action                                              | Source |
| -------- | --------------------------------------------------- | ------ |
| `c`      | Close currently selected trade (per-trade flatten)  | US-510 |
| `Esc`    | Cancel close-trade confirmation modal               | US-510 |

### Rules screen (F7)

| Key      | Action                                              | Source |
| -------- | --------------------------------------------------- | ------ |
| `/`      | Open in-file search                                 | US-513 |
| `g`      | Open grep modal (search across rule files)          | US-513 |
| `r`      | Refresh rule file list from disk                    | US-513 |
| `Enter`  | (in grep modal) open the selected match             | US-513 |
| `Esc`    | (in any rules modal) cancel                         | US-513 |

### Modal hotkeys (gate trace, kill confirm, mode confirm)

| Key      | Action                                              | Source |
| -------- | --------------------------------------------------- | ------ |
| `Esc`    | Dismiss / cancel                                    | US-507/508/514 |
| `q`      | Dismiss gate trace modal                            | US-514 |

> If a hotkey appears nowhere above, it is **not** a supported binding. Do not invent shortcuts.

---

## 3. Confirmation Flows

Every destructive action requires an explicit confirmation modal. The modal cannot be auto-dismissed — operator must press `Y`/`Enter` (proceed) or `N`/`Esc` (cancel).

### 3.1 Pause / Resume (`Space`)
- No modal. Toggles `scanner_paused` via `StateEngine.set_paused`.
- Publishes `control.pause` or `control.resume` on the event bus.
- Status strip updates to `PAUSED` / `RUNNING`.
- **Reversible**: press `Space` again.

### 3.1.1 Auto-halt Unhalt (`u`)
- No modal. Clears `halted` only after the TUI passes live-source checks.
- Required checks: LIVE runtime, OANDA connected, embedded scanner ready, heartbeat fresh, no open trades, scanner not paused, model stack complete, and model freshness not `STALE`/`CRITICAL`.
- On pass: sets `halted=false`, records `last_actor=TUI_UNHALT`, publishes `control.resume`, and appends `.claude/brain/strategic_log.md`.
- On fail: leaves `halted=true`, writes `UNHALT BLOCKED` to the brain feed, and appends the failed reasons to the strategic log.

### 3.2 Kill Switch (`k`)
- Modal: `KillModal` (red border, double-confirm).
- Modal text shows: number of open positions, mode (`DRY-RUN` / `LIVE`), reason field.
- Operator must explicitly confirm. `Esc` cancels.
- On confirm: `_kill_in_progress = True`, `ExecutionManager.flatten_all("operator_kill")` runs in a background worker, all hotkeys are blocked until completion.
- Result is a `FlattenResult` with per-position outcomes; logged to journal and surfaced in the Diagnostics tab.

### 3.3 Mode Toggle (`m`)
- Modal: `ModeModal`.
- DRY-RUN → LIVE: requires credential check (`OANDA_API_KEY`, `OANDA_ACCOUNT_ID` present and non-empty). If missing, modal shows the failure reason and refuses to flip.
- LIVE → DRY-RUN: no credential check, but the modal still requires explicit confirmation.
- On confirm: `StateEngine.set_mode`, status strip updates, event published, strategic log appended with timestamp + actor + nav + open trade count.

### 3.4 Abort Signal (`a`)
- Modal: confirms which signal will be aborted (pair, direction, age in replay buffer).
- If no pending signal exists, the action is a no-op and logs `no pending signal found in replay buffer`.

### 3.5 Per-Trade Close (`c` on Trades screen)
- Modal: `CloseTradeModal`. Shows trade ID, pair, unrealized P/L.
- On confirm: invokes `ExecutionManager.flatten_trade(trade_id, reason="operator_close")`.

---

## 4. Adjustment Inbox Workflow

The Inbox (`F2`) surfaces config-tuner proposals queued in `.claude/config_adjustments.json` (key `pending`).

**Standard workflow:**
1. Press `F2` to open the inbox.
2. Use arrow keys to highlight a proposal.
3. Press `v` to view detail (proposed value, current value, source learning, confidence).
4. Decide:
   - `a` **Approve** — moves proposal to `history` with `status="approved"`. ConfigAdjuster will apply on next `apply_adjustments()` cycle.
   - `r` **Reject** — moves proposal to `history` with `status="rejected"` and the rejection reason.
   - `s` **Snooze 24h** — defers visibility for 24h; proposal remains in `pending`.
5. Press `Esc` to close any detail modal.

**Approval validation gates (mandatory — see [.claude/rules/improvement.md](../.claude/rules/improvement.md)):**
- Confirm the `key` field matches an actual `ScannerConfig` dataclass field name. If it does not, **reject** — orphan keys silently `setattr` and never take effect.
- Confirm the proposed value sits within sane bounds (e.g., `min_confidence` ∈ [0.4, 0.9], `atr_sl_multiplier` ∈ [0.8, 3.0]).
- Confirm the source learning has at least 3 supporting observations unless flagged as catastrophic-evidence (single 10-loss-streak class).

**Round-trip verification after any approval:**
- Watch the next scan cycle in the `scanner` tmux pane.
- Confirm the live config attribute reflects the new value (Diagnostics tab → Config snapshot).

---

## 5. Kill Switch — What to Do When Hit

The kill switch is the single most consequential operator action. It liquidates every open OANDA position.

### When to use
- Drawdown guardian fires and you cannot diagnose within 60 seconds.
- Model staleness banner turns red mid-session and uncertainty is climbing.
- OANDA pricing feed reports stale bars (>5 min) on multiple pairs simultaneously.
- News-risk agent flags an unscheduled high-impact event you did not pre-clear.
- Anything you cannot explain within 90 seconds.

### When NOT to use
- A single losing position you simply do not like.  Use `c` (per-trade close) instead.
- A noisy notification you have not read.
- "I think there might be a problem." Investigate first.

### How to fire
1. Press `k` from any screen.
2. Read the modal. Confirm the open position count matches your mental model.
3. Confirm the mode (`LIVE` vs `DRY-RUN`). If `DRY-RUN`, the kill is a simulation; act accordingly.
4. Press the confirmation key.
5. Do **nothing** else until `_kill_in_progress` clears (status strip returns to normal).
6. Proceed to §9 (10-minute post-kill checklist).

---

## 6. Reading the Gate Trace

The Gate Trace modal (`GateTraceModal`) is opened from a journal entry or trade row. It shows, for one trade decision, the result of every gate in execution order.

**Columns:**
- **Gate name** — `confidence_gate`, `momentum_gate`, `risk_gate`, `correlation_gate`, `rr_gate`, `staleness_gate`, etc.
- **Passed** — `True` / `False`.
- **Score / threshold** — actual value vs configured threshold.
- **Reason** — short string when failed (e.g., `rr=0.91 below 1.20`).

**How to read it:**
- The first `passed=False` row is the proximate reason the trade was rejected (or, if all passed, the trade executed).
- A `passed=True` with a score within 0.02 of threshold is a **near miss** — flag it for the next config-tuning review.
- If `staleness_gate` is `False`, see §7. If `confidence_gate` is `False` while `staleness_gate` is `True`, the model is not stale — the signal was simply weak.
- Trend agent `passed=False` is an absolute veto for directional trades (per `.claude/rules/trading.md` 2026-04-15). If you see a trade executed with trend `passed=False` because WVS compensated, **escalate immediately** — that rule is being bypassed.

**Hotkeys in the modal:** `Esc` or `q` to close.

---

## 7. Reading the Staleness Banner

`StalenessBanner` (top strip) reflects the oldest model component age.

| Color  | Meaning                                                                 | Action                              |
| ------ | ----------------------------------------------------------------------- | ----------------------------------- |
| Green  | `max_component_age_days <= 3`                                           | Normal operation.                   |
| Amber  | `3 < max_component_age_days <= 7`                                       | Plan a retrain in next session.     |
| Red    | `max_component_age_days > 7`                                            | Hard-block on `uncertainty > 0.35`. |

When red, the system **already** tightens the uncertainty hard-block from 0.45 to 0.35 (see `.claude/rules/trading.md`). Operator action: schedule a model rebuild, and do not approve adjustment-inbox proposals that loosen the uncertainty threshold.

---

## 8. Reading the Weight Inspector

The Weight Inspector lives on the Agents screen (`F4`, `WeightInspector` widget). It is read-only.

**What it shows:**
- One row per agent (12 rows for the FX team).
- Current RL weight (read from `trained_data/models/agent_weights.json`).
- Delta vs previous snapshot (loaded from rolling history).
- Rank change indicator (↑ / ↓ / →).

**How to interpret:**
- Weights normalize across the team; a single agent rising means others fell.
- Sustained climb of `risk_sentinel` or `uncertainty` over 50+ trades indicates the system is becoming more defensive — consistent with regime change.
- Weight collapse (`< 0.02`) on any single agent is a signal that the agent is consistently wrong; queue a review.
- Do **not** edit the weights file by hand. RL sync writes the file on every trade close; manual edits are silently overwritten.

---

## 9. Incident Response — 10-Minute Post-Kill Checklist

Execute every item. In order. Do not improvise.

**T+0 to T+2 minutes**
- [ ] Confirm OANDA web dashboard shows zero open positions (cross-check the kill).
- [ ] Confirm NAV in the TUI matches OANDA web within 0.1%.
- [ ] Screenshot the Diagnostics tab and the most recent journal row.

**T+2 to T+5 minutes**
- [ ] Open the journal (`F5`) and locate the kill event row. Verify `flatten_all` returned `success=True` for every position. Any partial fills or errors → §10.
- [ ] Read the gate trace for the *last* trade that opened before the kill. Identify which gate(s) drifted (§6).
- [ ] Check the Staleness banner (§7). If red, note model ages.

**T+5 to T+8 minutes**
- [ ] Append a strategic log entry to `.claude/brain/strategic_log.md`: timestamp, NAV before/after, reason for kill, and the hypothesis.
- [ ] Update `.claude/brain/open_questions.md` with at least one question to investigate before re-enabling live mode.

**T+8 to T+10 minutes**
- [ ] Decide: stay in `DRY-RUN` until the post-mortem is written, or resume `LIVE` only after the root cause is identified and a guard added.
- [ ] If staying in DRY-RUN, press `m` to flip the mode.
- [ ] Notify on-call (§10) regardless of perceived severity.

---

## 10. On-Call Escalation Checklist

Use this when the post-kill checklist surfaces anything you cannot explain.

**Immediate (within 5 minutes of incident):**
- [ ] Page primary on-call via the standard channel.
- [ ] Provide: NAV before, NAV after, position count flattened, mode, kill reason, gate trace summary.
- [ ] Do **not** attempt remediation while waiting for callback. Stay in DRY-RUN.

**Information to have ready before the call:**
- Last 10 journal rows (Journal tab → copy block).
- Diagnostics snapshot (mode, paused state, model ages, NAV, drawdown).
- Recent learnings (`tail -30 .claude/learnings.md`).
- Any pending inbox proposals approved in the last 24h.
- A statement of what changed since the last green session (commits, config approvals, model rebuilds).

**Escalate to secondary on-call if:**
- Primary unreachable for > 10 minutes.
- The kill failed on any position (`success=False` in `FlattenResult`).
- NAV drift between OANDA web and TUI exceeds 0.5%.
- The kill required a manual close on the OANDA web UI.

**Do not re-enable LIVE mode until:**
- Root cause is identified.
- A guard (config change, rule promotion, or code patch) is in place.
- A second operator has reviewed and signed off in `.claude/brain/strategic_log.md`.

---

## Appendix A — Reality Checker Walkthrough

**Reviewer:** Reality Checker (US-519 pass)
**Date:** 2026-04-16
**Scope:** Verify every acceptance criterion of US-519 against the shipped runbook and the source code it describes.

### Walkthrough Transcript

| § | Claim under review                                    | Evidence checked                                                                                                                                                          | Result |
| - | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1 | `./buddy` sources `.env.local`, activates venv, auto-detects mode | `buddy` launcher in repo root; `CLAUDE.md` TUI Command Bridge section                                                                                                     | PASS   |
| 2 | Every binding from US-504 onward is listed             | `src/tui/app.py:494-508`, `inbox_screen.py:292-297`, `trades_screen.py:352`, `rules_screen.py:107-112,198-202`, `gate_trace_modal.py:218-221`, `kill_modal.py:24`, `mode_modal.py:41` | PASS   |
| 3 | All five destructive flows have documented modal behavior | `action_supervisor_pause` (app.py:1016), `_kill` (app.py:1075), `_mode`+`_apply_mode_change` (app.py:1168,1210), `_abort` (app.py:1280)                                  | PASS   |
| 4 | Inbox approve/reject/snooze matches `ConfigAdjuster`   | Bindings at `inbox_screen.py:292-297`; `.claude/rules/improvement.md` 2026-04-16 Config Adjustment Consumer Verification cited                                            | PASS   |
| 5 | Kill switch fire criteria + procedure                  | `ExecutionManager.flatten_all("operator_kill")` + `_kill_in_progress` guard                                                                                              | PASS   |
| 6 | Gate Trace reading rules + trend-veto escalation       | `GateTraceModal` at `src/tui/screens/gate_trace_modal.py`; `.claude/rules/trading.md` 2026-04-15 trend-veto rule                                                          | PASS   |
| 7 | Staleness banner thresholds + 0.35 hard-block          | `StalenessBanner` import at `src/tui/app.py:46`; rule from `.claude/rules/trading.md` 2026-04-15 promotion                                                                 | PASS   |
| 8 | Weight Inspector is read-only; manual edits overwritten | `WeightInspector` widget at `src/tui/screens/agents_screen.py:323-420`; RL sync rewrites `agent_weights.json` on every trade close                                        | PASS   |
| 9 | 10-minute post-kill checklist phases T+0→T+10          | Items map to existing artifacts (`.claude/brain/strategic_log.md`, `open_questions.md`, Diagnostics, Journal)                                                              | PASS   |
| 10| On-call escalation triggers + LIVE re-enable gate      | Consistent with `CLAUDE.md` execution standards; no contradictions                                                                                                        | PASS   |

### Cross-document checks

| Check                                                                                          | Result |
| ---------------------------------------------------------------------------------------------- | ------ |
| `CLAUDE.md` **Claude Brain** section references `docs/supervisor_console_runbook.md`           | PASS   |
| All relative links in the runbook resolve (`../CLAUDE.md`, `../.claude/rules/improvement.md`, `../.claude/rules/trading.md`) | PASS   |
| No external links (no flaky network checks needed)                                             | PASS   |
| Hotkey coverage: 100% — no binding in source is undocumented; no documented binding is dead    | PASS   |

### Verdict

All 10 enumerated sections are complete, accurate against source, and operationally actionable. Hotkey coverage is 100%. No broken internal links. `CLAUDE.md` references the runbook from the Claude Brain section.

**VERDICT:** PASS

**Signed off:** Reality Checker — 2026-04-16
**Next review:** on next destructive-hotkey addition (any future US past US-518) or 30-day calendar review, whichever is sooner.

---

## 12. Phase 95 — First Live-Mode Re-Enable Checklist

> Phase 95 (US-601 → US-606) added persistence and observability infrastructure on top of the Phase 91/92 Supervisor Console. This checklist is the operator-facing gate between "code shipped" and "real money trading again."
>
> **Context:** The 2026-04-24 ground-truth audit found the bot had been "LIVE" for 8 days and executed zero trades because the TUI process had died and was never restarted. Phase 95 fixes the underlying engineering gaps (no persistence, no state reconciliation, no heartbeat). This checklist proves the fixes are deployed, not just shipped.

### 12.1 Pre-flight (operator must execute)

```
[ ] 1. Stop currently-running TUI: focus tui pane, press 'q' (or kill via launchctl)
[ ] 2. Verify US-604 TUI telemetry wiring is still present (see §12.4 below)
[ ] 3. Run: bash scripts/install_launchd_service.sh
[ ] 4. Run: bash scripts/install_watchdog_service.sh
[ ] 5. Verify: launchctl list | grep com.buddy.trader     → status 0 (running)
[ ] 6. Verify: launchctl list | grep com.buddy.watchdog   → status 0 (running)
[ ] 7. Wait 60s, verify: cat .claude/heartbeat.json       → ts within last 15s, scanner_alive: true
[ ] 8. Wait 60s, verify: ls -la .claude/state_drift_log.jsonl  → file exists (drift event from initial reconcile)
[ ] 9. Wait 5 min, verify: wc -l trained_data/dry_run_validation.jsonl → line count growing
[ ] 10. Confirm state.json mode=dry_run (NOT live!)
```

### 12.2 48h validation window

```
[ ] 11. Let the bot run for 48 hours in dry_run with the heartbeat + reconciler + watchdog active
[ ] 12. Run: python scripts/analyze_dry_run.py trained_data/dry_run_validation.jsonl
[ ] 13. Confirm distribution health:
       - At least one pair-cycle row present (scanner is actually running)
       - No single block_reasons key accounts for > 95% of blocked cycles
         (would indicate over-tightened gate or broken signal path)
       - would_submit rate > 0% (bot is finding tradeable setups)
       - staleness_veto NOT dominant (would indicate stale models)
       - confidence_gate NOT > 90% of blocks (would indicate threshold mis-calibration)
```

### 12.3 LIVE re-enable

Only after ALL of §12.1 + §12.2 pass:

```
[ ] 14. Append distribution summary to .claude/ralph/reports/phase95_evidence.md
[ ] 15. Re-run architect verdict on phase95_evidence.md; require PASS / not RUNTIME_GATED
[ ] 16. In TUI: press M to open Mode toggle modal
[ ] 17. Type 'LIVE' (case-sensitive)
[ ] 18. Tab → ⚡ Go Live → Enter
[ ] 19. Verify state.json mode=live, scanner_paused=false, halted=false
[ ] 20. Confirm Trades tab (F3) updates in real-time
[ ] 21. Watch first 4h of LIVE operation closely. K (kill) is always one keystroke away.
```

### 12.4 US-604 TUI Telemetry Wiring

**Discovered during US-606 close-out, 2026-04-25.**

The `validation_stats` module was wired into `src/scanner/automation/continuous.py:533-540` (used by the CLI ContinuousScanner), but NOT into `src/tui/embedded_scanner.py` (used by the TUI `./buddy` launcher). When the bot runs via `./buddy`, the dry_run validation jsonl will accumulate ZERO rows, and §12.2 step 13 will produce empty distribution analysis.

This is the same TUI-wiring gap pattern that caused the 2026-04-16 ConfigAdjuster orphan-key incident ($3,527 loss).

**Status: PATCHED AND VERIFIED 2026-05-05.** The wiring is present in `src/tui/embedded_scanner.py::_post_scan_automation()` after the observation_log block, mirroring `continuous.py`. Current evidence: `trained_data/dry_run_validation.jsonl` has 2,304 rows across 155 scan cycles.

Current distribution remains a LIVE re-enable blocker: `would_submit=7/2304 (0.3%)` and `circuit_breaker=2297/2304 (99.7%)`, which violates the §12.2 “no single block reason > 95%” gate. Do not use `u` to clear an auto-halt until this distribution and model freshness are healthy.

If for any reason the patch is reverted or the wiring needs to be re-applied:

```python
# US-604: dry-run validation telemetry (mirror of continuous.py:529-540)
try:
    from src.scanner.automation.validation_stats import ScanDistributionStats
    if not hasattr(self, "_validation_stats"):
        self._validation_stats = ScanDistributionStats()
    self._validation_stats.record_cycle(result.analyses or [])
except Exception as _vs_err:
    logger.debug("validation_stats record error: %s", _vs_err)
```

Insert in `_post_scan_automation()` immediately after the observation_log try-block. After patching, restart TUI and verify `trained_data/dry_run_validation.jsonl` is being appended to within one scan cycle.

**Phase 96 candidate:** Refactor scanner asymmetry so all automation wiring lives in a single shared module (e.g. `src/scanner/automation/automation_pipeline.py`) imported by both `EmbeddedScanner` and `ContinuousScanner`. This permanently eliminates the wiring-gap bug class.

---

## Homework Review Workflow

The F2 Inbox now contains two streams: configuration adjustments (existing) and trade homework (new in Phase 96).

### Bootstrap your first session

If the inbox is empty, generate homework from existing journal entries:

```
python buddy_scanner.py homework --generate-batch --last 17
```

This studies the last 17 closed trades and produces ~17 entries in the inbox.

### Reviewing homework

1. Press `F2` to open the inbox.
2. The two-pane layout shows: **queue on left, detail on right**.
3. Use ↑↓ to navigate the queue. The detail pane updates automatically.
4. For each homework entry, decide:
   - **A** — approve. Buddy's analysis was right; deltas applied.
   - **R** — reject. Buddy was wrong; type a one-sentence note explaining what he missed.
   - **E** — edit. Buddy was partly right; modify the proposed deltas before applying.
   - **S** — snooze 24h. Come back later.
5. Cursor stays at same position after action — A-A-A-A through the queue at speed.

### Mental model

You are the master; Buddy is the apprentice. He does the fast pattern-matching against the heuristic catalog. You bring judgment about what matters and why. Approval = "yes, learn this." Rejection with a note = "no, you missed X" — and X becomes a candidate for a future heuristic.

---

## Document Provenance

- **Owner:** Supervisor Console maintainer.
- **Created:** 2026-04-16 (US-519).
- **Phase:** 91.
- **Source stories:** US-501 → US-518.
- **Update protocol:** any new hotkey or destructive action added in a future US must update §2 and §3 in the same PR. CI link-check (or `markdown-link-check`) must pass before merge.
