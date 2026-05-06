# Self-Heal & Self-Train Closed-Loop — Cross-Session Roadmap

> **Status banner — read first.**
> **Active step:** ➜ A1 — cherry-pick May 1 confidence-head leak fix (Step 0 closed).
> **Last updated:** 2026-05-06 (Step 0 commit; A1 dispatched to Git Workflow Master sub-agent in isolated worktree).
> **Next session must:** read this file top-to-bottom before any code or merge.

---

## Why this exists

The bot is halted and 19 days stale. Manual unhalt yesterday → one trade → loss → re-halt. Operator goal:
**deterministic self-heal + self-train closed loop. No LLM in the runtime hot path.**

Plan spans multiple sessions. Sessions die, context resets. This file is the durable plan; chat memory is not.

## Three rules every session in this workstream must follow

1. **Plan lives on disk.** Read this file first. Update it last (status banner + checkpoint log at bottom).
2. **One checkpoint per session, max.** Commit when the checkpoint is done. If session dies mid-checkpoint, leave the working tree as-is — next session resumes from `git status`.
3. **Briefing.md is the handoff baton.** End of each session: update `.claude/brain/briefing.md` "Next Actions" to name the *specific* checkpoint to start with.

---

## Current blockers (verified 2026-05-05)

| Blocker | Source on disk |
|---|---|
| `state.json:halted=true` since 2026-05-05T01:57Z | `.claude/state.json` |
| Last validated `trained_at`: 2026-04-16T16:00Z (19 days stale) | retrain log meta + `logs/retrain_*.log` |
| Holdout has rejected every retrain since Apr 16 (~44% across H1/M15/H4) | `logs/retrain_20260505_214428.log` tail |
| 4 of 7 training heads have **confirmed structural label leakage** | `docs/label_leak_audit_2026-05-05.md` (on branch `claude/trading-strategy-analysis-sAakL`, NOT on main) |
| Confidence-head leak fix exists, sits on branch, not merged | 13 commits on `claude/trading-strategy-analysis-sAakL` from May 1 |
| W&B per-head config control plane exists, sits on branch, not merged | 5 commits on same branch from May 2 |
| No auto-unhalt path anywhere in production code | grep `set_halted(False)` — sole hit is TUI hotkey at `app.py:2060` |
| Halt does NOT trigger immediate retrain (cooldown 12h, schedule Mon/Fri) | `autonomous_trainer.maybe_spawn_autonomous_retrain` |

## Heads classified (from `docs/label_leak_audit_2026-05-05.md`)

| Head | Verdict | Implication |
|---|---|---|
| Direction (Transformer + HistGB) | CLEAN | 44% holdout is the head's *true* accuracy — separate bug from leaks |
| Risk `expected_drawdown_pct` | CLEAN | — |
| Meta-labeler (XGBoost triple-barrier) | CLEAN | — |
| Trend/chop/MR regime (TransformerRegime) | SUSPICIOUS | label gates on `adx`/`rsi`/`zscore_20`, all in X; future-return only confirms |
| Volatility regime (TCN, 4-class) | **CONFIRMED LEAK** | label = percentile rank of `atr_pct_14`, also in X |
| Momentum `momentum_score` | **CONFIRMED LEAK** | label = mean(\|returns\|), `returns_*` in X |
| Momentum `acceleration` | **CONFIRMED LEAK** | label fully feature-derivable |
| Risk `streak_prob` | **CONFIRMED LEAK** | label uses `volatility_10/volatility_20`, both in X |

---

## The plan — 4 tracks, dependency order

> Each track is multi-session. Each *checkpoint* should fit in one session.

### Track A — Land the existing branch onto main
**Source branch:** `origin/claude/trading-strategy-analysis-sAakL` (27 commits ahead of main as of 2026-05-06; 25 map to A1–A4, 2 are orphan pre-A1 docs commits `7a65e43` + `7b03e2c` riding with A1 for context coherence).
**Strategy:** cherry-pick by sub-track in dependency order, validate after each.

| Checkpoint | Cherry-pick range | Verification |
|---|---|---|
| **A1** — confidence-head leak fix | 13 commits May 1: `a761175 → 5ce3c54` | `pytest tests/training/ -k confidence`; smoke `python scripts/scheduled_retrain.py --pairs EUR_USD --no-promote`; verify confidence head val R² in 0.05-0.25 range (per audit prediction) |
| **A2** — W&B control plane | 5 commits May 2: `3f84e6e, cd19e5e, cda10f7, ff514c9, 567a383` | `pytest tests/training/test_wandb_control_plane.py`; verify `pull_config(head)` works for all 7 heads with seeded defaults |
| **A3** — label-leak audit doc | 1 commit May 5: `8bdccbc` | Doc-only, no test. Verify `docs/label_leak_audit_2026-05-05.md` exists on main |
| **A4** — parallel scan (defer or merge) | 6 commits May 5: `70d6676 → ec2a470` | Independent feature. Operator decides timing — not a blocker for autonomy work |

**Track A done = main has confidence fix + W&B control plane + audit doc. Parallel scan separate decision.**

---

### Track B — Apply the 4 confirmed leak fixes
**Source:** none yet — fixes don't exist; audit only documents the leaks.
**Strategy:** one head per checkpoint. Per-head approach proposed in audit; operator confirms before patch.

| Checkpoint | Head | Audit-proposed fix paths |
|---|---|---|
| **B1** | Volatility regime (TCN) | (a) replace label with realized future-window vol, (b) drop `atr_pct_*` from X for this head, (c) replace head with rule-based binner |
| **B2** | Momentum `momentum_score` | (a) replace label with realized forward return / risk-adjusted return, (b) drop `returns_*` from X for this head |
| **B3** | Momentum `acceleration` | derived from `momentum_score` — fix or remove jointly with B2 |
| **B4** | Risk `streak_prob` | (a) replace label with realized loss-streak count from journal, (b) drop `volatility_10/20` from X for this head |

**Per-checkpoint workflow:** patch label generator → retrain affected head → run holdout → verify accuracy delta is *honest* (significantly lower than pre-fix; per-pair accuracy variance reasonable) → commit.

**Track B done = all 4 confirmed-leak heads retrained on honest labels, holdout passes ≥52%, journal-validated.**

---

### Track C — Investigate direction head's 44% holdout
**Independent of Tracks A/B.** Direction head is CLEAN per audit, yet scores below coin-flip across 3 timeframes. Real bug somewhere.

Hypotheses to check (in order of likelihood):

| Checkpoint | Hypothesis | Verification |
|---|---|---|
| **C1** | Train/inference feature-normalization mismatch | Diff feature stats: training-time scaler params vs inference-time scaling. Look at `modular_data_loaders.py:Feature scaling: max=10.00, mean_abs=0.5521` (training) vs runtime feature pipeline output stats |
| **C2** | Lookahead/temporal-leak bug in label or sequence construction | Trace `load_direction_data` (`modular_data_loaders.py:1722-1957`); confirm sequence at index `i` uses only bars `[i-seq_len, i)` and label uses bar `i+lookahead` only |
| **C3** | Distribution shift since training window | Compare training-window feature distributions vs current 30d window across `atr_pct_*`, `returns_*`, `adx`, `rsi` |
| **C4** | Sign-inversion (label or prediction polarity flip) | Build deterministic probe: train one model with labels intact, one with labels sign-flipped, compare holdout. Don't auto-promote — alert only. |

**Track C done = root-cause known + fix shipped + direction holdout ≥52%.**

---

### Track D — Closed-loop autonomy plumbing
**Depends on Tracks A + B + C being landed and direction holdout passing.** Building autonomy onto a broken foundation is the bug we're trying NOT to repeat.

| Checkpoint | What | Files | Approx LOC |
|---|---|---|---|
| **D1** | Auto-unhalt subscriber + strict conjunctive health gate | `cycle_autonomy.py`, `state_engine.py` | ~80 |
| **D2** | Halt-triggered retrain (force=True, 30min floor) | `engine.py:_maybe_auto_halt_on_loss_streak`, `autonomous_trainer.py` | ~30 |
| **D3** | Soak mode after auto-unhalt (raise `min_confidence + 0.05`, halve risk for first 5 trades) | `engine.py`, `state_engine.py`, new `.claude/soak_state.json` | ~60 |
| **D4** | Rejection-pattern detector (count consecutive holdout fails; route incident on N≥3 at <50%) | `scheduled_retrain.py`, MetaManager intake | ~40 |

**Track D done = bot can wake itself up safely, retrain on halt, soak before scaling up, escalate stuck rejection patterns.**

---

## Auto-unhalt strict conjunctive gate (frozen design — Track D1)

For posterity. All MUST be true to set_halted(False) without operator action:

1. `state.json:halted == true` (no point unhalting an unhalted bot)
2. `TRAINING_COMPLETED` event since last halt timestamp
3. `trained_at` advanced past last halt timestamp (i.e., holdout actually passed, not just attempted)
4. All open trades flat (existing `_evaluate_unhalt_health` rule)
5. Heartbeat fresh (`ts_iso` within 15s)
6. (optional, operator-confirms) ≥N successful virtual-scan cycles since retrain — gate-rejection log shows new model produces non-degenerate signals

Any failure = stay halted, log specific reason. No "any of" disjunctions.

---

## Do-NOT list (guardrails for any session in this workstream)

- ❌ DO NOT auto-unhalt before Tracks A + B + C are landed and verified.
- ❌ DO NOT auto-promote a label-flipped (sign-inverted) model. Track C4 is a *probe*; result alerts the operator only.
- ❌ DO NOT lower the holdout threshold (52%) to make rejected models accept. The threshold is the only thing keeping bad models out of production.
- ❌ DO NOT cherry-pick across tracks (e.g., grabbing W&B-control-plane commits before confidence-fix commits) — dependency order matters; W&B control plane references the fixed labels.
- ❌ DO NOT add LLM calls to the runtime path. Self-heal is deterministic. Period.
- ❌ DO NOT skip writing a checkpoint-log entry at the bottom of this file when finishing a checkpoint. The log IS the cross-session memory.

---

## Verification commands (canonical, copy-paste ready)

```bash
# State
cat .claude/state.json | python -c "import json, sys; s=json.load(sys.stdin); print('halted:', s['halted'], '| mode:', s['mode'], '| last:', s['last_updated'])"

# Heartbeat
cat .claude/heartbeat.json | python -c "import json, sys; h=json.load(sys.stdin); print('alive:', h['scanner_alive'], '| ts:', h['ts_iso'], '| pid:', h['pid'])"

# Holdout history
ls -lt logs/retrain_*.log | head -5
tail -30 $(ls -t logs/retrain_*.log | head -1) | grep -E "Hold-out|REJECTED|trained_at|RETRAINING"

# Branch deltas
git log main..origin/claude/trading-strategy-analysis-sAakL --oneline
git log --since="3 days ago" --pretty=format:"%h %ad %s" --date=short main

# Verification: which leaky heads are still on main?
grep -n "atr_pct_14" src/core/modular_data_loaders.py | head -5  # volatility regime label uses this
grep -n "compute_volatility_regime" src/core/modular_data_loaders.py | head -3
```

---

## Rollback procedures per track

**Track A rollback:** the source branch has rollback manifests already (`docs/plans/retrain_report_*` + `chore(rollback)` commits). Cherry-pick the matching `chore(rollback)` to revert artifacts; `git revert <fix-commit>` for code.

**Track B rollback:** each leak-fix commit must include a backup of the pre-fix model artifacts under `trained_data/rollback/<head>_pre_<date>/`. Restore via `cp -r trained_data/rollback/<head>_pre_<date>/* trained_data/models/joint/`.

**Track C rollback:** all C-checkpoints are read-only investigations until C4 patch lands. C4 patch must be reversible via `git revert`.

**Track D rollback:** every autonomy feature must be gated by an env var (`BUDDY_AUTO_UNHALT_ENABLED`, etc.) defaulting to OFF. Rollback = unset env var. No code changes needed in emergency.

---

## CHECKPOINT LOG — append, never edit prior entries

> Format per entry:
> `## YYYY-MM-DD HH:MM ACTOR — checkpoint id`
> `- did: <one-liner>`
> `- commit: <hash or N/A>`
> `- next: <next checkpoint id>`

### 2026-05-05 22:5? UTC Claude — Step 0 (roadmap drafted)
- did: drafted this roadmap; documented 28-commit unmerged branch; classified all 7 heads from leak audit; froze auto-unhalt strict gate design
- commit: N/A (doc-only, awaiting operator review before commit)
- next: operator reviews → commit roadmap → start checkpoint A1

### 2026-05-06 03:0? UTC Claude — Step 0 closed
- did: cold-resume verification (disk-first per honesty protocol); reconciled roadmap vs disk → 28-vs-27 commit-count drift surfaced (sub-track sums = 25; 2 orphan pre-A1 docs commits `7a65e43 docs(ml): audit training architecture` + `7b03e2c docs(claude): correct ML stack description` will ride with A1 for context coherence); confirmed halted=true, scanner_alive (pid 61892, cycle_count 4 post-Ctrl+R), trained_at pinned at 2026-04-16T16:00Z, latest holdout 43.8/44.5/43.8% all REJECTED; committed roadmap + briefing handoff atomically (staged by name only, did NOT touch the 50+ unrelated M files in the working tree)
- commit: see `git log --grep "Step 0"` (this commit; HEAD at time of write)
- next: A1 — cherry-pick `a761175 → 5ce3c54` (13 commits) plus orphan docs `7a65e43, 7b03e2c` for context, ON ISOLATED WORKTREE (operator green-lit cherry-pick + sub-agents); dispatching Git Workflow Master with verification gates: pytest -k confidence, smoke retrain `--pairs EUR_USD --no-promote`, val R² in 0.05-0.25 range

---

## Operator decisions still open

These need an answer before specific checkpoints can start. Tracked here so they don't get lost.

| Decision | Required by | Default if not answered |
|---|---|---|
| Cherry-pick vs squash-merge for Track A | A1 | cherry-pick by sub-track (preserves rollback granularity) |
| Track A4 (parallel scan) — merge now or defer? | end of Track A | defer (independent feature, not on the autonomy critical path) |
| Track B fix path per head (audit lists 2-3 options each) | B1, B2, B3, B4 | option (a) realized-outcome label per head — most honest, matches confidence-head fix pattern |
| Track D — gate behind env var or ship enabled? | D1 | env var, OFF default; flip after 1 week of soak observation |
| Auto-unhalt gate condition #6 (virtual-scan check) — include or not? | D1 | include — strongest deterministic gate short of soak mode |
