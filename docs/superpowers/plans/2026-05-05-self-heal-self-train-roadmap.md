# Self-Heal & Self-Train Closed-Loop — Cross-Session Roadmap

> **Status banner — read first.**
> **Active step:** ➜ A2 — W&B control plane (5 commits May 2 from `claude/trading-strategy-analysis-sAakL`). A1.5 closed with HONEST FINDING: per-pair val R² is BELOW the audit's 0.05-0.30 band on every pair (range -0.046 to -0.189), worse than the joint head (-0.0102). Leak fix structurally correct (0/7 leak-flagged) — the audit's R²-band prediction was overly optimistic, not a per-pair training bug. See A1.5 CHECKPOINT LOG entry for full table + analysis.
> **Last updated:** 2026-05-06 (A1.5 closed; per-pair pkls live but at noise-floor signal — system halted so no live impact; deeper Track-B-adjacent investigation queued).
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
- commit: 55a9228 docs(plans): self-heal/self-train closed-loop roadmap (Step 0 closed)
- next: A1 — cherry-pick `a761175 → 5ce3c54` (13 commits) plus orphan docs `7a65e43, 7b03e2c` for context, ON ISOLATED WORKTREE (operator green-lit cherry-pick + sub-agents); dispatching Git Workflow Master with verification gates: pytest -k confidence, smoke retrain `--pairs EUR_USD --no-promote`, val R² in 0.05-0.25 range

### 2026-05-06 ~03:?? UTC Claude — A1 closed (operator chose Option A: FF 13 picks; Q2/Q3 deferred to A1.5)
- did: rebased `heal-train/A1-confidence-leak-fix` onto `7f69b3a` (clean — 13 commits replay; new tip `77b70a7`); fast-forwarded main → 7f69b3a..77b70a7 (58 files, +14708/-218); zero overlap with 50+ unstaged WT files (verified pre-FF). Operator rationale: needs per-pair training infrastructure live; system halted so noise-floor joint R² has no live impact; per-pair pkls will be regenerated via `scripts/retrain_per_pair_confidence.py` (ships with `711722f feat(training): re-enable per-pair confidence fine-tunes + W&B observability`) which is the cleaner path than overwriting main's stale May-4 leak-version pkls. Q3 date-check earlier surfaced that main's USD_JPY.pkl was newer in mtime (May 4 via `279d486`) but older in correctness (created from leak-version code); Option A skips clobber by retraining fresh.
- commit: 13 cherry-picks landed: `9fc68e9, 451c9fc, 1a10663, 406322f, c790d28, c1ec625, e6d2474, 847c1c8, 710bd0e, 0bf4c9e, a92b37c, 711722f, 77b70a7` (post-rebase hashes; pre-rebase hashes were `ae39613..332d6c6` per worktree log)
- next: A1.5 — run `python scripts/retrain_per_pair_confidence.py` to populate per-pair confidence pkls on current main journal (replaces what the unpicked `a5f5f02` would have brought, but freshly fitted to journal-as-of-now). After A1.5: per-pair val R² should land in 0.05–0.30 band per audit prediction. Then A2 (W&B control plane).

### 2026-05-06 04:0X UTC Claude — STOP: Anthropic rate limit hit during A2 + Track B design parallel dispatch
- did: (a) A1 + A1.5 already on main (commits `ca4dee3`, `62570e9`); (b) A3 cherry-pick of `8bdccbc` direct on main → new commit `02cb006 docs(ml): label-leak audit across all 7 training heads` (label leak audit doc now lives at `docs/label_leak_audit_2026-05-05.md`, +366 lines); (c) parallel sub-agent dispatch in single message — A2 (Git Workflow Master, isolation:worktree, run_in_background=true) and Track B design (Software Architect, run_in_background=true). BOTH agents terminated within ~51s (A2) and ~0.2s (Track B) with hard rate-limit error: **"You've hit your limit · resets 1am (America/New_York)"**. Track B design doc was NOT created (0 tool uses, 0 tokens billed). A2 partial: branch `heal-train/A2-wandb-control-plane` at worktree `.claude/worktrees/agent-ad8ba06577b6894a3` (locked) got 2 of 5 W&B commits clean: `ce6dd5b feat(training): W&B control plane with versioned per-head configs` (= picked 3f84e6e), `44625fa feat(training): route per-head configs through trainers + online retrainer` (= picked cd19e5e). **CRITICAL — CROSS-TRACK CONTAMINATION:** the worktree also contains `59bb2f7 test(automation): cover DynamicSLTPOptimizer and DynamicDrawdownManager (#43)` (Raynergy-svg author, +3939 lines across coverage workflows + 8 test files) which is NOT in the operator's 5-commit A2 list (3f84e6e, cd19e5e, cda10f7, ff514c9, 567a383). Verified `59bb2f7` is NOT on main (`git merge-base --is-ancestor 59bb2f7 main` → not ancestor). This violates the operator's "DO NOT cherry-pick across tracks" guardrail. Sub-agent picked it before the rate limit cut it off — likely confused topological-order traversal of source-branch ancestry. 3 W&B commits remain unpicked: cda10f7 (manual scripts), ff514c9 (control plane unit tests + online retrainer integration), 567a383 (operator guide docs). Working tree on main = unchanged (no `M` files added by sub-agents; only the pre-existing 50+ session-noise M files I am not touching). Honest accounting: I (controller) used my own quota for the A1.5 retrain + 2 commits + the parallel dispatch; sub-agents used their portion before the limit; reset at 1am ET (~60 min from this entry's wall-clock).
- commit: A1.5 = `62570e9 docs(plans): A1.5 closed`; A3 = `02cb006 docs(ml): label-leak audit across all 7 training heads`. A2 partial state lives in worktree only (not on main); Track B design not started.
- next: ScheduleWakeup at +3600s (post-1am ET reset). On wake-up: (1) re-verify disk state (halted=true must hold); (2) decide A2 contamination recovery — recommend `git rebase --onto main 59bb2f7 heal-train/A2-wandb-control-plane` to drop the bad commit while preserving the 2 W&B picks (cheap), OR redo from scratch (safer if rebase produces conflicts); (3) cherry-pick remaining 3 W&B commits + verify pytest + verify pull_config(head) for 7 heads; (4) FF main; (5) dispatch fresh Track B design (Software Architect, foreground this time to fail fast if quota issues recur); (6) continue down operator's autonomous-run plan from "Hour 1-2: A3 + B-track design" through Track B fixes B1/B2/B3/B4 + Track C1 investigation. Operator's stop conditions still apply; rate-limit-hit is a 5th condition I'm adding to the list (escalate to operator if it recurs after reset, since it indicates broader account-level constraint).

### 2026-05-06 03:43 UTC Claude — A1.5 closed (per-pair pkls retrained; ALL 7 PAIRS BELOW R² BAND)
- did: ran `python scripts/retrain_per_pair_confidence.py` (background, 11s wall-time — way under the 10-30 min estimate because the per-pair fits are LightGBM regressors on ~3200 train / ~920 val each, not the full multi-head joint pipeline). 7/7 pairs trained successfully, 0/7 leak-flagged (no R² > 0.30 ceiling — leak fix structurally correct). Per-pair pkls saved to `trained_data/models/{INST}/ridge_confidence.pkl` for EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD. W&B logged offline (`WANDB_MODE=online` not set). Report at `trained_data/per_pair_confidence_report.json`. **CRITICAL FINDING — per-pair R² is BELOW the audit's expected 0.05-0.30 band on EVERY pair, and BELOW the joint head's -0.0102:** EUR_USD=-0.0868 (n_real=343), GBP_USD=-0.1235 (n_real=40), USD_JPY=-0.0462 (n_real=40), USD_CHF=-0.1888 (n_real=40), AUD_USD=-0.0637 (n_real=40), USD_CAD=-0.1320 (n_real=40), NZD_USD=-0.1570 (n_real=40). MAE range 35.25-37.91 (target scale [20,95]). Negative R² means models perform worse than predicting the y mean. **Root-cause hypothesis (not validated):** per-pair lost the joint's data-pooling advantage — joint trained on ~36k blended samples (across 14 pairs); per-pair gets ~3200 each, with only 40 real journal labels for 6 of 7 pairs (EUR_USD got 343 real because it has the most journal trades). The audit's prediction (per-pair lifts R² into 0.05-0.30) assumed per-pair specialization wins; this run shows data-pooling beats specialization at current sample sizes. Critically: **the leak was hiding that the realized-outcome confidence target may not be tractable at all from the 24-feature ridge_features set** — joint at -0.0102 and per-pair at -0.05 to -0.19 both bracket noise floor, suggesting the supervised confidence task as defined has very weak signal. This is a Track-B-adjacent finding (rethink confidence head label or feature set), but NOT an A1.5 blocker — A1.5 only required: (a) per-pair pkls present (yes), (b) leak-detection passes (yes), (c) report written (yes). Honesty protocol calls: I am NOT claiming per-pair models are useful — gates.py will consume near-random confidence scores from these pkls. System is halted, so no live impact. Operator must decide whether to escalate (deeper signal investigation) or proceed to A2 and circle back later.
- commit: A1.5 doc-only update (this entry + status banner + briefing handoff). Per-pair pkl artifacts written under `trained_data/models/{INST}/ridge_confidence.pkl` (NOT git-tracked artifacts; live on disk).
- next: **A2 — W&B control plane** (5 commits May 2: `3f84e6e, cd19e5e, cda10f7, ff514c9, 567a383`). Independent of confidence-head signal quality — lands per-head config infrastructure that Tracks B/C/D will use. Operator may want to insert an A1.6 investigation sub-checkpoint instead — see new "Operator decisions still open" entry A1.5.Q1.

### 2026-05-06 03:1? UTC Claude — A1 partial (13 of 15 commits clean, 2 data-file conflicts aborted)
- did: dispatched Git Workflow Master sub-agent (`isolation: worktree`) → cherry-picked 13 of 15 commits clean onto `heal-train/A1-confidence-leak-fix` (worktree at `.claude/worktrees/agent-a3e2d8e04e7938adb`, tip `332d6c6`, locked). Verified: 13 commits in topological order via `git log main..HEAD`; diff scope = 58 files, +14708/-218 (matches "code + artifacts + retrain reports + rollback backups" expectation). 2 conflicts aborted per protocol — `e63abe9` (journal backfill) on `trained_data/trade_journal_rl.json` data-file divergence (newer journal entries landed on main since May 1); `a5f5f02` (per-pair pkls + calibration v3) on binary `trained_data/models/USD_JPY/ridge_confidence.pkl`. **Both conflicts are data-file divergence, NOT code conflicts.** Code verification: realized-outcome label generator at `src/training/labels/realized_confidence_label.py:199-262` is structurally honest (forward-looking journal join + triple-barrier fallback at `lookahead_bars=24, sl_atr_mult=1.0, tp_atr_mult=2.5`); call site at `modular_data_loaders.py:3438-3446` invokes it; old closed-form code is gone; rescaling at `:3471-3474` preserves gates.py 0-100 contract. **pytest: 63/63 PASSED** on confidence-related tests (test_confidence_calibration + test_confidence_integration + test_per_pair_confidence_finetune + test_wandb_integration), exit 0. **Joint head val R² post-fix: -0.0102** (verified directly in `trained_data/models/joint/joint_training_meta.json:metrics.confidence.r2_score`); expected_r2_band=[0.05, 0.30]; BELOW band because joint pools only ~700 trades across 14 pairs — per-pair R² would be the correct evidence but lives in the unpicked `a5f5f02`. **Pre-fix R² was 0.9971 — the leak collapse from 0.99 → -0.01 confirms the audit's directional prediction.** Smoke retrain SKIPPED (no `--no-promote` flag in `scheduled_retrain.py` on this branch, only `--dry-run`; post-fix R² already documented in artifact metadata; would duplicate the retrain that produced `a8f6b53`).
- commit: 13 cherry-picks live on feature branch `heal-train/A1-confidence-leak-fix` @ `332d6c6` (NOT on main; awaiting operator decision on (Q1) fast-forward, (Q2) re-pick `e63abe9` script-only, (Q3) re-pick `a5f5f02` binary overwrite)
- next: **OPERATOR DECISIONS BLOCKING A1 CLOSURE.** See "Operator decisions still open" table for new entries A1.Q1, A1.Q2, A1.Q3.

---

## Operator decisions still open

These need an answer before specific checkpoints can start. Tracked here so they don't get lost.

| Decision | Required by | Default if not answered |
|---|---|---|
| Cherry-pick vs squash-merge for Track A | A1 | ✅ ANSWERED 2026-05-06: cherry-pick by sub-track (preserves rollback granularity) |
| **A1.Q1** — fast-forward 13/15 to main NOW, or hold A1 until per-pair fine-tunes (`a5f5f02`) also land? | A1 closure | ✅ ANSWERED 2026-05-06: Operator chose Option A (FF now). Main HEAD `77b70a7` → `ca4dee3`. |
| **A1.Q2** — re-pick `e63abe9` script-only and re-run `scripts/backfill_ridge_features.py` against current main journal? | A1 closure | ⏸ DEFERRED 2026-05-06 — may revisit if backfill becomes load-bearing. |
| **A1.Q3** — re-pick `a5f5f02` binary overwrite of main's USD_JPY.pkl? | A1 closure | ⏸ SUPERSEDED 2026-05-06 — A1.5 fresh per-pair retrain replaces this path; pkls regenerated from current data. |
| **A1.5.Q1** — per-pair R² across the board is at/below noise floor, audit prediction wrong. Escalate to A1.6 deeper investigation, or proceed to A2 and circle back later? | A2 start | trade-off: investigate now → discover whether realized-outcome confidence target is tractable at all (could lead to a Track-B-adjacent label/feature redesign) but blocks Track A landing. Proceed to A2 → land W&B control plane infra (independent of signal quality), then revisit. **Recommendation:** proceed to A2. The negative R² doesn't BREAK anything new (system is halted; gates.py would consume near-random scores anyway with joint at -0.01); investigation can be queued as a Track-B-adjacent checkpoint after Track A lands. Counter-recommendation: if operator believes per-pair noise floor proves the confidence head as defined is fundamentally broken, escalate now and consider replacing the head with a calibration-only layer (heuristic confidence → calibration → score). |
| Track A4 (parallel scan) — merge now or defer? | end of Track A | defer (independent feature, not on the autonomy critical path) |
| Track B fix path per head (audit lists 2-3 options each) | B1, B2, B3, B4 | option (a) realized-outcome label per head — most honest, matches confidence-head fix pattern |
| Track D — gate behind env var or ship enabled? | D1 | env var, OFF default; flip after 1 week of soak observation |
| Auto-unhalt gate condition #6 (virtual-scan check) — include or not? | D1 | include — strongest deterministic gate short of soak mode |
