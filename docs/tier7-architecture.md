# Tier 7 Autonomous Architecture (reference)

> Relocated from CLAUDE.md (2026-06-09). Deep internals with commit/line references that
> rot over time. CLAUDE.md keeps only the load-bearing invariants; read this for detail.
> Verify any commit hash / file:line here against current code before relying on it
> (per `.claude/rules/honesty.md`).

Closed control loop: **incident → propose → gate → soak → promote → close**. Runtime is
deterministic; **Claude is NEVER in the hot path** (per-scan, per-trade). Claude is for
planning, post-mortems, brainstorming only.

## Runtime entry (single source of truth)
- `src/bootstrap/env.py:ensure_runtime_env()` — called by `main.py` AND `src/tui/__main__.py`. Idempotent (re-init under `os.execv` Ctrl+R no-ops via marker attribute).
- `scripts/init.sh` sources `.env.local` + `.env.local.toggles` (meta-pipeline flags live here).
- `logs/buddy_debug.log` — every `logger.*` call, plain text, rotated 50MB×3. First place to look for any "did X fire?" question.

## TUI runtime path (NOT Orchestrator)
- `src/tui/embedded_scanner.py:EmbeddedScanner` is the live scanner driver. `Orchestrator` (`src/scanner/automation/orchestrator.py`) is **library code only — never instantiated in `src/tui/`** (`grep "Orchestrator(" src/tui/` = 0 matches; the f070d39 incident documented this lie).
- `EmbeddedScanner.run_one_cycle()` halt-checks via `StateEngine().get_halted()` early-return.
- `_maybe_route_to_meta_per_cycle()` ships per-cycle diagnostics to MetaManager.
- `_write_brain()` tees every brain-feed line to `.claude/brain/feed.jsonl`.
- Ctrl+R via `os.execv` preserves state (state.json `safe_restart` beacon).

## Tier 7 per-pair gate routing
- `GateEvaluator(use_per_pair_routing=True)` — auto-enabled by Scanner when ANY per-pair training subdir exists in `trained_data/models/{PAIR}/`.
- `_get_pair_evaluator(instrument)` builds a lazy per-pair sub (cached): own model_dir → own catboost/xgboost/lightgbm momentum, ridge confidence, RF/lightgbm risk, transformer, meta-labeler. Shares parent's TCN volatility regime (single source of truth).
- **Joint dir is DEPRECATED (2026-05-12 operator directive).** Per-pair routing is the only supported runtime path. The joint fallback logs `DEPRECATED joint fallback` WARNING every time it fires; removal sequence + status tracked in `.claude/rules/improvement.md` "Joint Fallback Deprecation Gates". Pairs without per-pair `transformer_direction.keras` are headed for an engine-startup filter that drops them from `active_pairs`.
- `modular_ensemble` + `joint_gates` are excluded from `model_freshness.get_model_freshness_for_pairs` rollup — audit-only context, never gate unhalt.
- Disable via `ScannerConfig.disable_per_pair_gate_routing=True` (legacy escape hatch).

## Auto-halt loop (production-fired 2026-04-30)
- AlertManager surfaces a `consecutive_losses` alert.
- Engine `_maybe_auto_halt_on_loss_streak()` triggers when value ≥ `auto_halt_consecutive_loss_threshold` (default 5).
- Calls `StateEngine.set_halted(True)` + routes `meta_manager.intake(kind="auto_halt_loss_streak")` → ChangePackage in inbox.

## Meta-pipeline (deterministic, no-LLM hot path)
- `MetaManager.intake(change_id, kind, payload)` — entry point. Throttle via `_concurrent_count()`; 2h orphan TTL prevents deadlock.
- `DeterministicSurgeon` — proposer, generates concrete config deltas WITHOUT LLM (closes the `use_llm=False` black hole).
- `cycle_autonomy.py` — honors no-LLM as a **hard kill** on any Claude fallback.
- `Constitution` (C1–C7 mapped to real `ScannerConfig` fields) — `policy_check` stage.
- `StagedDeployer.advance` — `pending → policy_check → deployed_shadow → deployed_canary → deployed_live → closed`. Soak gates: `shadow_cycles`, `canary_trades`.
- `MetaManager.drain()` — the actual stage-advancer. Only TUI call site is `_approve_meta_packages` in `inbox_screen.py`.

## F2 Inbox (operator approval surface)
- Filters: `[All] [📚 Homework] [🔧 Adjustments] [🧠 Meta]` (entry_type-keyed).
- `action_approve_all` runs three loops (homework, adjustments, meta) + calls `_PRODUCTION_MGR.drain()` inline so packages advance immediately.
- `_read_meta_packages()` reads `.claude/meta/changes/*.json` for live ChangePackage state.

## Self-heal subsystem
- `src/scanner/feedback/self_heal.py` — handlers keyed by action_type. 12h debounce per action (`.claude/self_heal_debounce.json`).
- `_handle_reset_gate_threshold(gate)` writes properly-shaped history entries to `.claude/config_adjustments.json["history"]`.
- `src/scanner/feedback/diagnostics.py` — gate-overtightening trap detection + schema mismatches.
- `AdjustmentApprover._save_approved` has a shrink-guard tripwire (refuses writes that would shrink history).

## Key files
- `src/bootstrap/env.py` · `src/tui/embedded_scanner.py` · `src/scanner/automation/meta_manager.py` · `deterministic_surgeon.py` · `cycle_autonomy.py` · `staged_deployer.py` · `constitution.py` · `src/scanner/feedback/{self_heal,diagnostics}.py` · `src/scanner/gates.py` · `scripts/cybernetic_smoke.py` + `cybernetic_promote.py` (operator validation tools).

## Trade Homework System (Phase 96 — apprenticeship workbench)
Closed trades become homework; operator grades each via F2 Inbox; corrections become RL signal.
- Closed trades → `HomeworkGenerator` (heuristic, **NO LLM**) → `.claude/homework_pending.jsonl`
- Approve/edit → `TrainingSignal` → `TrainingSignalApplicator` writes deltas to `agent_weights.json` atomically.
- Heuristic catalog: `src/scanner/automation/homework/heuristics.py` (~25 patterns / 6 categories). Spec: `docs/superpowers/specs/2026-04-25-trade-homework-system-design.md`.
