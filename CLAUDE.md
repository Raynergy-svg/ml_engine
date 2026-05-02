# ML Engine (Buddy) — FX Trading Bot

Autonomous ML-powered forex trading system. Scans markets, evaluates setups through multi-agent consensus, executes on OANDA, and learns from outcomes.

## Architecture
```
Scanner (engine.py) → Agents (agents.py) → Gates → Execution (execution.py) → OANDA
     ↑                                                        ↓
     └── Config Tuner ← Rules ← Learnings ← RL Feedback ←── Trade Outcomes
```

## Core Loop
1. **Scan** — multi-pair analysis with TCN/Ridge/RF ensemble models
2. **Agents** — 15-agent team (truth: `_BASE_WEIGHTS` in `src/scanner/agents/_team.py`)
   - Core 12: trend, mean_reversion, volatility, risk_sentinel, uncertainty, execution_quality, momentum, news_risk, multi_timeframe, pair_performance, session_timing, support_resistance
   - Extended 3: order_flow (0.95), trader_readiness (0.50), devil_advocate (1.30, runs LAST). All toggleable via `enable_*_agent` flags in `ScannerConfig`.
3. **Gates** — confidence, momentum, risk; all must pass
4. **Execute** — ATR-based SL/TP, regime-aware position sizing
5. **Monitor** — drawdown guardian, trailing SL, real-time P/L
6. **Learn** — RL weight updates, trade journal, pattern extraction

## Tier 7 Autonomous Architecture (current 2026-05-01)

Closed control loop: **incident → propose → gate → soak → promote → close**. Runtime is deterministic; **Claude is NEVER in the hot path** (per-scan, per-trade). Claude is for planning, post-mortems, brainstorming only.

### Runtime entry (single source of truth)
- `src/bootstrap/env.py:ensure_runtime_env()` — called by `main.py:34` AND `src/tui/__main__.py:7`. Idempotent (re-init under `os.execv` Ctrl+R no-ops via marker attribute).
- `scripts/init.sh` sources `.env.local` + `.env.local.toggles` (meta-pipeline flags live here).
- `logs/buddy_debug.log` — every `logger.*` call, plain text, rotated 50MB×3. **First place to look** for any "did X fire?" question.

### TUI runtime path (NOT Orchestrator)
- `src/tui/embedded_scanner.py:EmbeddedScanner` is the live scanner driver. `Orchestrator` exists in `src/scanner/automation/orchestrator.py` but is **library code only — never instantiated in `src/tui/`** (`grep "Orchestrator(" src/tui/` = 0 matches; commit f070d39 documented this lie).
- `EmbeddedScanner.run_one_cycle()` halt-checks via `StateEngine().get_halted()` early-return.
- `_maybe_route_to_meta_per_cycle()` ships per-cycle diagnostics to MetaManager.
- `_write_brain()` tees every brain-feed line to `.claude/brain/feed.jsonl`.
- Ctrl+R via `os.execv` preserves state (state.json `safe_restart` beacon).

### Tier 7 per-pair gate routing (commit 649bd3d)
- `GateEvaluator(use_per_pair_routing=True)` — auto-enabled by Scanner when ANY per-pair training subdir exists in `trained_data/models/{PAIR}/`.
- `_get_pair_evaluator(instrument)` builds lazy per-pair sub (cached). Each sub: own model_dir → own catboost/xgboost/lightgbm momentum, ridge confidence, RF/lightgbm risk, transformer, meta-labeler. Shares parent's TCN volatility regime (single source of truth).
- Joint dir = fallback (correlation-threshold-dropped pairs: USD_JPY, EUR_GBP, EUR_JPY).
- Disable via `ScannerConfig.disable_per_pair_gate_routing=True`.
- Aligns gates with `modular_inference._get_model_path` (was already per-pair-first; gates was the holdout).

### Auto-halt loop (production-fired 2026-04-30 15:54:30)
- AlertManager surfaces `consecutive_losses` alert.
- Engine `_maybe_auto_halt_on_loss_streak()` triggers when value ≥ `auto_halt_consecutive_loss_threshold` (default 5).
- Calls `StateEngine.set_halted(True)` + routes `meta_manager.intake(kind="auto_halt_loss_streak")` → ChangePackage in inbox.
- Live evidence: `logs/buddy_debug.log` 15:54:30 — `meta_manager.intake change_id=467c350af5f0 kind=auto_halt_loss_streak`.

### Meta-pipeline (deterministic, no-LLM hot path)
- `MetaManager.intake(change_id, kind, payload)` — entry point. Throttle via `_concurrent_count()` (narrows to actively-executing stages); 2h orphan TTL prevents deadlock.
- `DeterministicSurgeon` (commit 3692463) — proposer, generates concrete config deltas WITHOUT LLM. Closes the `use_llm=False` black hole.
- `cycle_autonomy.py` — honors no-LLM as **hard kill** on Claude fallback (commit 3124a5c).
- `Constitution` (C1–C7 mapped to real `ScannerConfig` fields) — `policy_check` stage.
- `StagedDeployer.advance` — `pending → policy_check → deployed_shadow → deployed_canary → deployed_live → closed`. Soak gates: `shadow_cycles`, `canary_trades`.
- `MetaManager.drain()` — the actual stage-advancer. **Only call site in TUI runtime is `_approve_meta_packages` in inbox_screen.py.** No other drainer wired.

### F2 Inbox (operator approval surface)
- Filters: `[All] [📚 Homework] [🔧 Adjustments] [🧠 Meta]` (entry_type-keyed).
- `action_approve_all` runs three loops (homework, adjustments, meta) + calls `_PRODUCTION_MGR.drain()` inline so packages advance immediately.
- `_read_meta_packages()` reads `.claude/meta/changes/*.json` for live ChangePackage state.

### Self-heal subsystem
- `src/scanner/feedback/self_heal.py` — handlers keyed by action_type. 12h debounce per action (`.claude/self_heal_debounce.json`).
- `_handle_reset_gate_threshold(gate)` writes properly-shaped history entries to `.claude/config_adjustments.json["history"]`.
- `src/scanner/feedback/diagnostics.py` — gate-overtightening trap detection + schema mismatches.
- `AdjustmentApprover._save_approved` has shrink-guard tripwire (refuses writes that would shrink history; logs the proposed payload).

### Verification surfaces (canonical, priority order)
1. `logs/buddy_debug.log` — every `logger.*` call.
2. `.claude/brain/feed.jsonl` — F1 brain feed mirror (Rich markup stripped).
3. `.claude/heartbeat.json` — TUI alive marker (pid, cycle_count, scanner_alive, ts_iso ≤ 15s = alive).
4. `.claude/state.json` — halted, mode, scan_cycle_count, safe_restart beacon.
5. `.claude/meta/changes.jsonl` + `.claude/meta/changes/*.json` — meta ledger + per-package source-of-truth.
6. `.claude/alert_state.json` — AlertManager state.
7. `trained_data/virtual_trades.jsonl` + `trained_data/trade_journal_rl.json` — gate-rejected scans + closed trade outcomes.

### Tier 7 key files
- `src/bootstrap/env.py` — runtime env init
- `src/tui/embedded_scanner.py` — live scanner driver
- `src/scanner/automation/meta_manager.py` — MetaManager + throttle
- `src/scanner/automation/deterministic_surgeon.py` — no-LLM proposer
- `src/scanner/automation/cycle_autonomy.py` — no-LLM policy enforcement
- `src/scanner/automation/staged_deployer.py` — shadow → canary → live
- `src/scanner/automation/constitution.py` — C1–C7 gates
- `src/scanner/feedback/self_heal.py` + `diagnostics.py` — auto-correction
- `src/scanner/gates.py` — `GateEvaluator` with Tier 7 per-pair routing
- `scripts/cybernetic_smoke.py` + `cybernetic_promote.py` — operator validation tools

## Key Decisions
- Soft uncertainty blocking (confidence penalty) over hard circuit breaker
- ATR-based dynamic SL/TP, never hardcoded pips
- Correlation filter prevents double exposure on correlated pairs
- Minimum R:R 1.2:1 gate before execution
- Position sizing scales to account size (5% base risk on practice)

## Claude Brain (Read First on Every Invocation)
1. `.claude/brain/briefing.md` — current situation, hypotheses, next actions
2. `.claude/brain/session_handoff.md` — runtime state from last shutdown
3. `.claude/brain/open_questions.md` — only if any marked URGENT

Other brain files (read on demand):
- `trade_narrative.md` — interpreted trade history
- `strategic_log.md` — append-only decision ledger
- `docs/supervisor_console_runbook.md` — required reading before LIVE mode

## Self-Improvement (Buddy's Mechanical Layer)
- `.claude/learnings.md` — date-stamped insights from outcomes
- `.claude/rules/` — promoted patterns that gate behavior
- `.claude/state.json` — session continuity
- `.claude/config_adjustments.json` — adaptive parameter tuning

## Trade Homework System (Phase 96 — apprenticeship workbench)
Buddy is a **student** doing supervised study of past trades. Closed trades become homework; operator grades each via F2 Inbox; corrections become RL training signal.

- Closed trades → `HomeworkGenerator` (heuristic-driven, **NO LLM call**) → `.claude/homework_pending.jsonl`
- F2 Inbox: two-pane (queue + live detail). Filters: `[All] [📚 Homework] [🔧 Adjustments]`. Hotkeys: V/A/R/E/S
- Approve/edit → `TrainingSignal` → `TrainingSignalApplicator` writes deltas to `agent_weights.json` atomically (Phase 96.5 closes the loop)
- Bootstrap: `python buddy_scanner.py homework --generate-batch --last N`
- Heuristic catalog: `src/scanner/automation/homework/heuristics.py` (~25 patterns / 6 categories: A Setup, B Risk, C Consensus, D Execution, E Context, F Meta)
- Spec: `docs/superpowers/specs/2026-04-25-trade-homework-system-design.md`

## Key Files
- `main.py` — CLI entry point
- `buddy_scanner.py` — library shim + `homework` subcommand only
- `src/scanner/engine.py` — Scanner class with model ensemble
- `src/scanner/agents/_team.py` — `ScannerAgentTeam` (15 agents + RL learning)
- `src/scanner/execution.py` — `ExecutionManager` (OANDA + RL sync + flatten_all)
- `src/scanner/config.py` — `ScannerConfig` (toggles + thresholds + profile dicts)
- `src/scanner/automation/` — 125+ modules; `continuous.py` = watch loop, `orchestrator.py` = run_cycle
- `src/scanner/automation/homework/` — homework subsystem (types, store, heuristics, generator, reviewer, applicator, journal_adapter)
- `src/risk/position_sizing.py` — `DynamicPositionSizer` + regime-aware factories
- `trained_data/trade_journal_rl.json` — outcomes for RL
- `trained_data/models/agent_weights.json` — learned weights (regime-keyed)

## TUI
- `src/tui/app.py` — Textual TUI (8 screens, dual-mode live/demo)
- `src/tui/theme.tcss` — cyberpunk TCSS (neon cyan/magenta/green on void black)
- `src/tui/data_provider.py` — thread-safe OANDA bridge
- `buddy` — launcher (auto-sources `.env.local`, activates venv)
- Launch: `./buddy` (auto-detects `--live` if OANDA creds exist, else `--demo`)

## Ralph (Autonomous Dev Loop)
- `scripts/ralph.sh` — iterative AI agent loop for PRD stories. Routes complex stories (≥7 ACs or async/benchmark keywords) to Opus, others to Sonnet.
- `.claude/ralph/prd.json` — active PRD; archives in `.claude/ralph/archive/`
- `.claude/skills/prd/` and `.claude/skills/ralph/` — PRD skills

---

## Working Rules — Project-Specific Imperatives

### Subagent specialization (MANDATORY)
- **Never** use `general-purpose` subagent. Every Agent dispatch must specify a domain `subagent_type`.
- TUI/Frontend → `Frontend Developer` · Architecture → `Software Architect` · Code review → `Code Reviewer` · Tests → `API Tester` · Docs → `Technical Writer` · Performance → `Performance Benchmarker` · Security → `Security Engineer` · Data → `Data Engineer` · DevOps → `DevOps Automator` · Codebase exploration → `Explore` · Planning → `Plan`
- **Subagents always pick their own skills.** Brief them on the goal/constraints/done-criteria; do NOT prescribe which Superpowers skill to use. They invoke whatever skills (TDD, debugging, requesting-code-review, etc.) the task warrants. Controller's job is goal+context, not method.
- **Parallelize by default.** Independent follow-ups dispatch in a single message with multiple Agent blocks. Sequential only when there's a real dependency.

### Token discipline (chat = prose, files = code)
- **No code blocks > 5 lines in chat replies.** Reference by file:line.
- Design docs, schemas, dataclasses → `docs/`. Chat replies link.
- Tables and prose summaries are fine (they compress information). Code blocks usually don't.
- "Wrote `X` to `path/to/file` — adds Y, replaces Z" beats pasting the diff.

### Trading invariants
- Never execute a trade with R:R < 1.2:1 (hard gate before submission)
- Always run correlation filter before execution (prevents double exposure on correlated pairs)
- ATR-based SL/TP only — never suggest hardcoded pip values
- Drawdown guardian runs every scan cycle — non-negotiable
- LOW regime `sl_mult >= 1.2` (Phase 91 promoted rule — ranging markets need wider stops)
- Trend agent `passed=False` is a hard veto on directional trades (Phase 91)
- Staleness uncertainty tightens to 0.35 when `oldest_age_days > 7` (Phase 91)
- MR composite veto: MR `passed=False` + `model_disagreement > 0.25` → block_trade=True (Phase 93)
- Never skip RL sync after a trade closes — outcomes must feed agent weights

### Self-improvement & state
- Promote a learning to a rule after 3+ observations (single-observation exceptions allowed only on catastrophic evidence; re-validate after 30 days)
- Atomic writes for all `.claude/*.json` and `.jsonl` files (`tmp + rename` or `flock` + `fsync`)
- JSON reads must `try/except` with graceful fallback; never crash on corrupt state files
- Validate config keys against `ScannerConfig` field names BEFORE writing to `config_adjustments.json` (orphan keys = silent dead writes)

### Code quality non-negotiables
- No bare `except:` or `except Exception: pass` — log and surface
- No silent failures in financial paths — surface as trade rejections
- Specific exception types, not `Exception`, for narrow recoverable errors
- TypeScript types must be explicit; Python type hints on public APIs
- Auth checks server-side, never trust client-side
- Environment variables, never hardcoded secrets
- **NO MOCK CODE.** No `unittest.mock`, `MagicMock`, `patch`, or test-double classes. Tests must use real `ScannerConfig`, real `ConfigAdjuster(persistence_path=tmp_path / "x.json")`, real `MetaManager`, real disk via `tmp_path`. Reason: 38 mocked tests passed while production wired `StagedDeployer` without `config_adjuster` — 11 packages went shadow→canary→live with zero actual config mutation. Mocks hid the integration gap. Going-forward rule (don't rewrite existing mocked tests retroactively, but never add new mocks; migrate when touching for other reasons).

### Refinement protocol (compact)
On every operator request: parse the goal, identify what's broken/missing/unclear, surface ambiguity if WHAT/WHERE aren't specified, propose options before executing destructive work, confirm before flipping LIVE-mode or pushing to remote. Don't say "should work" — either explain why it works or flag uncertainty.

### Honesty & verification protocol — MANDATORY

Caught lying once (2026-04-30 commit f070d39 incident). This cannot happen again. Hard rules — no exceptions:

**Origin of the lie**: Shipped commit f070d39 claiming "orchestrator routes per-cycle diagnostics through meta-pipeline". Unit tests on `Orchestrator._maybe_route_to_meta` passed (mocked dependencies). I told the operator the routing was wired and live. Reality: `grep "Orchestrator(" src/tui/` returns nothing — the TUI never instantiates Orchestrator, so the routing was dead code in the runtime path. Unit-test pass ≠ integration. Operator's "are you sure??" forced re-verification, only then did the gap surface.

**Verification rule (every status claim, every "wired" claim)**:
1. **Disk first**. Read the actual file/log/artifact in the current turn. Memory of an earlier tool call is NOT verification — files change, processes restart, hooks rewrite.
2. **Memory second**. Use `mem-search` / `get_observations` (claude-mem) to check prior observations on the same component. Skip rediscovery if a prior observation already answers it.
3. **Integration grep before "wired"**. Before saying "X fires from Y" or "wired into Y": `grep "<callable>(" src/<entry-point>/`. No instantiation found = NOT wired. Tests prove the class works in isolation; greps prove the path is reachable.
4. **Code-on-disk vs code-running**. The running process has whatever code was on disk when the process started. After a commit, state which generation is in the running process. "Fixed in commit X" ≠ "fix is live" if the process predates the commit.
5. **No "should work"**. Verify, or say "unverified" and stop the chain until you can.

**Unified verification surfaces (always check these, in order)**:
- `logs/buddy_debug.log` — every `logger.*` call from any module in the live process. Plain text, grep-friendly, rotated at 50MB. **First place to look** for any "did X happen?" question.
- `.claude/brain/feed.jsonl` — exact mirror of what the operator sees in the F1 brain feed (TUI rendering events, Rich markup stripped). One line per `_write_brain` call.
- `.claude/heartbeat.json` — TUI alive marker, ticks every 10s, includes `pid`, `cycle_count`, `scanner_alive`, `ts_iso`. `ts_iso` within 15s of "now" = TUI alive.
- `.claude/state.json` — runtime state (`halted`, `mode`, `scan_cycle_count`, `safe_restart` beacon).
- `.claude/meta/changes.jsonl` — meta-pipeline event ledger (one line per stage transition).
- `.claude/meta/changes/*.json` — full ChangePackage state (per-package source-of-truth for `revert_by_id`).
- `.claude/alert_state.json` — AlertManager state (consecutive_losses, drawdown, win_rate_drop, weight_instability).
- `trained_data/virtual_trades.jsonl` — per-pair gate-rejected setups (one line per scan per pair, includes raw_confidence + gate_failures).
- `trained_data/trade_journal_rl.json` — closed trade outcomes.

**Honesty rule (every status report, every checklist response)**:
- Each claim names its verification source: file path / grep query / mem observation ID. No source named = claim not made.
- No cheerful summary language ("loop is closed", "fully verified", "everything's wired") unless every component has a named source above.
- When operator challenges, treat as a calibration signal — re-verify from scratch, don't restate.
- Distinguish "shipped to disk" from "running in process". Always state which.
- Skipped verifications must be named explicitly and queued as next actions.

### Improvement protocol — work the gap, don't queue it

When investigating or fixing one thing, adjacent gaps WILL surface (grep results, integration mismatches, dead code paths, lying counters, silent bare-excepts). Default behavior:

1. **Fix in the same commit** if cheap + scoped + non-destructive. Examples that fit: fixing `n_master_pairs: 0` liar while landing the staleness honesty fix; fixing the "No momentum model available" holdout warning while shipping Tier 7 per-pair routing; fixing the alert log format `getattr(_alert, "type")` typo while wiring auto-halt.
2. **Surface as one-line scope-question** if expanding scope: "noticed: X is broken adjacent to this — fix as part of this?" Operator answers, proceed once known.
3. **Never sit on a finding**. If a grep result reveals a real bug adjacent to current work, don't note it for "later" — that's how `f070d39` happened (the integration gap was visible in a single grep but went un-run).

Scope guardrails (what NOT to do autonomously):
- Don't refactor unrelated code "while I'm here"
- Don't change profile values, trading thresholds, or gate logic without explicit operator decision
- Don't rewrite working code for style — only fix what's broken
- Don't spawn massive multi-commit chains without checking in

Examples of the protocol working tonight (2026-04-30):
- While fixing Inbox approve-all (Bug A: meta handler), found Bug B (no drainer in TUI runtime) → fixed both in same commit `5ae61d7`
- While verifying `f070d39`, found integration grep would have caught the lie → added the verification protocol rule in `88ecb52`
- While diagnosing config_adjustments state-loss, found `_load_approved` had bare-except + no shrink guard → fixed both in `7da0470`
- While shipping Tier 7 per-pair routing, surfaced that the same fix closes the holdout "No momentum model available" warning

### What we never do
- Execute on `main` without operator consent (worktrees on request)
- Truncate code mid-function or stub with TODO
- Hardcode values that belong in env vars
- Flip dry_run → live without typed confirmation
- Use LLM in the runtime hot path (per-scan, per-trade) — Buddy's runtime is Claude-free; Claude is for planning, post-mortems, and brainstorming only

---

## Pointers (deep dives, only when needed)
- Operator runbook: `docs/supervisor_console_runbook.md`
- Phase index: `.claude/ralph/archive/` (chronological)
- Trading rules ledger: `.claude/rules/trading.md`
- Improvement rules: `.claude/rules/improvement.md`
- Brain index: `.claude/brain/briefing.md` (current situation always at top)
