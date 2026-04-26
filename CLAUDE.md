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

### Refinement protocol (compact)
On every operator request: parse the goal, identify what's broken/missing/unclear, surface ambiguity if WHAT/WHERE aren't specified, propose options before executing destructive work, confirm before flipping LIVE-mode or pushing to remote. Don't say "should work" — either explain why it works or flag uncertainty.

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
