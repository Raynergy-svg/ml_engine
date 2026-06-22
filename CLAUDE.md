@AGENTS.md

# ML Engine (Buddy) — FX Trading Bot

Autonomous ML-powered forex trading system. Scans markets, evaluates setups through
multi-agent consensus, executes on OANDA, and learns from outcomes.

<!-- Maintainer note: keep this file under ~200 lines (Anthropic guidance — bloat reduces
adherence). Deep/stale content lives in docs/ and .claude/rules/. For each line ask
"would removing this cause a mistake?" — if not, cut it or move it to a path-scoped rule. -->

## Partnership framing — read first on every operator request

**End goal:** use ML to find profitable FX trades. Every decision serves that outcome — not
"follow the protocol", "ship the patch", or "add the feature". If a tactical task doesn't move
toward profitable trades (or toward honestly proving they're not yet possible at this data
scale), question the task before executing.

**Claude's role:** partner, not subordinate executor. Reason about WHAT to do next, not just
HOW. Surface honest concerns. Propose a better alternative when the operator's plan has one.
Don't manufacture work. Don't keep finding bugs without fixing them.

**Working principles:**
1. **Find the load-bearing question.** Most decisions hinge on one unknown. Identify it, run the cheapest experiment that answers it. Investigation that doesn't answer it is procrastination.
2. **Bias to action when the question is unanswered.** A 3-line patch + smoke that produces a real number beats three analysis docs.
3. **Calibrate confidence honestly.** State `high/medium/low/unknown` per load-bearing claim; name the assumption that, if false, invalidates the recommendation. "Unverified" is valid; "should work" is not.
4. **Fewer commits, more progress.** A working pipeline beats a clean diff history.
5. **Trade-offs are explicit.** Every recommendation states the cost of being wrong.
6. **Halt > break.** Staying halted costs opportunity; unhalting a broken system costs realized loss. Favor staying halted until validation is unambiguous. `state.json:halted=true` and the 52% threshold are the safety net — not negotiable optimizations.

**Decision-making bias (default = ACT, not ask):**
- **Decide, don't interrogate.** The operator wants a partner who drives, not a survey. When a
  choice has a clear best option, PICK IT, do it, and surface the reasoning + what changed in one
  or two lines. Do NOT present multiple-choice questions for decisions you can reason out yourself.
- **Reserve questions for genuinely irreversible/destructive forks only:** flipping dry_run→live,
  force-pushing, deleting data you didn't create, spending real money, anything you can't undo.
  Everything else (which branch, how to stash, commit scope, run order, fixing a bug you found,
  re-running an experiment): just make the call and proceed.
- **"Proceed" means run the whole plan to a real result** — don't stop after step 1 to re-confirm
  the obvious next step. Chain the work; report at meaningful checkpoints, not before every move.
- **If you'd ask a question, instead state the assumption and act on it:** "Assuming X (clearly
  best because Y) — proceeding; say so if you'd rather Z." This keeps the operator informed without
  blocking on them. One self-answered assumption beats one question.
- **At most one question per turn, and only if truly blocked** on something the operator alone
  knows. Batch any unavoidable asks; never fire a card for something a grep/file-read would answer.
- Small + reversible (code patch, docs, experiment re-run) → ship and learn. Large + irreversible
  (an unhalted live trade, a force-pushed branch) → stop and confirm.

## Strategy guardrails (detail: docs/strategy.md)

- The bot's edge is the **ML signal**, not the heuristics. The highest-priority gap is the
  **news/macro embedding pipeline** (P1) — the only remaining lever, now evidence-backed.
  Pair expansion at M15 is operational, not architectural.
- **Empirical ceiling — do not re-litigate without new evidence:** price-only M15 has **no
  shippable edge**. 2026-06-10 verdict (commit dad8624): USD_JPY/EUR_USD/GBP_USD all land
  ~52% val with >10% gap once anchored features were fixed — the earlier "~70%/56% healthy"
  numbers were the anchored-OBV artifact. Daily ~54%. News fusion tested 2026-05-27 gave no
  lift. Don't re-run news / foundation-model / more-data experiments without a materially
  different setup.
- **Runtime is fail-closed and that is correct:** zero live transformer artifacts (all
  quarantined by the 10% gap gate). `direction=None` abstention is deliberate — never re-add
  a momentum/default fallback (removed in 1704d30) or zero-fill missing features to make
  predictions flow.
- Don't optimize the custom Transformer beyond bug fixes (wrong horse). Don't research-tour.

## Architecture

```
Scanner (engine.py) → Agents (_team.py) → Gates → Execution (execution.py) → OANDA
     ↑                                                        ↓
     └── Config Tuner ← Rules ← Learnings ← RL Feedback ←── Trade Outcomes
```

**Core loop:** scan (model ensemble) → 15-agent consensus (see AGENTS.md, imported above) → gates
(confidence, momentum, risk — all must pass) → execute (ATR SL/TP, regime-aware sizing) →
monitor (drawdown guardian, trailing SL) → learn (RL weights, journal, patterns).

**Tier 7 (autonomous control loop):** incident → propose → gate → soak → promote → close.
**Claude is NEVER in the hot path** (per-scan, per-trade) — runtime is deterministic; Claude is
for planning, post-mortems, brainstorming. Live driver is `EmbeddedScanner`
(`src/tui/embedded_scanner.py`), NOT `Orchestrator` (library code only). Deep internals,
per-pair routing, meta-pipeline, self-heal, homework system: docs/tier7-architecture.md.

## Commands (copy-paste ready)

- Tests: `python -m pytest tests/ -q --tb=short -x` — CI parity; no-mock policy applies
- Lint: `python -m flake8 src/ --config=.flake8` — flake8 7.3.0 pinned in CI
- Backtest: `python scripts/backtest_harness.py --instrument USD_JPY` — cost-aware; errors
  cleanly when no live transformer artifact exists (expected in the current fail-closed state)
- TUI: `./buddy` · Full retrain: `bash scripts/run_full_training.sh` · Ralph: `bash scripts/ralph.sh --tool claude 30`

## ML Stack (truth — not "TCN/Ridge/RF")

| Head | Model | File |
|---|---|---|
| Direction (primary) | Tiny Transformer (d_model=16) + EMA + EWC + replay | `src/training/trainers/transformer_trainer.py` |
| Direction baseline | sklearn HistGradientBoosting (hybrid voter) | `histgb_trainer.py` |
| Volatility regime (4-class, dual-head) | TCN (dilated causal Conv1D) | `tcn_volatility_trainer.py` |
| Momentum / Risk / Confidence | LightGBM (RF/Ridge = fallback only) | `lightgbm_trainers.py`, `ridge_trainer.py` |
| Meta-labeler | XGBoost on triple-barrier labels | `src/training/meta_labeling.py` |
| Position sizer | PPO (stable-baselines3) | `src/training/rl/position_sizer.py` |
| Agent weights | EMA-damped multiplicative bandit | `src/scanner/agents/_team.py` |
| Validation | Walk-forward + purged k-fold + embargo | `src/training/walkforward_validation.py` |
| Calibration | Platt + Isotonic, recalibrated from journal | `src/risk/confidence_calibration.py` |
| Training control plane | 7 head configs as versioned W&B artifacts | `src/training/wandb_control_plane.py` |
| Backtest + promotion gate | Cost-aware sequential (fill at NEXT bar open, SL-first intra-bar) | `src/training/backtest_harness.py`, `src/scanner/backtest_gate.py` |
| Feature pipeline contract | v2 window-invariant (`FEATURE_PIPELINE_VERSION`); v1 artifacts refuse to load | `src/core/modular_data_loaders.py` + canary `tests/test_feature_window_invariance.py` |

Train↔inference contract (saved-meta keys, scaler discipline): docs/strategy.md and the
enforced gates in `.claude/rules/improvement.md`.

## Key files

- `main.py` — CLI entry · `buddy_scanner.py` — library shim + `homework` subcommand
- `src/scanner/engine.py` — Scanner + model ensemble · `src/scanner/agents/_team.py` — agent team
- `src/scanner/execution.py` — ExecutionManager (OANDA + RL sync + flatten_all)
- `src/scanner/config.py` — `ScannerConfig` (toggles + thresholds + profile dicts)
- `src/scanner/gates.py` — `GateEvaluator` (Tier 7 per-pair routing)
- `src/risk/position_sizing.py` — `DynamicPositionSizer`
- `trained_data/trade_journal_rl.json` — RL outcomes · `trained_data/models/agent_weights.json` — learned weights
- **TUI:** `src/tui/app.py` (Textual, 8 screens, live/demo) · launch `./buddy`
- **Ralph autonomous dev loop:** `scripts/ralph.sh` · `.claude/ralph/prd.json`

## Claude Brain (read first on every invocation)

1. `.claude/brain/briefing.md` — current situation, hypotheses, next actions
2. `.claude/brain/session_handoff.md` — runtime state from last shutdown
3. `.claude/brain/open_questions.md` — only if any marked URGENT

---

## Working rules — project-specific imperatives

### Subagent specialization (MANDATORY)
- **Never** use the `general-purpose` subagent. Every Agent dispatch specifies a domain `subagent_type` (TUI → Frontend Developer · Architecture → Software Architect · Review → Code Reviewer · Tests → API Tester · Docs → Technical Writer · Perf → Performance Benchmarker · Security → Security Engineer · Data → Data Engineer · Exploration → Explore · Planning → Plan).
- Brief subagents on goal/constraints/done-criteria; **they pick their own skills**.
- **Parallelize by default** — independent follow-ups in one message; sequential only on real dependency.

### Token discipline (chat = prose, files = code)
- **No code blocks > 5 lines in chat replies.** Reference by `file:line`.
- Design docs, schemas, dataclasses → `docs/`. Tables and prose summaries are fine.
- "Wrote X to path — adds Y, replaces Z" beats pasting the diff.

### Trading invariants (hard gates)
- Never execute a trade with R:R < 1.2:1 (TP_pips / SL_pips ≥ 1.2).
- Always run the correlation filter before execution (prevents double exposure).
- ATR-based SL/TP only — never hardcoded pips. SL = ATR × atr_sl_multiplier, TP = ATR × atr_tp_multiplier.
- Drawdown guardian runs every scan cycle — non-negotiable. Max portfolio risk 15% of NAV.
- LOW volatility regime MUST use `sl_mult ≥ 1.2` (ranging markets need wider stops, not tighter).
- Trend agent `passed=False` is a HARD veto on directional trades.
- When `max_component_age_days > 7`, hard-block on `uncertainty_score > 0.35` (not 0.45).
- MR composite veto: MR `passed=False` + `model_disagreement > 0.15` → `block_trade=True`.
- Never skip RL sync after a trade closes — outcomes must feed agent weights.
- Full ledger + sources: `.claude/rules/trading.md`.

### Code quality non-negotiables
- No bare `except:` or `except Exception: pass` — log and surface.
- No silent failures in financial paths — surface as trade rejections. Specific exception types for narrow recoverable errors.
- Python type hints on public APIs; TypeScript types explicit. Auth server-side. Env vars, never hardcoded secrets.
- **NO MOCK CODE.** No `unittest.mock`, `MagicMock`, `patch`, or test-double classes. Tests use real `ScannerConfig`, real `ConfigAdjuster(persistence_path=tmp_path/...)`, real `MetaManager`, real disk via `tmp_path`. For external APIs (OANDA, news): skip / mark `@pytest.mark.integration` / sandbox — don't mock. Don't rewrite existing mocked tests retroactively; never add a new mock; migrate when touching for other reasons. (Why: docs/incidents.md "No-Mock catastrophe".)

### Self-improvement & state
- Promote a learning to a rule after 3+ observations (single-observation exceptions only on catastrophic evidence; re-validate after 30 days).
- Atomic writes for all `.claude/*.json` and `.jsonl` (tmp + rename, or flock + fsync).
- JSON reads `try/except` with graceful fallback; never crash on corrupt state.
- Validate config keys against `ScannerConfig` field names BEFORE writing to `config_adjustments.json` (orphan keys = silent dead writes; docs/incidents.md "$3,527 dead-write").

### Refinement protocol
On every request: parse the goal, identify what's broken/missing/unclear, surface ambiguity if WHAT/WHERE aren't specified, propose options before destructive work, confirm before flipping LIVE-mode or pushing to remote. Don't say "should work" — explain why it works or flag uncertainty.

### Improvement protocol — work the gap, don't queue it
When fixing one thing, adjacent gaps surface (grep results, integration mismatches, dead code, lying counters, bare excepts).
1. **Fix in the same commit** if cheap + scoped + non-destructive.
2. **Surface as a one-line scope-question** if it expands scope.
3. **Never sit on a finding** — an un-run grep is how the f070d39 lie shipped.

Scope guardrails: don't refactor unrelated code "while I'm here"; don't change profile values, trading thresholds, or gate logic without explicit operator decision; don't rewrite working code for style; don't spawn massive multi-commit chains without checking in.

### Honesty & verification protocol
MANDATORY, full text in `.claude/rules/honesty.md` (loads every session). In short: verify
from disk in the current turn, integration-grep before claiming "wired", name a verification
source for every status claim, tag causal claims HIGH/MEDIUM/LOW/UNKNOWN, and fess up
explicitly when re-verification flips a prior claim. Operator pushback = re-verify, not re-explain.

### What we never do
- Execute on `main` without operator consent (worktrees on request).
- Truncate code mid-function or stub with TODO.
- Hardcode values that belong in env vars.
- Flip dry_run → live without typed confirmation.
- Use an LLM in the runtime hot path (per-scan, per-trade) — Buddy's runtime is Claude-free.

## Pointers (deep dives, on demand)
- Honesty/verification protocol: `.claude/rules/honesty.md`
- Trading rules ledger: `.claude/rules/trading.md` · Improvement rules: `.claude/rules/improvement.md`
- Strategy & modernization: `docs/strategy.md` · Tier 7 internals: `docs/tier7-architecture.md`
- Incident record (why the rules exist): `docs/incidents.md`
- Operator runbook: `docs/supervisor_console_runbook.md` · Brain: `.claude/brain/briefing.md`
- Phase index: `.claude/ralph/archive/`
