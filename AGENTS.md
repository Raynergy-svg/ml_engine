# ML Engine — Scanner Agent Team

## 12 Specialist Agents

Weighted voting system where each agent evaluates one aspect of a trade setup. Agents emit verdicts that combine into a weighted vote score. If the score falls below a regime-aware threshold, the trade is blocked.

| Agent | Base Weight | Default | Purpose |
|-------|-------------|---------|---------|
| trend | 1.15 | ON | SMA crossover + ADX trend strength |
| mean_reversion | 0.90 | ON | RSI-based pullback/extension detection |
| volatility | 1.00 | ON | ATR + regime scoring (LOW/NORMAL/HIGH/EXTREME) |
| risk_sentinel | 1.25 | ON | Drawdown ratio + portfolio risk check |
| uncertainty | 1.10 | ON | Confidence variance + model disagreement |
| execution_quality | 1.05 | ON | Spread, slippage, liquidity assessment |
| momentum | 1.05 | ON | MACD histogram + rate-of-change alignment |
| news_risk | 0.95 | OFF | Headline keyword scanning (NFP, CPI, FOMC) |
| multi_timeframe | 1.10 | OFF | H1/H4/D1 confluence from aggregated candles |
| pair_performance | 0.85 | OFF | Historical win rate per pair |
| session_timing | 0.80 | OFF | Forex session overlap awareness |
| support_resistance | 1.00 | OFF | Swing pivot S/R proximity scoring |

## RL Weight Learning

Weights adapt from trade outcomes via `update_weights_from_outcome()`:
- Agent voted FOR + trade won: weight += 0.10
- Agent voted FOR + trade lost: weight -= 0.15
- Agent voted AGAINST + trade won: weight -= 0.05
- Agent voted AGAINST + trade lost: weight += 0.075

Weights bounded [0.1, 2.0]. Decay toward baseline each scan cycle.

Learned weights persist in `trained_data/models/agent_weights.json`.

## .claude/agents/ Directory

Contains 37 LLM personality prompts (from agency-agents repo) for Claude Code sessions. These are NOT the scanner agents above — they're reference material for engineering, testing, and strategy roles.

## Ralph (Autonomous Dev Loop)

`scripts/ralph.sh` — Spawns fresh AI instances to implement PRD stories iteratively. PRD tracked in `.claude/ralph/prd.json`.


<claude-mem-context>
# Memory Context

# [ml_engine] recent context, 2026-05-04 9:09pm EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (18,484t read) | 1,119,754t work | 98% savings

### Apr 30, 2026
1052 12:13p 🔵 GateEvaluator instrument Param Already Plumbed to evaluate_confidence — But Used Only for One-Hot Encoding, Not Model Loading
1053 " 🔵 GateEvaluator Locks model_dir to joint/ at Construction — No Dynamic Per-Pair Loading Possible Without Refactor
1057 " 🔵 modular_inference Has Two-Layer Model Resolution: Path Fallback Chain + Metadata-Based Lookup
1058 " 🔵 evaluate_all_gates Internals: instrument Passed to evaluate_confidence Only, evaluate_momentum and evaluate_risk Are Instrument-Blind
1054 12:14p 🔵 evaluate_all_gates Already Receives instrument=pair at engine.py:2501 — Gap is Entirely Inside GateEvaluator
1059 12:17p 🟣 GateEvaluator Gets _pair_evaluators Cache — First Code Change of Tier 7 Per-Pair Gate Refactor
1060 12:22p 🟣 GateEvaluator._get_pair_evaluator() Implemented — Core of Tier 7 Per-Pair Gate Routing
1061 12:35p 🔵 Full runtime state cross-section — 2026-04-30T16:53 UTC
1062 " 🔵 consecutive_losses alert firing at value=16.0 every ~hour without reset
1063 12:55p 🔵 Tier 7 per-pair routing confirmed firing in PID 51337 — 15 per-pair subdirs detected
1064 " 🔵 buddy_debug.log stale since 12:54 — markdown-it DEBUG spam drowned scanner output
1065 " 🔵 Joint/ GateEvaluator parent loaded with "momentum: none" — per-pair sub-evaluator construction not logged
1066 " 🔵 STALENESS HARD BLOCK firing — all models scoring 0.000 HOLD, scans pre-blocked before gate evaluation
1067 " 🔵 All model loads failing with [Errno 9] Bad file descriptor — system-level FD error, not missing files
### May 1, 2026
1068 8:22p ✅ CLAUDE.md: "Work the Gap" Improvement Protocol Codified
1069 " 🟣 Tier 7 Per-Pair Gate Routing Implemented and Live
1070 " 🟣 Unified Debug Log: All Logger Calls to logs/buddy_debug.log
1071 " 🔴 Inbox Approve-All: Meta Packages Silently Ignored + Drain Never Called
1072 " 🟣 Brain Feed Teed to .claude/brain/feed.jsonl for External Verification
1073 " 🔴 EmbeddedScanner.run_one_cycle Reads Halted Flag
1074 " ✅ CLAUDE.md Hard Honesty + Verification Protocol (Post f070d39 Lie)
1075 " 🔴 Training Staleness Telemetry and n_master_pairs Counter Fixed
1076 " 🔴 EmbeddedScanner Routes Per-Cycle Diagnostics to Meta-Pipeline
1077 " 🟣 Auto-Halt on Consecutive Loss Streak with Meta Incident Routing
1078 " 🔴 MetaManager _concurrent_count Narrowed to Actively-Executing Stages
1079 8:26p 🔵 Meta Pipeline End-to-End Confirmed Live: Full Stage Progression in brain/feed.jsonl
1080 " 🔴 config_adjustments Runtime State Protection: Shrink-Guard Tripwire + .gitignore
1081 " 🔴 Retrain Crash: dict/dataclass Contract Mismatch in Correlation Transfer
1082 " 🔴 Trade Flow Unblocked: Relaxed min_confidence + Orphan Timeout in Meta Config
1083 " 🟣 Ctrl+R State-Preserving Safe-Restart via os.execv
1084 " 🟣 Brain Log Tail-Watches Meta-Pipeline Ledger for Live Stage Visibility
1085 " 🟣 Phase 1: Deterministic Surgeon Closes use_llm=False Black Hole
1086 " 🟣 Phase 1: Single-Source-of-Truth Bootstrap via ensure_runtime_env()
1087 " 🔴 Self-Heal Gate-Overtightening Trap: Debounce + Diagnostics + schema-fix
1088 " 🔴 Root Cause Fix: Default to Correlation-Transfer Training (Not Joint Multi-Pair)
1089 " 🔴 Security + Core Fixes: Pickle→JSON Migration, Missing Features Dedup, NameError/Warning Spam
1090 " 🔵 Commit Verification Pass: All 30 Recent Commits Confirmed Wired in ml_engine
1092 8:30p ✅ CLAUDE.md: "Tier 7 Autonomous Architecture" Section Added (2026-05-01)
1093 " 🔵 Deep Re-Verification Batch 1: drain() Confirmed, Per-Cycle Meta Route Has No Live Log Signal
S320 Deep re-verification of all 30 recent ml_engine commits — confirm each is genuinely wired in codebase and runtime, not just surface-claimed (May 1 at 8:30 PM)
S321 Commit verification audit (30 commits) — discovered and fixed critical meta-pipeline open-circuit: ConfigAdjuster missing from StagedDeployer production build (May 1 at 8:41 PM)
S322 Commit verification audit (30 commits) — fixed meta-pipeline open-circuit (ConfigAdjuster missing from StagedDeployer), then codified "no mock code" rule across CLAUDE.md, MEMORY.md, and feedback_no_mocks.md (May 1 at 8:42 PM)
S323 Fix failing regression test `tests/test_meta_pipeline_real_actuation.py` proving ConfigAdjuster→StagedDeployer autonomous loop wiring (May 1 at 8:48 PM)
S324 Fix failing regression test `tests/test_meta_pipeline_real_actuation.py` — constructor injection to share pending/approved paths between ConfigAdjuster + AdjustmentApprover (May 1 at 8:52 PM)
S325 Close the autonomous loop final yard — wire ConfigAdjuster into StagedDeployer so live ScannerConfig mutations actually apply (May 1 at 8:53 PM)
S326 Commit the autonomous loop final yard fix — ConfigAdjuster wired into StagedDeployer (commit 57f9aa8) (May 1 at 8:53 PM)
1094 9:06p 🔵 Pre-commit state: 3 intentional files changed, test file untracked
1095 " 🔴 Committed: autonomous loop final yard closed — ConfigAdjuster wired into StagedDeployer
S328 Runtime health check after committing autonomous loop fix — scanner halted, blockers identified before loop can be exercised (May 1 at 9:07 PM)
1096 9:11p 🔵 TUI alive but scanner halted in dry_run mode with zero cycles
1097 " 🔵 Scanner halted by auto-halt; meta ledger clean; models critically stale
1098 " 🔵 All 5 recent journal trades are losses; auto-halt triggered by 14-consecutive-loss streak
S327 Post-commit system health check — verify TUI state, scanner halt cause, trade journal, and meta-pipeline readiness after 57f9aa8 merge (May 1 at 9:12 PM)
1099 9:13p 🔵 Ctrl+R safe-restart confirmed; ConfigAdjuster fix active; 'k' key is supervisor_kill not unhalt
S329 Post-Ctrl+R verification — fix confirmed loaded, but misleading 'k' key unhalt guidance discovered and halt still active (May 1 at 9:14 PM)
1100 9:15p 🔵 Misleading 'k' unhalt message located: embedded_scanner.py:376-377
1101 " 🔵 Full TUI key binding map and failing model load chain identified in gates.py
1102 " 🔵 Exact halt message code confirmed: one-shot latch at embedded_scanner.py:370-381
1103 " 🔵 action_safe_restart pattern documented as template for new action_unhalt
1104 " 🟣 TUI unhalt hotkey added: 'u' → action_unhalt binding in app.py

Access 1120k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>