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

# [ml_engine] recent context, 2026-04-23 11:02pm EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (21,872t read) | 725,032t work | 97% savings

### Apr 16, 2026
24 9:04p 🟣 US-516 E2E Test Harness: conftest.py Upgraded with Three Shared Pytest Fixtures
27 " 🟣 US-516 E2E Test File Created: 12 Flow Tests for Supervisor Console
28 " 🔵 ConfigAdjuster.apply_adjustments: Approved History Read from Hardcoded Path, Not Constructor Arg
31 9:05p 🔴 Test Fix: ConfigAdjuster Uses persistence_path Not pending_path for Approved History
34 " 🔵 11/12 E2E Tests Pass; test_per_trade_close_calls_oanda Fails — httpretty Mock Not Intercepting ExecutionManager
36 9:08p 🔴 httpretty + urllib3≥2 Incompatibility Fixed: fakesock.__getattr__ Patched for shutdown/close
37 " 🟣 US-516 All 12 E2E Tests Pass Green in 6.42s
38 " 🟣 US-516 Deliverables Complete: Reality Check Report + CI Workflow Written
39 9:10p 🟣 US-517: Phase 91 Security + Code Review Gate Initiated
40 " 🔵 Phase 91 PRD: Full Module List US-501–US-515 Confirmed
42 9:17p 🟣 US-517 Phase 91 Security + Code Review Gate Initiated
43 " 🔵 All 15 Phase 91 Stories (US-501–US-515) Confirmed Passes=True
45 9:20p 🟣 US-517: Phase 91 Security + Code Review Gate Reports Written
46 " ⚖️ Phase 91 Closure: All 15 Stories Confirmed Complete Before Gate
47 9:21p 🟣 phase91_security_review.md Written: 180 Lines, All-Green Gate
50 9:22p 🟣 phase91_code_review.md Written: 146 Lines, All Gates PASS
51 " 🔵 US-518 Pre-flight: tests/perf Directory Missing, Phase 91 Reports Present
52 9:24p 🔵 DataProvider Architecture: threading.Lock Snapshot Swap, No deepcopy
53 " 🟣 US-518: bench_supervisor_console.py Performance Benchmark Suite Created
57 " 🔴 bench_supervisor_console.py Fails Direct Run: ModuleNotFoundError for src.tui
S11 US-518: Performance benchmark — TUI refresh under scanner load — implement bench_supervisor_console.py and generate phase91_perf_report.md (Apr 16 at 9:25 PM)
59 9:26p 🔵 Supervisor Console Hotkey Bindings Confirmed for US-519 Runbook
60 9:27p 🔵 Complete TUI Hotkey Map Confirmed Across All Screens and Modals
61 " 🔵 CLAUDE.md Brain Section Location and docs/ Structure Confirmed
63 9:28p 🟣 docs/supervisor_console_runbook.md Created — Full 10-Section Operator Runbook
65 9:30p ✅ Reality Checker Walkthrough Inlined into Runbook as Appendix A
67 " ✅ CLAUDE.md Claude Brain Section Updated to Reference Supervisor Console Runbook
68 9:31p 🟣 US-520: Phase 91 Final Reality Check + Evidence Collection Initiated
70 " 🔵 Phase 91 PRD Structure: 20 userStories, US-501–US-519 All passes=True, US-520 passes=False
72 " 🔵 All 19 Phase 91 Story Validation Commands Mapped for Evidence Collection
74 9:32p 🔵 Phase 91 Story Definition-of-Done: Sub-agent Sign-offs and Evidence Artifacts Required Per Story
76 " 🟣 Out-of-Band Verification Script Written for US-520 Evidence Collection
78 9:34p 🔵 OOB Verification: All 19 Phase 91 Stories Confirmed REAL After Two Check Fixes
81 9:35p 🟣 US-520: Phase 91 Evidence Collection Report Completed
82 9:39p 🟣 Phase 91 Evidence Report Written: All 19 Stories Verified REAL
S14 US-520: Final reality check + Phase 91 evidence collection — independent OOB verification of all 19 Supervisor Console stories (Apr 16 at 9:39 PM)
### Apr 21, 2026
440 9:49a 🔵 ML Engine Self-Heal Triggered — CRITICAL Drawdown Streak + RL Model Staleness
441 9:50a 🔵 ML Engine Config Adjustments State — 5 Prior Self-Heal Changes Active Since Apr 16
442 " 🔵 ML Engine Model File Inventory — agent_weights.json Last Modified Apr 16, RL Journal Updated Today
444 9:51a 🔵 Buddy MCP Tool Failures — OANDA_API_URL Unresolved, src.scanner Missing, Feedback Log Empty
445 " 🔵 Trade Journal Shows 17 Entries All With pl=None — No Closed Trades Recorded Locally
446 " 🔵 Agent Weights Snapshot — devil_advocate Highest at 1.3, trader_readiness Lowest at 0.5, All Regimes Nearly Identical
447 " 🔵 Learnings.md Documents Config Levers Maxed — Root Cause Is Outside Config Plane
### Apr 23, 2026
455 10:43p 🔵 Inbox Duplicate Adjustments — Root Cause Confirmed: 756 Pending, 252 Each for 3 Duplicate Keys
456 " 🔵 InboxScreen Architecture — Current State Before Approve-All/Reject-All Additions
457 10:44p 🔵 Inbox Duplicate Root Cause — engine.py Calls collect_adjustment() Every Scan Cycle Per Pair Without Dedup Guard
458 10:54p 🔵 ML Engine Self-Heal Cycle #1 — Critical Drawdown Streak + Stale RL Model
459 " 🔵 ML Engine Self-Heal Cycle #1 — Root Cause Confirmed: Retrain Never Executed + 4 Config Proposals Still Pending
460 10:56p 🔴 Inbox Duplicate Proposals Fixed — ConfigAdjuster and ConfigTuner Now Suppress Identical Pending Entries
461 " 🟣 Inbox Bulk Actions — AdjustmentApprover.approve_all() / reject_all() + InboxScreen Buttons
462 10:59p 🔴 Inbox Duplicate Adjustments — Root Cause Fixed in ConfigAdjuster + ConfigTuner
463 " 🟣 Inbox Approve-All / Reject-All Buttons Added to TUI InboxScreen

Access 725k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>