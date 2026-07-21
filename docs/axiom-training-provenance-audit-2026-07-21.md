# src/axiom_training Provenance & Scope Audit — 2026-07-21

**Trigger:** clean-checkout replay of `77355b8` failed with
`ModuleNotFoundError: No module named 'src.axiom_training'`. The package —
16 modules, 4,485 lines — had never been committed to any branch
(`git log --all -- src/axiom_training/` is empty), is not gitignored, and is
imported by seven tracked files including the live FX lane's risk gate and
position sizer. The local machine was the only known copy.

## Preservation (before anything was touched)

- Byte-exact backup: `~/axiom_preservation/2026-07-21T103515Z/` (operator-only
  permissions), verified byte-identical against the working copy, 16 files.
- Per-file SHA-256 + size + mtime manifest stored alongside the backup.
- Aggregate manifest SHA-256:
  `c665396f4802857474c54cf3ca7719cda96889f4fbae07eb2d715aae4129de2c`
- The local `src/axiom_training/` directory was not edited, moved, or
  reformatted at any point. Commits were built by copying into a throwaway
  worktree and hash-verifying against the manifest.

## Provenance

File mtimes span 2026-07-09 19:42 → 2026-07-11 08:51, matching the
resident-loop / risk-training build window recorded in session memory for that
week. A 14-file companion test suite (`tests/test_axiom_*_2026_07_{09,10,11}.py`,
also untracked) was written alongside it — the package was developed
test-first and simply never committed by the session that built it.

## Classification (16 files)

| File | Class | Writes | Notes |
|---|---|---|---|
| `__init__.py` | production runtime (package init) | none | re-exports 10 modules → defines the import closure |
| `risk_runtime.py` | **production runtime — load-bearing** | status JSON under `trained_data/axiom/`, audit JSONL | `evaluate_runtime_risk` + `scale_target_units` consumed by the live FX lane. De-risk-only: scale clipped [0,1], invalid → 0.0, promotion-not-enabled → 1.0 |
| `risk_gym.py` | production runtime (policy model) | none | `RiskPolicy`/`RiskPolicyParams`; `gymnasium` import guarded — runtime path needs none of it |
| `risk_release_safety.py` | production runtime | none | pure checks |
| `runtime_health.py` | production runtime | none | pure evaluator |
| `promotion_criteria.py` | production runtime | none | pure evaluator |
| `system_health.py` | production runtime | none | pure evaluator (dashboard) |
| `action_policy.py` | production runtime | example/report JSON | consumed by resident loop + dashboard |
| `episode_ledger.py` | production runtime | episodes JSONL | resident loop |
| `outcome_linker.py` | production runtime | links JSONL | package init closure |
| `policy_advisor.py` | production runtime | advice JSONL | `pickle.load` of local model artifact (line 76) |
| `online_risk_learning.py` | production runtime | learner state JSON | post-trade loop |
| `action_promotion.py` | production runtime | promotion state | dashboard; `pickle.load` of local candidate (line 100, guarded) |
| `action_stress.py` | training-only | none | consumer is an **untracked** script — stays out |
| `release_report.py` | training-only | report JSON | consumer untracked — stays out |
| `stress_scenarios.py` | training-only | none | consumer untracked — stays out |

Scans: **zero** secrets/tokens/keys, **zero** machine-specific absolute paths,
**zero** network calls (no requests/urllib/socket anywhere in the package),
no generated artifacts or runtime state inside the package. All writes are
local JSON/JSONL under `trained_data/` and `logs/` via atomic replace or
append. No duplication: `evaluate_runtime_risk`/`scale_target_units` exist
nowhere else in the tree.

## Tracked consumers → symbols

| Consumer | Symbols consumed |
|---|---|
| `src/equity/oanda_trend.py` | `risk_runtime.evaluate_runtime_risk`, `risk_runtime.scale_target_units` |
| `src/agent_runtime/loop.py` | `action_policy.{ACTION_POLICY_EXAMPLES_PATH, ACTION_POLICY_MODEL_PATH, ACTION_POLICY_REPORT_PATH, append_cycle_examples}`, `episode_ledger.{EPISODES_PATH, append_episode_from_cycle}`, `policy_advisor.{POLICY_ADVICE_PATH, advise_and_append}` |
| `src/scanner/feedback/post_trade_loop.py` | `online_risk_learning.OnlineRiskLearner` |
| `dashboard/server/data_sources.py` | `action_policy.{ACTION_REGISTRY, read_policy_status}`, `system_health.{axiom_system_health_to_dict, evaluate_axiom_system_health}`, `risk_runtime.read_promotion_status`, `online_risk_learning.read_online_learning_status`, `action_promotion.{read_action_promotion_status, validate_action_candidate}` |
| `dashboard/server/app.py`, `axiom_evidence_control.py` | route-string / doc references only — no direct import |
| `scripts/run_oanda_trend.py` | file-path references (`risk_runtime.py`, `online_risk_learning.py`) for freshness checks |

External deps beyond stdlib: `numpy` (declared), `gymnasium` (declared at
`requirements.txt:125`, import-guarded). First-party deps `src.agent_runtime.*`
and `src.axiom_operator.session` are tracked.

## Import closure ⇒ minimal committed slice

Any `from src.axiom_training.X import …` executes `__init__.py`, which imports
10 modules; those transitively pull `risk_release_safety`; the dashboard
additionally consumes `action_promotion`. Minimal slice = **13 of 16 files**.
The 3 excluded files' only consumers are untracked training scripts — they stay
out until a tracked consumer or explicit retained capability justifies them.

## Test evidence

- 10 package test files (all mock-free, consistent with the No-Mock rule):
  44 passed, 1 failed against the local machine — the failure is `gymnasium`
  missing from the local interpreter, not an implementation mismatch (dep is
  declared; import guarded).
- Committed: the 4 suites that pass on the clean tracked tree
  (episode_ledger, outcome_linker, policy_advisor, risk_promotion) — 25
  passed, 1 skipped, plus the new gate.
- New load-bearing gate: `tests/test_clean_tree_imports.py` imports every
  tracked consumer from the tracked tree and asserts the exact consumed
  symbols exist. A missing first-party module fails loudly regardless of
  environment; only genuinely third-party gaps (fastapi/textual/…) skip.
  On the clean tree: 11 passed, 1 skip (fastapi, declared third-party).

## Residual unversioned surface (found during commit staging — NOT resolved)

The clean tree exposed three further layers of the same disease:

1. **Untracked scripts consumed by committed code.**
   `online_risk_learning.py:67,104` lazily imports
   `scripts.train_axiom_risk_policy` (untracked). The import is
   function-level and the caller is documented never to raise into the
   trade-close path, so a clean checkout imports fine and only online
   *retraining* would fail. Also untracked: `scripts/train_axiom_action_policy.py`,
   `scripts/backfill_trade_feedback.py`, `scripts/write_axiom_model_release_report.py`,
   `scripts/sweep_axiom_risk_policy.py`, `dashboard/server/axiom_training_control.py`.
2. **Test files deferred because their deps are unreviewed/untracked**:
   `test_axiom_action_policy_2026_07_09.py`,
   `test_axiom_promotion_and_runtime_health_2026_07_10.py`,
   `test_axiom_risk_gym_2026_07_10.py` (import the untracked scripts),
   `test_axiom_model_stress_and_action_promotion_2026_07_10.py`,
   `test_axiom_risk_state_v2_2026_07_11.py` (import the 3 excluded
   training-only modules).
3. **Uncommitted wiring on a tracked file.** The `OnlineRiskLearner` hookup
   (`PostTradeLoop._run_online_risk_learning`) exists only in uncommitted
   local modifications to `src/scanner/feedback/post_trade_loop.py` — the
   tracked copy has no such method, so
   `test_axiom_online_risk_learning_2026_07_10.py` fails on a clean tree and
   is deferred with it. That file is co-mingled with another session's work;
   hunk-level extraction is a separate, operator-visible step.

All deferred files are preserved byte-exact in the private backup
(`companion_tests/`, `companion_scripts/`).

## Commit sequence (session/loop-restoration-2026-07-20)

1. pure contracts/models: `risk_gym`, `risk_release_safety`, `runtime_health`
2. deterministic risk runtime: `risk_runtime`
3. readers + policy layer + package init: remaining 8 + `__init__`
4. tests: 10 package suites + clean-tree import gate + this audit

Every committed file is byte-identical to the preserved manifest (SHA-256
verified at commit time); the sequence alters no runtime behavior — it
versions the exact implementation currently executing.
