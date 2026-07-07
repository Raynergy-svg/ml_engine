# Policy-as-code promotion gates

CI-only governance policies, enforced with [Open Policy Agent](https://www.openpolicyagent.org/)
via [Conftest](https://www.conftest.dev/). These formalize safety rules the repo already holds
implicitly (see `CLAUDE.md` Hard NOs, `docs/ENGINEERING_BRAIN.md` Part II — The Safety Model,
`.claude/rules/improvement.md`). Nothing here changes runtime, trading, execution, or halt
behavior — these policies only gate what merges into the repo.

## What runs where

| Command | Purpose |
|---|---|
| `conftest verify --policy .ci/policy/rego` | Rego unit tests (`*_test.rego`) — pure logic tests against mocked `input`, no git needed. |
| `conftest test --policy .ci/policy/rego <input.json>` | Evaluates one PR-context JSON document against all three rules. |
| `scripts/ci/generate_policy_input.py` | Produces the real PR-context JSON from `git diff` + PR labels + evidence files on disk. |
| `.ci/policy/run_fixture_tests.sh` | Runs `conftest test` against every `testdata/*_allow.json` (must PASS) and `*_deny.json` (must FAIL) fixture — the acceptance-level check that the rules actually reject/allow as intended. |
| `python -m pytest tests/test_generate_policy_input.py` | Real-git tests (no mocks — builds throwaway git repos under `tmp_path`) for the generator script's file-classification logic (eval-report timestamp filter, ship-gate vs eval-report bucketing, checklist-file detection, risky-diff-pattern detection). Runs in the existing `test` job in `code-quality.yml`, not the `policy` job — it's a Python unit test like any other. |

CI wiring: `.github/workflows/code-quality.yml` job `policy` runs all three in sequence.

## Input contract

Every rule reads the same JSON document (see `scripts/ci/generate_policy_input.py` for the
generator, `.ci/policy/testdata/` for worked examples):

```jsonc
{
  "pr": {"number": "123", "base_ref": "main", "head_ref": "abc123"},
  "changed_files": ["src/brokers/oanda.py", "tests/test_brokers_oanda.py"],
  "diff_signals": {
    // substrings found in the ADDED lines of the diff that indicate a
    // live/arm-relevant edit, regardless of which file it landed in
    "risky_live_patterns_found": ["oanda_environment=\"live\""]
  },
  "review_evidence": {
    // true iff the PR adds/modifies a file under .ci/policy/evidence/{live-arm,model-promotion}/
    // (other than the TEMPLATE.md placeholder)
    "checklist_file_changed": true,
    // PR labels, if available from GITHUB_PR_LABELS env (comma-separated)
    "labels": ["policy:live-arm-reviewed"]
  },
  "gate_evidence": {
    "ship_gate_files_changed": ["trained_data/backtests/SHIP_GATE.json"],
    "eval_report_files_changed": []
  }
}
```

The rego rules only reason about this structured document — they never parse git or read the
filesystem themselves. That keeps the policy logic declarative and testable with plain JSON
fixtures, and keeps the (comparatively messier) git/diff plumbing in one reviewable Python script.

## Rules

### 1. `policy.broker_execution` — broker/execution changes need test coverage

File: `rego/broker_execution_test_coverage.rego`

**Triggers** when `changed_files` includes anything under `src/brokers/`, or
`src/scanner/execution.py`, `src/equity/executors.py`, `src/equity/order_lifecycle.py`,
`src/equity/kill_switch.py`.

**Requires** at least one changed file under `tests/` matching `test_*.py`, or under
`tests/e2e/`, in the *same* PR.

**Why**: `.claude/rules/improvement.md` "Test Coverage Gates" — "NEVER ship a new subsystem
without at least 5 unit tests"; broker/execution is the highest-blast-radius surface in the repo.

### 2. `policy.live_arm` — live/arm/unhalt changes need review evidence

File: `rego/live_arm_review_evidence.rego`

**Triggers** when `changed_files` includes `src/equity/live_gate.py`,
`src/crypto/crypto_live_gate.py`, `src/crypto/crypto_carry_live_gate.py`,
`src/equity/track_b_live_gate.py`, `src/agent_runtime/policy.py`, or
`src/scanner/automation/state_engine.py` — **or** when `diff_signals.risky_live_patterns_found`
is non-empty (the generator script greps added diff lines for patterns like
`oanda_environment="live"`, `armed = True`, `halted": false`, `LIVE_CONFIRMATION_TOKEN`,
regardless of which file they land in).

**Requires** either a checklist file under `.ci/policy/evidence/live-arm/` (copy
`TEMPLATE.md`, fill it in, commit it in the same PR) or the `policy:live-arm-reviewed` label.

**Why**: Hard NOs #1/#2 in `CLAUDE.md` (`oanda_environment` stays `"practice"`; respect
`halted:true`) and `docs/ENGINEERING_BRAIN.md` §II.4 (LiveGate is operator-token-gated —
`arm()` requires a typed confirmation + passing ship-gate). This policy does not, and cannot,
verify the *content* of the review — it only enforces that a human-readable evidence artifact
exists in the PR. That is a deliberate, documented limit (see "What this does not do" below).

### 3. `policy.model_promotion` — model promotion needs eval/gate evidence

File: `rego/model_promotion_gate_evidence.rego`

**Triggers** when `changed_files` includes `src/brain_loop/promotion.py`,
`src/training/promotion_policy.py`, `src/equity/ship_gate.py`,
`src/scanner/automation/staged_deployer.py`, or `src/evaluation/soak_orchestrator.py`.

**Requires** either a changed file under `trained_data/backtests/` (fresh ship-gate/eval
output committed alongside the code change) or a checklist file under
`.ci/policy/evidence/model-promotion/`. "Fresh eval output" is deliberately narrow: the
generator only counts a `trained_data/backtests/` file as eval-report evidence if its name
matches the timestamp convention every producer in this repo already uses (`\d{8}T\d{6}Z`,
e.g. `equity_harvester_singlestock_20260702T050603Z.json`) — an incidental touch of an
unrelated stale file in that directory (rename, delete, formatting fix) does not count.

**Why**: Hard NO #3 (`CLAUDE.md`) — "nothing promotes to champion without passing the ship gate"
(`HARD_MAX_GAP = 0.10`); `.claude/rules/improvement.md` "Hard Ship Gate — 10% Train/Val Gap".

## What this does not do

- **Not a runtime gate.** These policies evaluate PR diffs in CI. They do not run inside
  `EmbeddedScanner`, `execution.py`, or any hot trading path, and cannot halt/unhalt/arm
  anything at runtime. The Hard NOs remain enforced at runtime by the existing code (e.g.
  `src/scanner/execution.py` halt re-check, `src/equity/live_gate.py` `arm()`).
- **Not a substitute for code review.** The live-arm and model-promotion rules check that an
  evidence *artifact* was added, not that a human actually reviewed it carefully. Treat a
  passing policy job the same way you'd treat a passing linter — necessary, not sufficient.
- **No bypass/override mechanism.** By design. If a rule is wrong for a legitimate change, fix
  the rule (with a fixture proving the fix), don't work around it.

## Adding or changing a rule

1. Edit the `.rego` file and its paired `_test.rego` file first (`conftest verify --policy
   .ci/policy/rego` must stay green).
2. Add or update a `testdata/*_allow.json` / `testdata/*_deny.json` fixture pair.
3. Run `.ci/policy/run_fixture_tests.sh` locally — it must report every `_deny` fixture as
   FAIL (policy correctly rejects it) and every `_allow` fixture as PASS.
4. Update this README's rule table.
