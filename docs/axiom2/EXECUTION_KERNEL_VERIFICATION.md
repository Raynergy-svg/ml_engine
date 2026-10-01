# Axiom 2.0 Execution Kernel Verification

## Current checkpoint

**Task 11: implemented and locally verified on the connected Mac.** Tasks 12-18 remain unimplemented. This is a shadow-only deterministic portfolio/risk core, not a broker integration, production risk service, deployable order gateway, or proof of an investable edge.

The predecessor was verified exactly before editing: branch `codex/axiom-2-task-10`, HEAD `a348bed38c45e6bba066d04820b22ba841c2e057`, clean worktree; 795 passed in 10.37s, dedicated Task 10 E2E 2 passed in 0.19s, standalone research isolation PASS. Work continues in the same linked Mac worktree on `codex/axiom-2-task-11`.

Authoritative plan: `docs/superpowers/plans/2026-10-01-axiom-2-execution-kernel.md`.
Machine-readable local receipt: `docs/axiom2/TASK11_VERIFICATION_RECEIPT.json`. This receipt records local verification; it is not a signed capital authorization.

## Implemented interface and boundary

`src/axiom2/portfolio/contracts.py` defines frozen, strictly validated RiskPolicy, Position, PortfolioSnapshot, RankedOpportunity, PortfolioRequest, TargetPosition and RiskDecision records. `src/axiom2/portfolio/authority.py` defines `evaluate_portfolio(request, policy)` and `risk_policy_digest(policy)`.

The evaluator consumes explicit normalized observations and evidence references. It owns no model adapter, signer, account credential, broker transport, network client, global mutable trading state, or implicit current clock. Every result binds account alias, source class, model/promotion/economic/universe/calendar/snapshot identities, policy and canonical input/output digests. Financial arithmetic uses integer cents, integer quote microdollars and integer basis points. Timestamp comparisons use elapsed UTC time, including DST folds.

The entry policy remains top five, equally weighted within an explicitly supplied capital sleeve, five trading-session holding horizon and rebalance interval, without overlapping cohorts. Weight remainder stays cash. A failed name, sector, correlation-group, gross, turnover, liquidity, spread, expected-net-edge, freshness or eligibility check rejects the entire cohort. The core never substitutes rank six or optimizes new weights after seeing a failure. Cash budgets subtract reserved funds, require settled cash and retain the declared reserve after conservatively rounded estimated costs. A cost buffer below the observed entry half-spread is rejected, not silently increased.

No risk thresholds or live capital sleeve have been selected for production. The tests use explicit illustrative policies; their returns have not been compared with the historical campaign. Risk/implementation overlays must be frozen before shadow evaluation, and material strategy changes remain new experiments.

## Deterministic states

| State | Meaning | New targets |
|---|---|---|
| READY_SHADOW | Complete valid hypothetical entry cohort | Five desired values, never broker orders |
| REJECTED | Entry evidence or risk rule failed | None |
| HOLD | Active cohort, off-schedule entry or closed session | None |
| HALTED | Kill state, or flat account in risk-off/drawdown stop | None |
| RECONCILE | Ambiguous/open orders, stale/unreconciled account, corporate action or unknown cohort | None |
| EXIT_REQUIRED | Known holdings reached maturity or need risk management | None; explicit handoff to future exit authority |

Empty targets never mean liquidation. Execution and reconciliation workers do not exist in this phase; later tasks must implement the handoffs and continue observation independently of a kill. Every result has `execution_enabled=false` and `capital_authorized=false`. SYNTHETIC, DEVELOPMENT and LIVE_SHADOW inputs all remain non-capital.

## Findings fixed during this phase

1. **Signed promotion could accept a NaN or negative-infinite baseline.** A real ephemeral signer exposed PROMOTED results that the previous unsigned negative tests did not catch. Seven failing tests also exposed boolean/float attempt-count and holdout-count aliases. The narrow correction at `ce225fb` rejects those cases without changing thresholds, adding models, rerunning historical research or opening real holdouts. Thirty new signer-present tests check specific rejection reasons and verify signed envelopes.
2. **New authority files violated the old whole-namespace inventory assumption.** The gate now separately validates exactly three portfolio source files under `config/axiom2/portfolio_boundary.json`, with pinned hashes, fixed import allowlists and false execution/capital flags. They are not packaged into research. Missing/partial inventory, drift, new native/resource/Python files, network dependencies and research-to-portfolio imports fail. All original isolation tests remain in place.
3. **A small cost buffer could understate the observed entry half-spread.** Four adversarial RED cases now reject that inconsistency with exact integer comparisons. This lower bound does not prove future fees, slippage, fills or cost-model accuracy.
4. **The new test basename collided with existing promotion tests in the full gate.** Renamed only the new file to `test_portfolio_authority.py`. No pytest discovery setting, cache, old test or legacy code was changed.

Three inline review passes covered requirements/semantics, adversarial/security/failure modes, and integration/regression/isolation. In-scope findings were fixed and rerun. No independent subagent review was performed or claimed.

## Exact verification

Environment is the existing complete Mac virtualenv, not a newly installed environment:

```sh
cd ~/ml_engine-task-5
P="$HOME/.local/share/axiom/repo-suite.Gi7lHb/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONWARNINGS=error

"$P" -m pytest tests/axiom2/portfolio tests/axiom2/promotion -q -W error
# 166 passed in 1.81s (127 portfolio, 39 promotion)

"$P" -m pytest tests/axiom2 tests/evidence/equity_research \
  tests/test_evidence_contracts.py tests/test_evidence_store.py \
  tests/test_equity_research_evidence_slice.py \
  tests/test_equity_research_evidence_worker_no_authority.py -q -W error
# 952 passed in 12.04s

"$P" -m pytest tests/axiom2/integration/test_research_kernel_e2e.py -q -W error
# 2 passed in 0.19s

"$P" scripts/axiom2_verify_isolation.py --bundle <new-external-path>/research.zip
# PASS, execution_enabled=false; implemented_tasks remains [1,2,3,4,5,6,7,8,9,10]
```

The final research bundle SHA-256 is `33b19496ee1f760e5533b6340e5c512086f0675b0a46f3031e515154fb732f5c`, identical to the post-promotion-safety-fix research bundle. Portfolio implementation added no bytes to that artifact. The pre-existing conservative `research_kernel_complete=false` receipt field is left visible, not converted into live readiness.

RED receipts: initial portfolio specification 81 failures, expanded requirements 84 failures, inventory integration 15 failures, cost/spread review 4 failures with 22 passing controls. GREEN includes 200 seeded allocation/rounding invariant cases inside the test suite, exact cap/freshness boundaries, settlement/reserves, kill/risk-off/cohort states, canonical replay, mutation/type rejection, UTC/DST and privilege-import negatives.

A separate Mac process, without pytest or research/model/broker modules, exercised **1,000 synthetic scenarios and 2,000 deterministic evaluations**. It independently recomputed receipt hashes and budget invariants and produced 200 cases each of READY_SHADOW, RECONCILE, HALTED, REJECTED and HOLD. No network or subprocess side effects were attempted. EXIT_REQUIRED and held-cohort behavior are covered by the pytest suite, not that flat-account probe.

Local raw logs, XML reports, the standalone probe script and its receipt are retained at `~/.local/share/axiom/execution-20261001.MqDecD/`. The initial older Task 5 environment lacked pandas; rerunning in the existing complete environment resolved collection without installs. A diagnostic receipt-print helper emitted an unclosed-file warning once; it was replaced and rerun cleanly. No product warning was suppressed. The unrelated repository-wide legacy suite was not run or made a prerequisite.

## Remaining gates and limitations

Hash references and booleans in normalized observations are not source authentication. This core has not verified a real broker account, a production promotion key's role, real universe/corporate-action data, a live calendar/quote stream or a genuine holdout result. Task 16 must resolve actual immutable evidence, exact signed identities and trust roles before execution admission; a caller-supplied PROMOTED string or valid-looking hash is insufficient. Python immutability and import tests are not OS-level credential isolation; the operational denial proof remains Task 17.

Next is **Task 12: broker-neutral equity order contracts and bounded risk-intent evidence**, still without broker submission. Robinhood read-only verification, shadow fills, durable idempotency/crash recovery, actual fill/cost reconciliation, sole execution authority and deployment privilege separation remain Tasks 13-17. Actual fees/slippage and engineering shadow gates are not established by these synthetic checks.

The genuine final holdout was not accessed by this work. No new model family was searched. No broker connection, review/approval request, real order, cancellation, production credential change, merge or deployment occurred. Capital remains prohibited through Task 17 and the Task 18 dry run; separate explicit human approvals and all evidence gates are required for the controlled proof.
