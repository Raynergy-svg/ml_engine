# Axiom 2.0 research isolation and release gates

## Scope and decision

The research kernel does not inherit the historical scanner's execution path.
A pre-existing failure outside its demonstrated dependency closure is tracked
separately, not made a prerequisite for Tasks 5–10. This does **not** waive any
regression caused by Axiom, a failing shared dependency it uses, repository
branch protection, or a safety failure on a deployment's actual execution path.

This boundary profile covers **only the implemented Tasks 1–4 contracts**. The
experiment registry, holdout, features, campaign and promotion workflow are not
implemented by this change. A passing boundary receipt explicitly reports
`research_kernel_complete: false` and `execution_enabled: false`.

## Executable gate

```bash
python -m pytest tests/axiom2 -q
python scripts/axiom2_verify_isolation.py --bundle /tmp/axiom-research-contracts.zip
```

Use a new output filename for every invocation. Existing bundles are never
overwritten. A failed probe does not publish a bundle. Successful invocation
prints a JSON receipt including the bundle digest, exact source digests,
exercised APIs, loaded project modules and unavailable legacy modules.

The dedicated `Axiom Research Boundary` workflow runs all current Axiom tests
and this proof. It uses a read-only repository token, disables credential
persistence, and installs pytest rather than the legacy ML/trading stack.
Existing workflows and branch-protection requirements are unchanged. Committing
this workflow is not proof that GitHub ran it or that it is a required check;
those statuses must be verified separately on the actual pull request.

## What the proof checks

1. `config/axiom2/research_boundary.json` pins the audited SHA-256 of all seven
   project source files, including `src/__init__.py`. A changed or missing file,
   an extra Axiom source/native module, symlink, or unsupported profile fails.
   Bytecode caches are never included in the artifact.
2. Every explicit import, including function-local and conditional imports, is
   checked against the small approved closure. The current external dependencies
   are only `__future__`, `dataclasses`, `datetime`, `re` and `zoneinfo`. Wildcard
   imports and direct dynamic-execution/file-open capabilities are rejected.
   This is a second tripwire, not a complete Python static analyzer.
3. The bundle is built from the captured audited bytes (not a second disk read).
   Its exact member inventory and contents are verified, rejecting extra,
   missing, duplicate or altered members. No scanner, broker, credentials,
   runtime state, models, launch scripts or legacy operator code is packaged.
4. The exact artifact is exercised in a fresh interpreter using `-I -S -B`,
   a sanitized environment, and a temporary working directory. The repository
   and site packages are not on the child import path. The real proposal,
   temporal and universe APIs run; unknown availability is also rejected.
5. Import attempts for `src.scanner.execution`, `src.brokers.oanda`,
   `src.training.correlation_group_config`, `src.sota_core`, `src.axiom_operator`,
   `src.evidence` and `src.axiom2.execution` genuinely fail because those modules
   are absent. Source origins and all loaded `src` modules are checked. Audit
   probes detect socket-connect/bind and subprocess-execution attempts during
   the exercised calls. They do not replace OS isolation.

The trust base is the reviewed source inventory, gate/tests/workflow, Python
standard library, interpreter and runner. A person able to rewrite both the
source and gate can defeat repository-local checks. This is **not** a security
sandbox against hostile Python, an arbitrary-code execution service, or a proof
about every possible input to future modules. Source hashes require human
review; updating hashes mechanically is not approval of new dependencies.

## Separate artifacts and launch paths

The verified artifact is a **source-only research-contract bundle**, not the
whole monorepo or an installable trading application. Its supported use is a
clean research process with only this source root plus approved dependencies.
Do not launch it through `buddy`, the legacy scanner/dashboard/operator, or a
process already carrying broker objects, credentials or a second repository
on its import path. Those are different deployments outside this proof.

No live process was migrated, no order gateway was enabled, and no legacy
execution defect was repaired by this isolation work. An audit/test receipt is
not signed promotion evidence and does not enter a parallel evidence database.

## Known legacy risk: correlation lookup

At the reviewed Task 4 base `273a0b59de14c4111b2e60cad62463e4cddeb894`,
`src/scanner/execution.py::_check_correlation_exposure` imports a missing
`get_pair_correlation` and its `ImportError` path returns an allow result.
Repair PR #56 has also recorded this unresolved fail-open behavior.

Disposition: **BLOCKED FOR USE BY ANY DEPLOYMENT RELYING ON THAT EXECUTION PATH.**
It is not in this research artifact or its dependency closure. Keep it as a
separate legacy risk repair; do not invent coefficients, weaken thresholds,
remove the regression, or call that execution path safe because Axiom tests pass.
If future Axiom work imports or delegates to it, it becomes an Axiom blocker.

## Extending this gate during Tasks 5–10

Any new Axiom module or change to pinned bytes fails this profile until reviewed.
When adding shared evidence modules, explicitly audit their **transitive** imports,
source inventory, package initializers, side effects and entrypoints. Include
all relevant signing/hashing/storage and regression tests; do not allowlist the
entire old repository to get a green check. The artifact boundary must expand
with the implementation, not leave new modules silently untested or unpackaged.
The current subprocess exercise is then extended to the real integrated path.

Task 10 remains governed by the approved plan:

```bash
pytest tests/axiom2 tests/evidence/equity_research -v
```

That must include proposal → registry → temporal/universe validation → features
→ labels/splits/baselines/ranker → frozen candidate → one-time holdout → promotion,
plus the existing relevant evidence regressions. Missing directories or unbuilt
stages must not be treated as a pass. Current isolation success is not Task 10.

A later execution release additionally requires deterministic portfolio/risk,
broker-neutral orders, a read-only Robinhood adapter, shadow reconciliation,
sole order authority, privilege-negative tests and human-approved execution
proof. No safety gate on that actual path can be exempted as unrelated legacy.

## Current disposition

Research development may continue on the approved Axiom track with this closure
check maintained. Legacy repair PR #56 remains separate. Tasks 5–10 and live
readiness remain unverified. Preserve three review passes and stop for the
operator after each phase; this boundary change does not start Task 5.
