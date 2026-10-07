# Nautilus cloud completion verification — 2026-10-07

Status: **replay implementation and sequential review complete; full integration
handoff BLOCKED**. Production remains disabled. This file fulfills Task 5 of the
2026-10-06 plan; its evidence was collected on 2026-10-07.

## Exact scope and provenance

- Completion branch: `codex/axiom2-nautilus-cloud-completion-2026-10-07`.
- Inherited checkpoint and untouched runtime branch:
  `9e996c8bd3e0a37fafed2793be1d1165d91209b3`.
- Final tested code: `13f4e1e7fd6a7d91ec63dff075684eb3d53c5926`.
  Subsequent handoff commits change documentation/evidence only.
- Unmodified Nautilus source: `4f021bafc2e99c5490cee204b0fc2bd2c83baab4`.
- Original build run: `37553124116`; artifact: `11454954090`.
- Original ZIP SHA256:
  `fad40bc654f5965a8bef2317036f1b9a20504138f30e2a516afcfa90414ec721`.
- Original Linux extension SHA256:
  `09832798c20e663d7917a72d427308960925f34594e0345f69351fc2b08df6fb`.
- Exact manifest/cache digest:
  `6eaefb9f2c089663e5d59aecb7ae91cd4685b7459588129e04a75bc7f8134270`.

The original archive contains only the extension: its hidden manifest and venv
were not uploaded. Reuse checks the original archive digest, run/head/artifact
identity and a freshly reconstructed source/toolchain manifest matching the
supplied cache digest. The isolated environment was recreated with upstream
`make sync`; **no native compilation** occurred in this completion task.
Every cache hit now also checks the extension against the original binary digest.
Source wrapper imports report absent distribution metadata honestly. Neither
the environment nor these Linux results establish Mac equivalence.

## Implemented / tested / reviewed / Mac pending

| Path | Implemented | Tested | Reviewed | Mac pending |
|---|---|---|---|---|
| Pinned public Clock/MessageBus and BacktestEngine/DataActor lifecycle | Yes | Actual Linux smoke passed | Passes 1 and 3 | Yes, independent owner |
| Candidate policy, persistent replay freshness/version binding, atomic state/outbox | Yes | 11 candidate + 18 monitor cases passed | Passes 1, 2 and 3 | Yes, independent owner |
| Runtime publish, bounded wakeups, SIGKILL/restart/idempotent recovery | Yes | 5 native runtime cases passed | Passes 1, 2 and 3 | Yes, independent owner |
| Durable native order replay and Axiom projection/reconciliation consumers | Yes | 9 native order + 7 projection cases passed | Passes 1, 2 and 3 | Yes, independent owner |
| Exact binary/artifact reuse and credential-stripping download redirects | Yes | 8 contract/integrity cases passed; real extension digest verified | Pass 3 finding fixed with RED→GREEN tests | Yes, independent owner |
| Full current Axiom/evidence checkout isolation admission | Blocked by unchanged inventory gate | 1,407 passed; 3 failed; 0 collection errors | Block confirmed in pass 3 | Linux block must be resolved separately |

The 58 focused cases passed locally after the review fix in 0.45 seconds.
Tests use actual pinned native interfaces for runtime and order behavior.
Projection checks existing Axiom transition and reconciliation contracts;
it does not create signed execution authority. Rejected events, identity
conflicts, lifecycle parity gaps, overfills, unsupported currencies and
unrepresentable quantities/money remain retained and unresolved/HALTED.

## Three sequential adversarial passes

1. Requirements/API/provenance: checked the exact source and original artifact,
   public constructors, actual smoke and required integration consumers.
   Fixed typed identifier and fixed-point quantity assumptions; did not add
   upstream timer dispatch or unsupported direct actor lifecycle calls.
2. Durability/freshness/authority: exercised transactional rollback, raw-byte
   integrity, immutable nested facts, candidate-version supersession, late
   research results, deadline shortening/expiration, persistent replay clock,
   SIGKILL/restart, duplicate/conflicting wakeups, seed drift, cross-connection
   order replay, event identity collisions, terminal late fills and overfills.
3. Fresh whole-branch reviewer: reviewed checkpoint `65922ec4` independently
   and ran 56 focused cases (passed in 0.46 seconds). No additional runtime
   correctness defect was found. Review confirmed the inventory admission
   blocker and reproduced both research failures on untouched `9e996c8`.
   Its binary cache evidence finding was fixed at `13f4e1e7`: two new corruption
   and ambiguous-extension regressions failed before implementation, then
   the 58-case focused selection passed. Every cloud cache path verifies the
   pinned binary digest and retains a JSON receipt. No second reviewer or
   native rebuild was used.

The journal integrity checks establish internal consistency of unsigned
mechanics, not resistance to replacement of the entire database. The Python
socket guard proves the exercised Python smoke path denied connections; it is
not process-level containment of arbitrary native code. Replay clock advancement
is explicit. Live timer callback dispatch, feeds, native provider transport,
signed journal/fencing, capital/risk/final command authorization and the sole
production gateway remain Axiom-owned or outside this replay authorization.

## Final cloud results

Run [37562644230](https://github.com/Raynergy-svg/ml_engine/actions/runs/37562644230)
tests code `13f4e1e7` with `reuse_only=true`, `export_artifact=false`.
Exact binary verification, all 58 focused cases and actual smoke passed.
The complete scoped suite finished: **1,407 passed, 3 failed, 0 skipped,
0 collection errors**, 221.022 seconds. The workflow conclusion is **failure**;
all three failures are listed below. No native build step ran.

Artifact `11457224348` retains the complete JUnit XML, smoke, binary receipt
and reconstructed manifest for 90 days. Its downloaded ZIP SHA256 is
`70389af2700c72519082fec72095498245c3aba10ea316ee2200c963c5863d1d`.
Stable tracked summaries/receipts are in
[nautilus-evidence-2026-10-07](nautilus-evidence-2026-10-07/linux-cloud-test-summary.json).
The retained manifest hashes to the exact approved cache digest and the final
cloud binary receipt records the original extension digest.

Full scoped command (warnings are errors):

```text
python -m pytest tests/axiom2 tests/evidence/equity_research \
  tests/test_evidence_contracts.py tests/test_evidence_store.py \
  tests/test_equity_research_evidence_slice.py \
  tests/test_equity_research_evidence_worker_no_authority.py \
  -q -W error --junitxml=axiom-scoped-tests.xml
```

## Remaining blockers and exact next actions

1. **Isolation inventory admission:**
   `tests/axiom2/portfolio/test_boundary.py::test_research_artifact_cannot_import_portfolio_or_execution`
   rejects the Nautilus source directory through the exact observed-source check
   in `scripts/axiom2_verify_isolation.py`. No gate/profile/hash was changed,
   skipped or xfailed. The parent must obtain explicit authorization for an
   independently reviewed, hash-pinned Nautilus exclusion inventory/profile
   extension. Keep Nautilus outside the research bundle and preserve unknown
   source/drift denial. The proposed inventory supplied with this handoff is
   **unapplied** and grants no authority.
2. **Inherited research verifier failures:**
   `test_finite_five_arm_campaign_retains_signed_reports_and_history` and
   `test_report_verifier_recomputes_signed_derived_conclusions` in
   `tests/axiom2/research/test_phase1_development.py` both fail at
   `src/axiom2/research/development.py:562` with `paired bound mismatch`.
   The fresh reviewer reproduced both on untouched `9e996c8` (2 failed in
   23.88 seconds). Assign/authorize a separate bounded numerical-report verifier
   repair and rerun these two tests plus the scoped gate. That repair is not
   hidden inside this Nautilus task.
3. **Mac evidence:** independently owned by thread
   `01a11280-4b86-722e-8f8d-edcfa7181030`. Parent coordinates that handoff;
   this task performed no Mac build or validation.

The offline native-read declaration probe passes with zero broker actions,
`execution_enabled=false`, `capital_authorized=false`, and
`live_connection_verified=false`; it is declaration integrity only, not
operational provider proof. Existing research/portfolio/execution inventory
hashes remain unchanged. The original runtime branch, separate workspace,
and paused LEAN PR63 were not changed. No merge, reset, amend, force push,
deployment, inference purchase, credentials or real orders were used.

The completion branch/worktree and stable evidence are retained for review.
Full integration completion cannot be claimed until the first two blockers
are resolved and the scoped gate passes. Production activation requires its
separate existing authorization and is not part of this handoff.
