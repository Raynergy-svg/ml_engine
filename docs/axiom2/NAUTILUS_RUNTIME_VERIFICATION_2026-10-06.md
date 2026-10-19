# Nautilus cloud completion verification — 2026-10-07

Status: **reviewed cloud replay completion handoff COMPLETE; full scoped gate
PASS**. Production remains disabled. This file fulfills Task 5 of the
2026-10-06 plan; its evidence was collected on 2026-10-07.

## Exact scope and provenance

- Completion branch: `codex/axiom2-nautilus-cloud-completion-2026-10-07`.
- Inherited checkpoint and untouched runtime branch:
  `9e996c8bd3e0a37fafed2793be1d1165d91209b3`.
- Final verification code: `6fa68a036d468f0206fe6f0eae83704380892def`.
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
| Signed paired-report exact numerical verification | Yes; two verifier expressions preserve producer arithmetic order | 3 focused cases passed, including signed one-unit tampering; full scoped gate passed | Independent follow-up APPROVE | Yes, independent owner |
| Full current Axiom/evidence checkout isolation admission | Exact reviewed seven-file exclusion applied | 136 boundary cases passed; complete scoped gate 1,453 passed | Independent follow-up APPROVE; research artifact bytes unchanged | Yes, independent owner |

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

A subsequent user-authorized bounded blocker closure received independent
review of numeric commit `eb4d231b` and exclusion proposal digest
`ba5bccaee62a716f60ffb3d93b7173d79aba40e6e39d44a403142e7ad91ae7f0`.
The reviewer independently passed 3 numeric tests (35.31 seconds) and 42
exclusion tests (0.42 seconds), finding no material issue or permission expansion.
Its [full recommendation](nautilus-evidence-2026-10-07/independent-followup-review.md)
compares unchanged research ZIP bytes before/after registration. After application,
136 current inventory/research/portfolio boundary tests passed in 5.21 seconds.

The journal integrity checks establish internal consistency of unsigned
mechanics, not resistance to replacement of the entire database. The Python
socket guard proves the exercised Python smoke path denied connections; it is
not process-level containment of arbitrary native code. Replay clock advancement
is explicit. Live timer callback dispatch, feeds, native provider transport,
signed journal/fencing, capital/risk/final command authorization and the sole
production gateway remain Axiom-owned or outside this replay authorization.

## Final cloud verification

Final run [37563968612](https://github.com/Raynergy-svg/ml_engine/actions/runs/37563968612)
tests code `6fa68a03` on a bounded completion-branch code push. Cache/artifact
reuse is enforced for that branch; documentation-only pushes do not rerun it.
Exact binary verification, all 58 focused cases and actual smoke passed.
The complete scoped suite finished: **1,453 passed, 0 failed, 0 skipped,
0 collection errors**, 275.140 seconds. Workflow conclusion: **success**.
No native build or environment recreation step ran on this final cache hit.

Artifact `11458660699` retains complete JUnit XML, smoke, binary receipt and
reconstructed manifest for 90 days. Its downloaded ZIP SHA256 is
`be35466b50aa5cee9fb62e765ec7566f7a0c279f2e8837f9e7e31d499052b5b8`.
[Final tracked summary](nautilus-evidence-2026-10-07/linux-final-cloud-test-summary.json),
[binary receipt](nautilus-evidence-2026-10-07/linux-final-cloud-binary-receipt.json),
[manifest](nautilus-evidence-2026-10-07/linux-final-cloud-manifest.json) and
[actual smoke](nautilus-evidence-2026-10-07/linux-final-cloud-smoke.json) are stable
handoff evidence. The downloaded archive digest, exact approved manifest digest
and original extension digest were independently checked during capture.

Historical run `37562644230` at `13f4e1e7` produced 1,407 passed/3 failed;
its summary/receipts remain retained as `linux-cloud-*`. Those three failures
are resolved by the reviewed fixes below and the final clean-checkout run.

Full scoped command (warnings are errors):

```text
python -m pytest tests/axiom2 tests/evidence/equity_research \
  tests/test_evidence_contracts.py tests/test_evidence_store.py \
  tests/test_equity_research_evidence_slice.py \
  tests/test_equity_research_evidence_worker_no_authority.py \
  -q -W error --junitxml=axiom-scoped-tests.xml
```

## Resolved integration blockers and remaining scope

1. **Isolation inventory admission — resolved by reviewed registration:**
   `f2935d2c` applies the exact seven original proposed source hashes in
   `config/axiom2/nautilus_boundary.json`. The complete literal-false replay-only
   profile is required. Unknown source/resources, drift, missing files/policy,
   symlinks, arbitrary path registration, forbidden network imports and research
   imports of Nautilus remain denied. The existing research `SOURCE_FILES`,
   import permissions and dependency versions are unchanged. Native bytes and
   profile are excluded from the returned research snapshot and ZIP; an isolated
   import probe confirms the package is unavailable from that ZIP. The independent
   reviewer determined this is exact known-source coexistence under the user's
   stated exception, without material research/security/authority expansion.
   [Reviewed precise diff](nautilus-evidence-2026-10-07/reviewed-exact-exclusion.patch)
   is retained. Its test file was renamed `test_nautilus_inventory_boundary.py`
   on application to avoid collision with the existing portfolio test filename;
   all reviewed code and test contents match exactly.
2. **Inherited signed numeric verifier mismatch — repaired separately:**
   `eb4d231b` changes only two paired-return expressions to preserve producer
   IEEE-754 evaluation order: subtract separately computed cost-adjusted returns.
   A concrete reproduction differs by two floating-point units when reassociated.
   No tolerance, threshold, bootstrap parameter, signed-evidence validation or
   campaign authority changed. Only the repaired `development.py` hash was
   updated to `de6935ca014c234995e1c4424e80735282668db243d772359d523a5061a38300`;
   the other 56 existing source hashes remain unchanged. Both inherited tests
   and a new exact signed-book/one-unit-statistic-tampering regression pass
   (3 in 35.69 seconds; independently 3 in 35.31 seconds). Exact altered signed
   means/lower bounds for both comparisons remain rejected.
3. **Mac evidence:** independently owned by thread
   `01a11280-4b86-722e-8f8d-edcfa7181030`. Parent coordinates that handoff;
   this task performed no Mac build or validation.

The offline native-read declaration probe passes with zero broker actions,
`execution_enabled=false`, `capital_authorized=false`, and
`live_connection_verified=false`; it is declaration integrity only, not
operational provider proof. Existing portfolio/execution inventory hashes and the other 56 original source
hashes remain unchanged; only the explicitly repaired research verifier was repinned. The original runtime branch, separate workspace,
and paused LEAN PR63 were not changed. No merge, reset, amend, force push,
deployment, inference purchase, credentials or real orders were used.

The completion branch/worktree and stable evidence are retained for review.
The fresh post-repair scoped gate passes. No cloud replay integration blocker
remains; Mac and production/live operations retain their independent scope. Production activation requires its
separate existing authorization and is not part of this handoff.
