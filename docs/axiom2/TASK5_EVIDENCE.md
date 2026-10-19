# Task 5 — Experiment registry and sealed-holdout evidence

## Scope

Implements Task 5 of the approved research-kernel plan. These are local,
execution-disabled evidence services, not a research runner, promotion decision,
broker adapter or live deployment. Task 6 is not included.

All durable authority lives in the existing `EvidenceStore` root. Its new
`research/` stream contains immutable signed event receipts; existing package,
disposition, verdict, index and champion formats are unchanged. There is no
parallel database, current-state JSON file, or researcher-selected fallback path.

## Public interfaces

Configure `ExperimentRegistry(store, signer=..., actor_id=...)` with a trusted
local evidence-service signer. `register_experiment(proposal)` stores the full
Task 2 proposal without changing its text. Identical retries return the stored
registration and do not inflate attempted-search count; a different proposal
under the same experiment ID is rejected. Each actual experiment attempt must
receive its own ID; the future campaign runner must enforce preregistration
before execution. This registry cannot discover off-registry computations.

`record_failure(experiment_id, reason)` appends a terminal failure without
removing the original proposal. Failed attempts remain in every subsequent
registry count. A failed experiment cannot freeze or use holdout evidence.

`freeze_candidate(experiment_id, artifact_hash, code_commit)` requires a valid
registered proposal, an exact SHA-256 artifact reference and the proposal's full
Git identity. It returns a frozen candidate bound to the registration digest,
code/artifact identity, signer/actor, timestamp, sequence and complete attempt
count at freeze. Its candidate ID is the signed event's content digest. One
experiment cannot change its frozen identity; exact retries are idempotent.
A later code change requires a new registered attempt, not an edit to history.

Both required names also exist as module-level functions with an explicit
`registry=` keyword. No global singleton, environment-discovered signer, default
storage path or implicit credential acquisition is introduced.

## Holdout lifecycle

Configure `HoldoutAuthority` against the same store with the appropriate actor
and signer. The service uses these existing authority roles:

| Operation | Required role |
|---|---|
| Register an experiment, record failure, freeze candidate | LOCAL_IMPORTER |
| Declare holdout identity and coverage | OPERATOR |
| Open holdout and record its result | INDEPENDENT_VERIFIER |

A producer's valid signature is not enough to authorize these operations. The
store independently verifies signatures, role/key bindings and transition
semantics under its lock. Holdout evaluation additionally requires both an actor
and a signing key different from the candidate freezer.

1. `register_holdout(id, dataset_digest, coverage_start, coverage_end)` declares
   immutable metadata with an aware, nonempty half-open coverage interval.
   The declaration must precede candidate freeze. Reused dataset digests or
   overlapping declared coverage are rejected across this store, even under
   a different holdout ID. This conservative policy also rejects overlapping
   periods for different universes; a new disjoint period needs its own manifest.
2. `assert_untouched(id)` checks status. It does not reserve the holdout and
   must not be used as a check-then-read authorization.
3. `open_holdout(candidate_id, holdout_id)` atomically records OPENED **before**
   the evaluator accesses data. It is a one-time claim, not an idempotent read.
   The opening receipt includes the full attempt count then known, so trials
   registered after freeze are not concealed. A candidate cannot shop among
   several final holdouts.
4. `consume_holdout(candidate_id, holdout_id, result_hash)` attaches exactly
   one result digest to that reservation. The candidate and evaluator actor
   must match the opener. Changed-result and identical replays are rejected.
   The function also exists at module level with explicit `authority=`.

An evaluator crash leaves OPENED evidence permanently unavailable as untouched;
result recording can resume without opening again. No reset/delete operation
exists. Failed candidates and results remain in the signed history. Unknown
holdout IDs fail closed rather than counting as virgin evidence.

## Durability and reconstruction

`EvidenceStore.load_research_ledger()` reads every canonical signed receipt and
reconstructs history before returning it. `append_research_event()` repeats that
verification, checks the expected head, validates the proposed transition, and
uses the existing store-wide `fcntl` lock plus fsync/atomic-create primitives.

The journal is one global sequence for this evidence root. Duplicate IDs,
sequence gaps, incorrect previous digests, signature corruption, malformed
schemas, bad receipt addresses, symlinks, unauthorized events and inconsistent
claims all block reads/writes. Timestamps must be monotonic, signing time must
match event time, and local receipt skew cannot exceed 30 seconds. Historical
key trust is checked at receipt; current signing authority is checked on writes.

No mutable index supplies experiment counts. `snapshot()` returns immutable
records and read-only mappings reconstructed from the journal. The existing
package/champion index intentionally does not project this new stream.

Operations retry compare-and-swap contention up to 32 times, then explicitly
fail for the caller to retry. A lost response never rolls back committed data.
Unpublished atomic-create temporaries left by an interrupted writer are ignored;
committed receipt files are never overwritten.

## Security and evidence limits

The store root, its lock, trusted clock, role/key administration and historical
trust records require protected, persistent operator administration. No keys are
created by the application services; tests/probes generate ephemeral keys only.
Reconstructing a brand-new root is not permission to reuse real holdout data.

Signatures and predecessor links detect modification and interior deletion;
malicious deletion of a valid journal suffix or rollback of the entire store
requires an external trusted checkpoint/WORM backup to detect. No claim of
protection from an administrator who can rewrite the store and its trust base
is made. Ordinary Python immutability is not a hostile-code sandbox.

Holdout declarations and artifact/result hashes are **references**, not proof of
source truth, coverage completeness, correct evaluation, or data secrecy. This
module neither stores nor returns sealed dataset bytes. The later dataset and
evaluator boundary must authenticate manifests, keep holdout bytes inaccessible
to ordinary researchers, release data only after durable opening, and verify the
actual result bytes/metrics before promotion. A process that already possesses
sealed data can bypass a metadata API; it must not receive that capability.

Registration/freezing/consumption do not write a champion pointer, grant orders,
weaken validation thresholds or prove economic edge. Task 8's campaign and
Task 9's promotion authority must consume the verified history, including failure
state and attempted-search counts, rather than trusting free-form conclusions.

## Verification gate

The maintained Axiom job runs all `tests/axiom2`, all Task 5 tests under
`tests/evidence/equity_research`, both shared evidence/store test files and the
existing equity evidence slice/authority tests. The source artifact includes
only the reviewed 21-file project closure plus explicitly installed Pydantic and
cryptography dependencies. The standalone probe executes real signed registry
and holdout transitions; legacy broker/scanner modules remain unavailable.

Tests use real filesystem operations, Ed25519 signatures and spawned processes.
No mocked persistence/signing, production credentials, real holdout data, model
training, external broker calls or live authority are used. Exact commands and
results accompany the review checkpoint. The repository-wide suite remains a
separate diagnostic; this Task 5 gate cannot claim Task 10 or live readiness.
