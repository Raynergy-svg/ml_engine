# Offline service composition checkpoint

The six fixed Task 13 read bindings can now be hosted by `TrustedReadHost`.
It accepts the exact normalized adapter and exposes account, market and order
reads. Authentication and callable bindings remain external trusted inputs.
The host creates no connection, credential, generic tool forwarder or write path.
Existing assistant connector authentication is not a native Python service grant.

`OfflineExecutionService` composes actual artifact resolution, reconstructed
signed risk admission, one-use operator authorization, shared POSIX fencing,
durable lifecycle and a private `FakeRobinhoodTransport`. The service accepts
only SYNTHETIC/DEVELOPMENT classification, one mapped instrument, whole shares,
whole-cent limit prices, regular hours and good-for-day orders. The captured
review/place/cancel declarations are descriptive contract evidence; no provider
calls were made. The fake has no network or injectable real transport.

Review, placement, fills, cancel acceptance and cancel confirmation are signed
in a persistent fake provider stream under the same EvidenceStore lock. Restart
reconstructs that history. An ambiguous placement is observed rather than retried;
a reservation with no provider record stays unresolved. Fake cancellation retry
uses exact signed history and records at most one matching cancel effect. This is
an explicit fake guarantee, not evidence of real provider cancellation idempotency.
Cancel acceptance stays pending; partial/late fills remain visible, including a
canceled remainder. Lifecycle provider checkpoints validate authenticated history,
exact request/response, deterministic request/order identity, quantity/state and
provider receipt ordering at append and replay. Replay of provider-backed rows
requires the explicitly bound exact shared provider journal.

Hypothetical reconciliation derives expected fills from the retained signed
lifecycle checkpoint and observations from independently replayed fake history.
Synthetic marks are explicit; missing coverage and unresolved orders fail closed.
Acknowledgement records zero lifecycle fills and never asserts execution completion.
Actor registration, revocation and receipt/signing/event time all apply to the
fake journal; retired actors can be replayed but cannot create effects.

Validation: 1318 full Axiom/evidence tests passed in 36.90 seconds with warnings
as errors and plugin autoload disabled; 18 new fake/service tests pass. Both
artifact-isolation probes pass. Regression coverage includes crash-before-cancel recovery, lost placement acknowledgement,
no blind placement after reservation, pending/late cancellation fills, four spawned
processes sharing one request identity, unsupported profiles, changed requests,
expired fencing, retired roles, pre-registration signatures, future provider
checkpoints, absent/altered checkpoints and cross-order evidence substitution.
Three independent read-only reviews identified defects that were corrected and
cleared the final scoped changes. The research artifact remains unchanged. The
standalone execution artifact contains 47 reviewed sources and imports service
composition without training/provider SDKs or network/process side effects.

This increment fits offline Tasks 13–17 and introduces no live architecture grant.
Task 13 source truth/settlement/approval/calendar gaps remain. Tasks 14–15 genuine
live shadow and authenticated reconciliation remain unverified; twenty sessions
and four cycles are not qualified. Task 16 real provider transport/correlation,
independent genuine holdout provenance and capital admission remain blocked.
Task 17 deployed identities, secrets/tools/network denial and cross-host/provider
fencing remain unverified. All public ExecutionAuthority actions stay BLOCKED;
all receipts keep execution_enabled=false and capital_authorized=false. No
settings, credentials, live broker actions, genuine holdout, publication or
deployment occurred. The original checkout was not changed by this increment.
