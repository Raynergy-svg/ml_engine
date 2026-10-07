# Nautilus observation preparation — 2026-10-07

David approved the existing proposed persistent preparation at 04:07 UTC,
receipt `Sentinel_ad84d8c8f8c8819193613c66de301f9f`: a separate observation process,
candidate SQLite storage capped at 64MiB including WAL/SHM, and owner-only
observation-runtime status. The framed-record bound remains 64KiB. Production
activation is separate. This receipt authorizes these preparations; it does not
prove source admission, publisher identity, Site wiring or installed supervision.

Current code checkpoint: `15d4fda039b4484e2d4c9050a83efe32a0aec831`, based on
`65694bf43472181d53a3b750091cc7b240689de3`, isolated branch
`codex/axiom2-nautilus-operational-bindings-2026-10-07`. Earlier green continuous
checkpoint `3c0ca52c` and cloud-tested code `933753a9` remain preserved.

## Implemented preparation

`scripts/axiom2_nautilus_observation_worker.py` provides an isolated child using
inherited private input/reply pipes. Native C/Rust logs go to stderr. The default
entrypoint refuses activation before creating state; only explicit offline
fixtures can start it. Its environment retains PATH/PYTHONPATH/language/timezone
settings and drops inherited credentials. No socket, provider, execution client,
research/inference job or signer is constructed. This is process preparation,
not an installed supervisor or an authenticated remote transport.

The fixed journal path is `<configured evidence root>/observation_runtime/candidates.sqlite`
with its WAL/SHM companions. The owner configures the root outside records.
Directory mode is 0700 and files are 0600 under umask 077; symlink components,
hardlinked files, foreign-owned or group/world-accessible runtime paths are rejected.
The evidence root is owner-controlled and all ancestors reject non-sticky group/world
write access. Trusted ancestry owners are the process owner and filesystem-root
owner, accounting for mapped UID65534 in this managed Linux namespace; ordinary
root remains UID0. Sticky `/tmp` retains its normal protected-entry semantics. Input records
cannot select paths/configuration. Exact action/field sets are status, observe,
pending, complete and shutdown; restart/credentials/configuration are rejected.
Newline framing is bounded before JSON parsing, duplicate keys/nonfinite values
are rejected, and responses are bounded to 64KiB too.

Offline observation/result fixtures exercise existing admitted-record interfaces.
Their confirmation/qualification booleans are synthetic policy fixtures, not
source admission or promotion proof. The actual owner receipt/evidence/thesis
adapter and publisher binding remain blocked on source coverage below. Every
response retains execution_enabled=false and capital_authorized=false; local
health retains owner_transport_verified=false and live_feed_verified=false.

Optional `CandidateJournal(max_storage_bytes=...)` enforces the approved combined
64MiB ceiling using conservative SQLite page allocation, no cache spill, and
checkpoint-before-write. Database allocation is at most one quarter of the cap,
less 64KiB overhead; unused headroom reserves WAL/SHM and transaction overhead.
A pinned WAL reader denies further writes. Existing oversized stores are rejected
before opening. Allocation exhaustion retains committed history; no TTL prune,
fresh-history substitution or retention-policy change is provided. This bounds
the cooperative single-writer mechanics, not arbitrary owner filesystem/SQLite
mutation. Smaller bounds are used only in offline exhaustion tests.

The continuous runtime checks storage before timers/start, retains local FAULTED
truth when SQLite is unwritable, stops the actual native handle, and clears
references/timers even if final status persistence fails. Status additionally
reports the existing lease and data deadline; reading never refreshes either.
The owner reply pipe carries a distinct observation_runtime object. No Site
status handler, signed publisher namespace/key, persistent Site schema or
production environment has been changed.

Independent review of initial checkpoint `bc0848c6` found that pinned native
MessageBus callbacks can log/swallow Python exceptions: a failed invalidation
could leave READY truth unchanged. It also found writable/hardlinked storage
paths. Repair `87d971bc` explicitly captures callback failure, stops/faults the
runtime, rejects delivery after native publish, and preserves SQLITE_FULL instead
of masking it with an unconditional rollback. A second review required ancestry
replacement protection, implemented by `0ac4529d`/`15d4fda0`. Actual regressions
fail against the preserved defective checkpoints and pass on current code. The
initial green test receipt is not promoted as reviewed completion.

## Source coverage and exact blocker

The current Library skill was used for supported consumer-local materialization.
The supplied `...v0` references were not visible; a Library list legitimately
rediscovered canonical archive identities:

| Source | Resolved Library identity | Coverage here |
|---|---|---|
| Existing partial source handoff | `libfile_8cb6ef2dfde88191bd421423873f9801`, `axiom2-current-source-review-2026-10-05.tar.gz`, 665111 bytes | Metadata only; no archive bytes transferred/verified. Parent's previously known SHA c3007bcd… and 31 missing server files remain unverified here. |
| Staged canary patch | `libfile_fb8eec91cba88191a762c72d6e7acab9`, `axiom2-market-canary-staged-2026-10-05.tar.gz`, 23249 bytes | Metadata only; a patch is not complete source. |
| Site v71 archive | Source commit `6633d32a3b9884c7f8fa64ae6722bcee827f9b5d`; declared 115 files / 2334720 bytes, SHA 4d71bfbc… | Zero server source bytes verified in this checkout. Existing MCP declarations/read-only status metadata were inspected. |
| Linux repository | All 64 registered research/portfolio/execution/native sources | Exact source inventory is locally verified; only journal/runtime native hashes change in this preparation. Research SOURCE_FILES and authority profile are unchanged. |
| Mac checkout/extension | Independently owned by thread `01a11280-4b86-722e-8f8d-edcfa7181030` | No Mac checkout bytes or preparation tests verified by this Linux agent. |

Bundled current `library_download.py` plus its companion helpers were read via
the Library skill and used unchanged with the complete list result, selection
`011013`, and consumer-local destination `nautilus-authorized-site-handoff`.
The helper exited 1 before transfers:

```text
library download request failed: hosted apps tools/list request failed with HTTP status 401
```

This is consumer-helper authentication failure, not a confirmed file-specific
permission denial. No alternate byte path, raw URL, token, credential or generic
download of the undiscovered Library backing file was attempted after that
failure. The earlier Site backing-file error was authorization-or-resolution,
not proof of denied permission. The Site Library projection is not an archive.

Required action: restore the existing consumer's Library helper authentication,
or have the source owner provide the existing complete authorized handoff through
a supported healthy consumer. Verify archive SHA/manifest and coverage of the
actual admission producer, worker ingress, publisher signature/type/owner scope,
physical status store/reader and installed supervision. The reported partial
handoff and Mac checkout require their owning agents' exact coverage verification.
Do not invent missing filenames or infer complete coverage from a declared SHA.

## Verification and remaining status

Pinned upstream source remains unmodified at
`4f021bafc2e99c5490cee204b0fc2bd2c83baab4`. The reused Linux extension SHA256 is
`09832798c20e663d7917a72d427308960925f34594e0345f69351fc2b08df6fb`; existing
approved artifact/manifest/toolchain provenance remains in the continuous handoff.
No native rebuild was performed. This is not a Mac equivalence claim.

148 focused preparation/continuity/operational/isolation/inventory tests passed
in 11.06s, with warnings as errors. They include
real native idle timers, READY expiry, generation restart, old-generation denial,
byte framing, malformed/control rejection, owner-private paths, real SQLite FULL,
pinned-reader denial, retained records, unwritable-store native shutdown, native
MessageBus SQLITE_FULL delivery denial from READY, and replaceable-ancestor and
hardlink rejection (the linked source is explicitly 0600).
Three independent sequential review passes found and verified the repairs above.
Final review approved exact `15d4fda0`; its 26 native preparation/continuity cases
passed in 7.96s with warnings as errors. The complete existing Axiom/evidence gate
passed **1484 tests, zero failures/errors/skips, in 168.08s** on that exact code
in the selected Linux cloud workspace. This is an offline reused-artifact result,
not a new GitHub Actions run; earlier 1465 Actions passes belong to `933753a9` only.

Stable receipts: [summary](nautilus-observation-preparation-evidence-2026-10-07/summary.json),
[final gate](nautilus-observation-preparation-evidence-2026-10-07/preparation-final-full-gate.log),
[JUnit](nautilus-observation-preparation-evidence-2026-10-07/preparation-final-full-gate.xml),
[independent review](nautilus-observation-preparation-evidence-2026-10-07/independent-preparation-final-review.md),
[64-source integrity](nautilus-observation-preparation-evidence-2026-10-07/preparation-final-source-integrity.json),
[source coverage and exact materialization error](nautilus-observation-preparation-evidence-2026-10-07/preparation-source-coverage.json).
The directory also retains the earlier rejection receipts and real failure/fix
probes; final approval does not hide the initial uncovered failures.

| Binding | Implemented | Tested | Reviewed | Mac pending |
|---|---|---|---|---|
| Pinned native continuous mechanics | Yes, retained upstream node/actor/clock | Linux real timers, crash/restart/fencing and callback fault | Approved exact 15d4fda0 | Independent current qualification |
| Isolated child / bounded private pipe | Offline fixture preparation; default activation denied | Real Linux process, idle timers, malformed records, restart, trusted ancestry | Approved exact 15d4fda0 | New child not qualified |
| Owner-private candidate store / 64MiB combined cap | Yes, conservative SQLite allocation; no pruning | Real SQLite exhaustion, pinned WAL reader, retained history and hardlink rejection | Approved exact 15d4fda0 | Quota/filesystem behavior not qualified |
| Local observation_runtime status | Owner child reply object, no live transport proof | Native handle/lease/deadline and fault truth | Approved exact 15d4fda0 | Not qualified |
| Source admission / signed Site status binding | Blocked on current source materialization | Interface-only prior proofs; no signed wiring claim | Source unavailable | Source-owner verification needed |
| Existing process supervisor installation / production activation | Unapplied | No production process started | Separate activation decision | Independent owner task |

Axiom evidence integrity/promotion, candidate/thesis policy, freshness,
risk/capital/human/final command authorization, signed journal/fencing,
reconciliation truth and sole production gateway remain authoritative and
unchanged. LEAN PR63 remains open/draft and paused. No merge, provider access,
production activation or Mac work was performed.
