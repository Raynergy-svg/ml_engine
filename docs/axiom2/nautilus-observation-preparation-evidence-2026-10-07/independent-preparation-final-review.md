# Independent final preparation review — 2026-10-07

Decision: APPROVE the bounded, offline-fixture-only preparation at exact commit 15d4fda039b4484e2d4c9050a83efe32a0aec831. Both material findings from the bc0848c6 review are resolved. This approves preparation code and its exact reviewed scope, not production activation or missing owner/source bindings. No implementation edited by reviewer.

Reviewed sequence: bc0848c6 -> 87d971bc -> 0ac4529 -> 15d4fda. Final diff from87d971 changes only worker ancestry validation and its tests; native journal/runtime sources are unchanged from the independently verified callback fix.

P1 resolved: actual native MessageBus observation errors now fault locally and propagate explicit rejection after publish. SQLite automatic rollback no longer masks SQLITE_FULL. The earlier independent fixed-code probe verified original records retained, failed invalidation absent, FAULTED/no readiness, native handle stopped, and combined storage229376 bytes under the1MiB test cap. The real native FULL regression was rerun at final HEAD as part of the26 passing tests.

P2 resolved: all path ancestors are checked for symlinks, trusted ownership and non-owner write permissions; safe sticky-directory semantics are retained. The direct evidence root still requires current-owner control and no group/world write bits. DB/WAL/SHM require private owner-owned regular files with exactly one hardlink. The filesystem-root owner's numeric UID is read from stat('/') to accommodate the managed namespace (observed65534; current1000), rather than granting a generic foreign-owner exception. Independently repeated the exact prior layout: owner0700 evidence root under nonsticky0777 ancestor now rejects before namespace creation. Trusted sticky ancestor/private root succeeds. Same-owner0600 hardlinked database rejects. Normal private layout under mapped-root sticky/tmp succeeds. Regression source now explicitly uses0600, so hardlink rejection is independently exercised.

Independent final verification:26 passed in7.96s with warnings treated as errors, using the unchanged pinned Linux extension. Log: /workspace/nautilus-operational-evidence/independent-preparation-final-tests.log. Namespace probe: /workspace/nautilus-operational-evidence/independent-final-namespace-probe.log. Earlier independent actual-native fixed-quota probe: /workspace/nautilus-operational-evidence/independent-fix-sqlite-full-probe.log.

Final file SHA-256:
- scripts/axiom2_nautilus_observation_worker.py: abd0e3ee3bc002a6a684027f5f1a274723e4dd59d56f8fc74aebff8643f6848a
- src/axiom2/nautilus_runtime/journal.py: 1bfeb84c67b980e9d16fcbc00c873ea6794f8e7038c2bb438b4ac8f103a57550
- src/axiom2/nautilus_runtime/runtime.py: 6ff0be3413a86bfd6de5eb6e353aace90e50663a26f695321ff83db53313fd3d

Scope assessment: conservative database page cap, disabled spill, pre-write WAL checkpoint and pinned-reader denial bound cooperative DB/WAL/SHM growth. Failures retain history and clear current readiness; there is no automatic pruning. Framing is bounded to64KiB. Production activation defaults disabled and requires --offline-fixture for the prepared child. Input confirmation/qualification flags remain synthetic fixture assertions, not signed admission. Only the two reviewed native source hashes were repinned; protected research/provider/trading authority was not expanded. No new provider, credentials, research-job dispatch, risk/capital grant, signed-journal writer or alternate execution gateway was added.

| Deliverable | Implemented | Independently tested | Reviewed | Mac |
|---|---|---|---|---|
| Offline fixture child, bounded framing, local status/lifecycle | Yes | Final26 targeted tests pass | Approved for preparation only | Pending independent owner |
| Candidate DB/WAL/SHM quota and SQLite-full denial | Yes | Real native FULL regression and prior independent probe pass | P1 resolved; approved bounded mechanics | Pending independent owner |
| Private namespace/ancestor/hardlink denial | Yes | Prior failing layouts now reject; safe sticky/tmp case passes | P2 resolved | Pending independent owner |
| Exact native source hash registration | Two reviewed hashes updated | Native targeted checks; final full gate owned by parent | No protected authority expansion found | Pending independent owner |
| Site source admission / owner publisher / production activation | Not wired; activation disabled | Not established | Blocked by unavailable source (helper401;0-byte coverage); no bypass approved | Pending independent owner |

Limitations: targeted Linux review/tests only, not an independent full scoped gate, cloud run or native rebuild. Parent's final full-HEAD gate must supply final regression status; an earlier gate is not substituted. Path checks assume a cooperative owner and trusted filesystem-root owner, not hostile same-UID/root mutation. Python guards are not OS network isolation. No production process installation, real provider/broker interaction, new research campaign or Mac duplication occurred. An untracked preparation document may be updated by parent; approval is bound to the code commit and file hashes above.
