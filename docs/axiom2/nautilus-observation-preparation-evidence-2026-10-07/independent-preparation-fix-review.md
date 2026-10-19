# Independent preparation-fix review — 2026-10-07

Reviewed exact commit 87d971bc6b19aa547b6c5e195792aa72a1d2b289 against bc0848c6bd0325915db29f9948be5259af7d248e. Decision: NOT YET APPROVED; P1 resolved, remaining P2 ancestor-path check required. No implementation edited by reviewer.

P1 resolution verified independently: the observation callback captures errors and faults locally; publish explicitly checks fault truth after native bus delivery, so swallowed native callback exceptions cannot produce ACCEPTED. Conditional rollback preserves the original SQLITE_FULL error, and COMMIT failure now shares cleanup. A repeated actual-native probe established READY, filled a 1MiB capped journal, and submitted a 20KiB invalidation. The retained original remained, invalidation was not committed, publish raised 'observation callback failed closed: OperationalError', status was FAULTED with no readiness, and the native handle stopped. Combined DB/WAL/SHM was 229376 bytes. Log: independent-fix-sqlite-full-probe.log.

Remaining P2 (scripts/axiom2_nautilus_observation_worker.py:29–36): ancestry checking rejects symlinks but verifies ownership/write bits only for the evidence root itself. A same-owner mode0700 evidence root beneath a non-sticky mode0777 directory is accepted. Another writer of that parent can rename the whole evidence root and replace it with a symlink between validation and SQLite open. Independent probe accepted this layout and demonstrated the returned DB path resolving outside the approved root. This concerns untrusted ancestor writers, not a hostile same-UID owner. Required: reject unsafe replacement permissions/ownership across the ancestry, preserving safe sticky-directory handling for /tmp. Add a writable-ancestor denial regression. Log: independent-fix-ancestor-probe.log.

The direct writable-root and multiply-linked DB/WAL/SHM checks are present. Improve the hardlink regression by explicitly chmodding its source DB to0600; otherwise permissive creation mode can make the test pass via the existing permissions check without exercising st_nlink.

Independent targeted run: 25 passed in 7.84s with -W error, pinned unchanged Linux extension. Log: independent-preparation-fix-tests.log. No full-suite run by reviewer; earlier or concurrently running full-gate outcomes do not resolve the reproduced path gap.

| Deliverable | Implemented | Independently tested | Reviewed | Mac |
|---|---|---|---|---|
| Offline child/framing/lifecycle/status | Yes | 25 targeted tests pass | Callback/readiness fix approved; path approval pending | Pending independent owner |
| Quota/page-cap failure handling | Yes | Actual SQLite-full repro now fails closed; retained state verified | P1 resolved | Pending independent owner |
| Namespace confinement | Partial | Writable-root/hardlink cases covered; unsafe ancestor reproduced | P2 remains; extend ancestry checks | Pending independent owner |
| Source admission / owner publisher / production activation | Blocked/not wired/default disabled | Not established | Site source helper401/0-byte coverage remains blocker; no bypass reviewed | Pending independent owner |

Scope remains preparation only: synthetic policy flags are not admission evidence; no provider, broker, research-job, trade, capital or signing authority added. No build, cloud run, Mac duplication, production start or original-branch changes by reviewer. Receipt is bound to87d971 even if parent subsequently fixes the path check.
