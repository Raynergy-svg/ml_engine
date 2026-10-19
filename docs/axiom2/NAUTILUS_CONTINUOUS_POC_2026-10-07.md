# Nautilus continuous observation PoC — 2026-10-07

Scope: isolated synthetic/injected admitted observations; not a provider-fed
production monitor. Green replay checkpoint `479d868` is preserved on its
original completion branch. This slice is on
`codex/axiom2-nautilus-continuous-poc-2026-10-07`.

## Current-state/API evidence

- Pinned source and unchanged extension remain `4f021bafc2e99c5490cee204b0fc2bd2c83baab4`
  and SHA256 `09832798c20e663d7917a72d427308960925f34594e0345f69351fc2b08df6fb`.
- `Clock.new_test()` is public; the Rust `PyClock::new_live()` helper is **not**
  Python-exposed. `DataActor.clock` receives a real live clock from registered
  `LiveNode` actors. The actual pinned extension confirms these facts.
- `LiveNodeConfig` admits an upstream SANDBOX node with zero data/exec clients,
  factories, plugins, catalogs and external message-bus/cache configuration.
  No strategy or execution algorithm is registered.
- Upstream `LiveNode.run_async()` owns the native runner and callback dispatch
  through its host asyncio bridge. The PoC registers a native actor timer;
  it does not implement a clock, timer, event bus or event-loop engine.
- The current Axiom admitted-observation boundary is `CandidateObservation`
  through the existing runtime topic into `CandidateMonitor`. This slice
  consumes already-admitted injection; it does not admit raw provider inputs.
- Existing owner-private/stdio transport is a bounded research request handoff.
  No established continuous status/observation binding or deployed native owner
  endpoint is present. Do not label the PoC connected to that owner transport.

## Implemented first checkpoint

`NautilusContinuousRuntime` in the existing excluded runtime module:
real actor heartbeat/freshness timers, native MessageBus observation delivery,
SQLite generation/expiring lease, and a connection wrapper that checks the lease
inside each existing candidate-journal write transaction. SQLite serializes
takeover with mutation. Candidate policy, durable outbox and revalidation remain
Axiom-owned. No automatic inference/research invocation occurs.

Seven focused tests pass against the actual Linux extension (3.32 seconds):
live timer revokes READY without virtual advancement; shutdown drops the upstream
node and cancels timers; stale/future input denial; persistent result idempotency;
old-generation journal transaction denial; host-dispatch stall status/fault; actual
SIGKILL recovery retains the pending wakeup and increments generation.

The initial two feature tests failed before implementation. An upstream rule
requiring the old live node to be dropped before another can be built on the
same thread was exposed by the second test and fixed before the seven-case run.

## Fault verification checkpoint

Twelve focused tests pass against the pinned Linux extension with warnings as
errors (`12 passed in 3.90s`). Added active-lease denial before node creation,
clock regression preserving the original fault, cancelled-driver cleanup,
future research-result rejection, and native startup-failure status truth.
Fault tests initially exposed three cleanup/status defects; the reviewed change
preserves the first fault, cleans up after cancellation, and records startup
failure as FAULTED. Future-dated research results cannot promote READY.

The native per-thread node guard follows object lifetime. A separate direct
upstream probe, without this adapter, showed that a caller-retained cancellation
traceback retains that guard. After releasing the traceback, rebuilding succeeds.
The cancellation test checks stopped runner/timers first, releases its exception
frame, then checks same-thread rebuild. The adapter does not rewrite caller
tracebacks or use garbage collection as a production recovery mechanism.
Process termination and fresh-process recovery are independently tested.

| Deliverable | Implemented | Tested | Independently reviewed | Mac |
|---|---|---|---|---|
| Preserved replay completion `479d868` | Yes | 1,453 cloud tests | Yes | Pending independent owner |
| Isolated continuous observation runtime | Yes | 12 native Linux tests; included in 1,465 cloud tests | Approved cb140d71 | Pending independent owner |
| Exact native exclusion registration | Yes; exact reviewed runtime hash/imports | 179 focused/isolation tests | Approved proposal | Pending independent owner |
| Final scoped cloud verification | Yes; code 933753a9 | 1,465 passed; no failures/errors/skips | Reviewed source and exact registration | Pending independent owner |
| Owner transport / provider / production worker | No | No | Out of this slice | Not claimed |

## Final cloud evidence and boundaries

Independent review approved `cb140d715cb4f5b9839b69d4356c2df62af51a7a` and
the exact runtime registration. The reviewer independently passed all 12 cases
with warnings as errors. After applying that registration, 179 native runtime
and research-isolation tests passed in 7.19s. Final cache-only cloud run
[37568201717](https://github.com/Raynergy-svg/ml_engine/actions/runs/37568201717)
passed **1,465 tests**, with **zero failures, errors or skips**, in 276.985s.
Tested code: `933753a9675e750da805ba73b39a297aa7b3b902`. Native compilation was
skipped. Exact original Linux artifact `11454954090` from run `37553124116`,
ZIP digest `fad40bc654f5965a8bef2317036f1b9a20504138f30e2a516afcfa90414ec721`,
extension digest above, and matching source/toolchain manifest were verified
before use. The manifest key remains
`6eaefb9f2c089663e5d59aecb7ae91cd4685b7459588129e04a75bc7f8134270`.
No research profile or unrelated source hash changed; all 64 registered source
files match. Replay branch `479d868`, original runtime branch `9e996c8`, original
workspace `4c6ace9` and open draft LEAN PR 63 are preserved.

Verification ran sequentially: first seven native lifecycle cases, twelve
expanded fault/admission cases, independent twelve-case review plus restricted
import audit, then the registered 179-case isolation/native gate and final cloud
gate. No broker/provider or new inference campaign was exercised.

Stable evidence is retained in
[`nautilus-continuous-evidence-2026-10-07`](nautilus-continuous-evidence-2026-10-07/):
cloud summary/run/manifest/binary/smoke and original-artifact reuse receipts;
independent review/proposal/tests; source-integrity receipt; cancellation-reference
probe; and bounded synthetic observation snapshots. The smoke artifact included
native logs before its JSON; raw bytes are preserved separately and the final
JSON object is extracted explicitly. Verification artifact `11459598755` ZIP
SHA256 is `66d545a504a96a8f1d357ab772fdd3785f4cc5d476627f5abea70661c3e4a682`.

The bounded demonstration records NOT_STARTED, RUNNING with READY, RUNNING with
expired data/false READY after native callbacks, and STOPPED. Execution, capital,
live-feed and owner-transport flags remain false throughout. Nothing was installed
or left running as an observation service.

Reproduce the native slice using the pinned source and corresponding native
extension in PYTHONPATH, with the exact Axiom test dependencies already specified
by the workflow, then run:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/axiom2/nautilus_runtime/test_continuous_runtime.py -q -W error
```

For Mac qualification, the independent Mac owner must run this twelve-case slice
and the scoped gate against code `933753a9` using its own verified arm64 artifact.
The reported arm64 import/model mechanics probe is progress, not this qualification.

These leases protect unsigned observation mechanics, not Axiom's signed
production journal/fencing or authorization. Status never enables execution,
capital, owner transport or live-feed claims. Python audit guards are not native
OS network isolation. No process is installed or left running as a production
worker. Provider/collector access, owner endpoint binding and production restart
supervision remain unactivated; parent must define/authorize those exact bindings
before operational continuous monitoring can be claimed. Mac build/validation
remains independently owned and is not inferred from Linux evidence.
