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

## Remaining work and boundaries

Independent review, additional fault/lease contention checks and final scoped
verification remain pending. The native exclusion hash/import registration is
deliberately unchanged pending that review; source drift is expected to block
the full gate until exact reviewed registration. No research profile expansion
or unrelated source repin is authorized.

These leases protect unsigned observation mechanics, not Axiom's signed
production journal/fencing or authorization. Status never enables execution,
capital, owner transport or live-feed claims. Python audit guards are not native
OS network isolation. No process is installed or left running as a production
worker. Provider/collector access, owner endpoint binding and production restart
supervision remain unactivated; parent must define/authorize those exact bindings
before operational continuous monitoring can be claimed. Mac build/validation
remains independently owned and is not inferred from Linux evidence.
