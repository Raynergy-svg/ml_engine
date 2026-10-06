# Nautilus runtime review — 2026-10-06

## Scope and provenance

This is the bounded upstream review for the Nautilus integration branch
codex/axiom2-nautilus-runtime-2026-10-06. The Axiom branch is based on
64bf4d3780bc193bf122f830b1df04027ba8fe47; the paused LEAN branch remains
separate and is not an input to this work.

The audited upstream candidate is:

| Field | Value |
|---|---|
| Repository | nautechsystems/nautilus_trader |
| Source revision | 4f021bafc2e99c5490cee204b0fc2bd2c83baab4 |
| Upstream package version at revision | 2.0.0rc7 |
| Artifact identity | Source checkout; unmodified upstream make build; maturin release extension; wheel name nautilus_trader-2.0.0rc7-cp312-cp312-linux_x86_64.whl |
| Enabled features | arrow, ffi, python, high-precision, streaming, defi |
| License | LGPL-3.0-only; upstream LICENSE is GNU LGPL version 3 |
| Runtime role | Event/runtime mechanics only; Axiom retains policy, provenance, authority, and persistence ownership |

No matching Nautilus component review or Nautilus implementation was present
in the tracked Axiom base. Existing Axiom execution-kernel, offline-service,
controlled-execution, protected-probe, and phase-orchestrator audits were
reused as the authority boundary and regression references.

## Actual build requirements and evidence

The exact upstream python/pyproject.toml, rust-toolchain.toml, and Makefile
require or declare:

- Python >=3.12,<3.15.
- Rust 1.99.0.
- uv>=0.12,<0.13, with the source workflow using make sync.
- maturin==1.15.0 and patchelf.
- A native build toolchain including clang/lld on Linux.
- Source build through upstream make build, which runs upstream stub
  generation and maturin develop --release --locked. No source files from
  Nautilus are copied into Axiom.

The first completed source-build run, 37516990231, built the exact source
revision successfully in 39 minutes 24 seconds. It installed the
2.0.0rc7 editable package and produced the wheel identity recorded above.
The Axiom contract tests passed: 2 passed in 0.25 seconds.

That first run exposed a smoke-harness assumption, not a build failure:
the harness expected Clock.set_time() to dispatch a queued timer callback,
but the public Python Clock surface only advances the virtual timestamp.
The audited public Python API exposes no advance_time method. The corrected
harness now verifies timer registration, timer count, next scheduled time,
and deterministic timestamp advancement without claiming callback dispatch.
The corrected smoke run is independently tracked and must finish before
smoke success is claimed. No Rust bridge was added because the required
observation-boundary behavior is available without timer callback dispatch.

## Public API findings

The smoke harness and implementation use only public supported Python
interfaces:

- Clock.new_test(), set_timer_ns, and set_time for deterministic clock state.
- MessageBus(TraderId, clock=...), subscribe, and publish for synchronous
  event delivery.
- DataActor.start(), is_running(), stop(), and is_stopped() for lifecycle
  start/stop.
- Public model constructors for OrderSubmitted, OrderAccepted, OrderRejected,
  OrderCanceled, OrderCancelRejected, and OrderFilled.
- Public LimitOrder.apply(event) for offline order-event replay.

The order constructor probe and lifecycle adapter are deliberately limited:
they prove that public model and state-machine APIs can be called with valid
synthetic identifiers. They do not claim that a configuration object or
model-only replay is an execution engine. No broker or execution client is
created.

## Harness safety boundary

The initial harness and CI job:

- use only synthetic/replay data;
- provide no broker credentials or execution-client configuration;
- provide a Python socket guard that fails on attempted external connections;
- do not create ExecutionClient, DataClient, a live node, or a broker adapter;
- never submit, cancel, amend, or authorize an order;
- report READY only as a revocable candidate state in later Axiom code;
- do not claim live-feed connectivity.

The upstream LICENSE and security guidance are retained as notices in the
provenance record; the dependency remains unmodified and pinned.

## Implemented bounded path

The implemented path is:

admitted observation -> Nautilus public MessageBus/DataActor runtime -> Axiom
candidate policy -> durable material event -> bounded ResearchWakeup -> Axiom
revalidation

Candidate state supports WATCHING, WAITING, TRIGGERED, REVALIDATING, READY,
INVALIDATED, and EXPIRED. State and pending wakeup rows commit together in
SQLite WAL mode with FULL synchronous durability. Pending and in-progress
wakeups survive restart; duplicate consumption is durably idempotent. No
exactly-once delivery claim is made.

The order replay path separately persists raw upstream event payloads and
dispositions, then rebuilds a Nautilus LimitOrder from APPLIED reports after
restart. Nautilus-derived snapshots never become Axiom execution authority.

## Integration boundary and open gaps

A deterministic recorded/synthetic replay demonstrates runtime behavior, not
live market connectivity. Live-feed connectivity, broker reconciliation,
credential handling, and any production execution path remain explicit gaps
and are outside this authorization.

All production execution gates, approval requirements, capital limits, risk
checks, and execution permissions remain unchanged.
