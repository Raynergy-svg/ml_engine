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
| Artifact identity | Source checkout plus upstream make build/maturin extension; captured by the smoke report |
| License | LGPL-3.0-only; upstream LICENSE is GNU LGPL version 3 |
| Runtime role | Event/runtime mechanics only; Axiom retains policy, provenance, authority, and persistence ownership |

No matching Nautilus component review or Nautilus implementation was present
in the tracked Axiom base. Existing Axiom execution-kernel, offline-service,
controlled-execution, protected-probe, and phase-orchestrator audits were
reused as the authority boundary and regression references.

## Actual build requirements

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

The declared extension feature set at this revision is recorded in
config/axiom2/nautilus_runtime.json. The workflow records the actual default
feature selection emitted by upstream scripts/cargo-features.bash alongside
the installed artifact.

## Public API findings

The first smoke harness uses only public supported Python interfaces:

- Clock.new_test(), set_timer_ns, and set_time for deterministic timer
  delivery.
- MessageBus(TraderId, clock=...), subscribe, and publish for synchronous
  event delivery.
- DataActor.start(), is_running(), stop(), and is_stopped() for lifecycle
  start/stop.
- Public model constructors for OrderSubmitted, OrderAccepted, OrderRejected,
  OrderCanceled, OrderCancelRejected, and OrderFilled.

The order constructor probe is deliberately limited: it proves those public
model APIs can be called with valid synthetic identifiers. It does not claim
that a configuration object or model-only replay is an execution engine.
The bounded runtime implementation must use the upstream event path and
lifecycle interfaces; no broker or execution client is created in the smoke
harness. A minimal Rust bridge is not justified by the audited Python surface.

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

## Build/import/smoke evidence

The first real source-build workflow is
.github/workflows/axiom2-nautilus-runtime.yml. Its result is intentionally
recorded after the run completes. Until then, this section is a pending
verification item rather than a success claim.

Expected report fields are source revision, distribution/version, extension
path, enabled features, license, event delivery, deterministic timer
delivery, lifecycle state transitions, callable order constructors, and
network/credential safety assertions.

## Integration boundary and open gaps

The planned implementation is:

admitted observation -> Nautilus public event runtime -> Axiom candidate
policy -> durable material event -> bounded ResearchWakeup -> Axiom
revalidation

A deterministic recorded/synthetic replay demonstrates runtime behavior, not
live market connectivity. Live-feed connectivity, broker reconciliation,
credential handling, and any production execution path remain explicit gaps
and are outside this authorization.

All production execution gates, approval requirements, capital limits, risk
checks, and execution permissions remain unchanged.
