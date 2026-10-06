# Axiom 2.0 Nautilus order-lifecycle integration — 2026-10-06

This increment is offline and replay-only. It feeds synthetic order reports
through the public LimitOrder.apply(event) interface from the pinned Nautilus
source revision 4f021bafc2e99c5490cee204b0fc2bd2c83baab4.

The adapter keeps raw_order_observations durable and separate from the
Nautilus-derived OrderSnapshot. On restart it creates a new LimitOrder and
replays only raw rows marked APPLIED. Duplicate report identities are stored
as DUPLICATE; reused identities with different bytes are stored as CONFLICT;
reports rejected by the actual Nautilus model are stored as REJECTED. These
dispositions are observations about the replay boundary, not provider truth.

The scenario covers:

- submitted/acknowledged;
- partial fill and full fill;
- cancellation request, cancellation rejection, and cancellation;
- late fill after cancellation;
- same-report duplicate, conflicting same-report bytes, and Nautilus duplicate-trade rejection;
- process restart and state reconstruction.

The equivalent Axiom lifecycle path is checked against the existing transition
table, and canonical accepted fills are passed through the existing pure
reconciliation function. A conflicting observation is expected to halt
reconciliation, preserving the discrepancy for investigation instead of
silently selecting either implementation.

No order is submitted. No execution client, broker, credential, capital
authorization, or readiness grant is created. Live-feed connectivity remains
out of scope; this is deterministic synthetic replay.
