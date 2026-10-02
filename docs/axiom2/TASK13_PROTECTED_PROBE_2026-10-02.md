# Task 13 protected Robinhood probe and remaining integration blockers

Observed 2026-10-02 approximately 01:46–01:51 UTC (2026-10-01 evening America/New_York).
Source reviewed: 93068428eaf32f934a851b622306686a7685f4c7 on codex/axiom-2-checkpoint-recovery-efc9554.
This report contains no account identifiers, credential material, order identities or financial amounts.

## Corrected connection diagnosis

The Robinhood ChatGPT plugin is connected. A fresh protected get_accounts call
succeeded and identified one active caller-accessible dedicated Agentic account.
Five further protected calls succeeded against that explicit account binding:
get_portfolio, get_equity_positions, get_equity_orders, get_equity_quotes (SPY)
and get_equity_tradability (SPY). Equity positions and orders were empty terminal
pages; their next cursor was omitted. No nonempty or multipage path was exercised.

Buying power and pending deposits were independently reported. Pending deposits
must not be substituted for settled cash or buying power. The SPY response included
nanosecond bid/ask timestamps, an after-hours trade timestamp and a separate
official settled close. Account-type equity tradability was reported. The read
occurred outside regular hours and does not qualify a regular-session observation.

The six current declarations match the six pinned declarations exactly after
removing only the platform-added sentence identifying the Robinhood Trading
plugin. No schema/parameter semantic drift was found. Do not repin the adapter
because of presentation attribution alone. This comparison is declaration
inspection, not a run of the Python adapter or its hash verification script.

## Root cause and safe correction path

Connection success in ChatGPT is not authentication or callable bindings in the
native Axiom Python service. The recovered read-only implementation already
documents this boundary; TrustedReadHost accepts six exact trusted callable
bindings but neither it nor this report creates them. No Codex dispatch/control
tool is exposed to this chat. The user's native Codex installation may be
connected independently; this chat cannot infer its checkout or runtime state.

The earlier RESUME_CHECKPOINT_2026-10-01.md statement that no protected tools
were exposed is historical, not current. Protected read access is now verified
for this ChatGPT caller, while end-to-end Axiom service access remains unverified.

The exposed Robinhood inventory supplies no read for a machine-verifiable broker
trade-approval setting, authoritative settled/reserved cash, common atomic
account snapshot token/time, signed external-flow/corporate-action history,
advanced equity order completeness, or exchange holiday/session calendar.
The earnings calendar is not an exchange session calendar. Account eligibility
and ChatGPT confirmation permissions cannot replace broker approval state.
Fresh quote timestamps cannot replace account snapshot provenance.

## Native Codex continuation

1. Discover the native checkout, applicable instructions, branch, HEAD and dirty
   state; preserve unrelated and concurrently running agent work. Inspect the
   recovered commit and current Task 13 files before editing.
2. Verify the exact six protected read callables available to that native
   runtime. A ChatGPT connection report or replayed response is not a native
   live credential grant. If native protected bindings exist, bind the explicit
   dedicated account privately and invoke TrustedReadHost/RobinhoodReadOnly
   through their existing methods; do not build a generic MCP forwarder.
3. Run the focused adapter tests and existing complete Axiom/evidence dependency
   gate in the established environment. Produce alias-only live audit evidence,
   with local read windows distinct from provider source times.
4. Check provider gaps against actual additional official read capabilities.
   Preserve explicit UNKNOWN/BLOCKED states for missing source facts. Do not
   synthesize settled cash, atomic snapshots, corporate-action closure,
   approval state or regular-session/calendar qualification.
5. Exercise safe negative paths with controlled fixtures and distinguish them
   from actual live revocation/capability drift. Do not revoke a working user
   connection merely to generate a test result. Do not create positions or
   orders to force nonempty pages.
6. Keep Task 13 incomplete until all its live requirements are proved, or obtain
   an explicit reviewed plan change for the inaccessible provider requirements.
   Do not silently weaken the task or the later capital admission gates.

## Verification scope

IMPLEMENTED: existing fixed read-only adapter/service composition in reviewed source.
LIVE READ-ONLY VERIFIED: six protected ChatGPT reads and the empty terminal page branch.
NATIVE SERVICE LIVE VERIFIED: no.
NONEMPTY/MULTIPAGE/REVOCATION VERIFIED: no.
TASK 13 COMPLETE: no.
SHADOW QUALIFIED: no.
EXECUTION ENABLED / CAPITAL AUTHORIZED: false.

No code change, test execution, real-order review/preview, placement, cancellation,
watchlist write, approval change, credential transfer, holdout consumption, merge,
deployment or funding action occurred. This is a sanitized human-readable
observation report, not a signed Axiom readiness/promotion/capital receipt.
