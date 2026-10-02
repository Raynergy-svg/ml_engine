# Official Robinhood read-only adapter checkpoint

This is software verification, not Task 13 completion or trading authorization.
No private account identifier, financial value, position or live order is included
in the captured declarations, tests or this report. All fixture values are
hand-authored synthetic examples. The original checkout remains untouched.

The parent reported successful protected reads of accounts, dedicated-account
portfolio, empty equity positions/orders, equity quotes and tradability on
2026-10-01 UTC. Empty terminal pages omitted `next`. Nonempty and multipage
provider branches, revocation and error behavior were not live exercised. This
worker made no provider calls and used the available exact six tool declarations
plus the parent's sanitized semantic verification packet.

## Implemented scope

`ReadOnlyBrokerPort` has fixed capabilities, market, account and order observation
methods. Trusted service configuration supplies exactly six callable official
read bindings and an explicit raw-account/alias mapping; no account is selected
by default, nickname or response order. The adapter has no arbitrary forwarding,
credential files, URLs, broker review/preview/order/cancel or permission changes.
Bindings are trusted service inputs, not a hostile-code sandbox or standalone
proof that Python has the ChatGPT connector's access.

Declared metadata is pinned by SHA256. Successful structured MCP responses must
validate the expected schema; text and guides never drive policy or execution.
Provider failures are sanitized reason codes. Read timeouts and pagination have
finite budgets, duplicate identities/cursors and incomplete pages fail closed,
and no failed read returns cached connected/readiness state. Cursor strings are
opaque and forwarded verbatim with the same explicitly bound account. Traversal
completeness means ordinary equity-query pages only, not an atomic snapshot or
complete broker activity. Nullable rows are explicit schema gaps.

Normalized immutable observations retain local UUID/read-start/receive windows.
They never substitute local times for broker source time. Quotes retain separate
bid/ask raw nanosecond strings, exact age checks, newer regular/nonregular trade
selection, stale-trade indication and missing official-close facts. Order/fill
source times retain nanoseconds; creation/update/fill chronology uses exact
rational comparisons. Money conversion is exact and independent of ambient
Decimal precision. Unsupported subunit precision fails rather than rounding.
Fractional and zero/delisted positions remain present, sellable quantity is the
server field, and missing/null average cost is distinguished. Buying power null
is distinct from zero. Cash, pending deposits, unleveraged buying power and lagging
unsettled funds remain separate reported facts; none becomes settled cash.

Unknown order states are preserved and marked unsupported. Dollar-based orders,
partial/cancelled fills and missing broker references retain their facts. Broker
`ref_id` is explicitly an unverified broker reference, not the application's
original request identity. Execution detail completeness requires matching
quantities/fees and chronological coverage; contradictory or missing details
cannot establish complete reconciliation.

The offline `axiom2_verify_readonly.py` probe validates only the declaration
artifact's exact hash and inventory. PASS creates no connector call and does not
establish connection, approval, settlement, live branch coverage or deployment
isolation. Public execution routes remain disabled and `as_risk_account()` always
denies conversion while authoritative account prerequisites are absent.

## Verification and external gaps

RED: 33 tests failed before the module existed. Two review-driven decimal/detail
regressions then failed before fixes; four nanosecond ordering regressions also
failed before fixes. Final focused gate: 60 passed in 0.40 seconds. Full
Axiom/evidence gate: 1300 passed in 30.24 seconds, warnings as errors and plugin
autoload disabled. Three independent read-only reviewers cleared the final scope
without performing their own test runs or provider calls. Research isolation PASS
retains SHA256 `1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74`.
The fresh standalone execution ZIP imports the adapter and retains zero broker
actions/capital disabled; this is not operational proof.

Unavailable provider facts remain explicit blocked prerequisites: common broker
snapshot time/token, settled and reserved cash, signed external-flow and corporate
action histories, advanced-order coverage, regular-session/holiday calendar and
machine-verifiable broker trade-approval setting. No corresponding approval,
calendar or advanced-order read tool/resource was discovered by the parent.
ChatGPT app confirmation permissions and account caller eligibility cannot fill
these gaps. Human confirmation of approval settings would remain distinct from
machine proof. Future authentic observations, live pagination/nonempty/revocation
verification, twenty regular shadow sessions/four completed cycles and actual
service identity/secret/tool/network denial evidence are separate prerequisites.
No holdout consumption, financial write, preview, access/security change,
publication, merge or deployment occurred in this checkpoint.
