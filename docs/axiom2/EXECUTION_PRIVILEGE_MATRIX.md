# Execution privilege matrix

Task17 operational isolation remains **BLOCKED**. A shared Mac development identity cannot prove hostile-agent isolation. Source scans, monkeypatches, local tests and claimed JSON flags are insufficient.

| Identity | Permitted | Denied |
|---|---|---|
| Research, LLM, development, healing | research inputs, candidates, diagnostics | production secrets, gateway orders, provider egress |
| Evidence/promotion | authenticated immutable evidence and signed decisions | broker credentials and order tools |
| Portfolio/risk | authenticated observations and bounded intents | broker sessions and direct submission |
| Read-only broker service | fixed reads and minimized snapshots | broad tokens exposed downstream and order capability |
| Shadow service | hypothetical fills and journals | review, submit, cancel |
| Sole production gateway | separately approved authenticated bounded intent | arbitrary research code and unknown tools |

Deployment must use separate OS/service identities and pin secret mounts, gateway identity, exact tool allowlists, provider egress and unknown-tool default denial. Demonstrate actual cross-process credential/filesystem/network denials, forged roles, broad MCP forwarding, alternate gateways, stale authorization, accidental deployment imports, fault injection, revocation, kill and restart reconciliation.

A signed ExecutionBoundaryReceipt must bind exact build/artifact bytes, service identities, secret mounts, tool/network policies, gateway configuration, operational probes and independently trusted verifier role/key. Material build or policy changes invalidate it. No operational verifier or signed boundary receipt is configured by this work. The local script always returns BLOCKED (exit 2), including with supplied evidence. Task17 completion is not claimed.
