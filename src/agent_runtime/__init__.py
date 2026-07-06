"""AXIOM Agent Runtime — foundation layer for a future resident AI operator.

This package is scaffolding ONLY: a tool registry (src.agent_runtime.tools),
a 3-tier action-policy engine (src.agent_runtime.policy), and an audit trail
(src.agent_runtime.audit) that extends the existing dashboard control-audit
log. There is no LLM reasoning loop here and nothing in this package can
place an order, arm the live gate, or unhalt trading — see policy.py's
ActionTier.ESCALATION contract for the structural guarantee.
"""
