"""AXIOM subscription-backed operator runtime.

The operator is a logical session persisted to disk. Individual Claude CLI
invocations are disposable epochs so the runtime can roll context forward
without depending on one immortal chat process.
"""

from src.axiom_operator.policy import ActionPolicy, PolicyDecision
from src.axiom_operator.runner import AxiomOperator, OperatorRunResult
from src.axiom_operator.session import (
    DECISIONS_PATH,
    SESSION_PATH,
    default_session,
    read_operator_snapshot,
)

__all__ = [
    "ActionPolicy",
    "AxiomOperator",
    "DECISIONS_PATH",
    "OperatorRunResult",
    "PolicyDecision",
    "SESSION_PATH",
    "default_session",
    "read_operator_snapshot",
]
