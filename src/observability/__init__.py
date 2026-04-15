"""Observability layer for Buddy (traces, metrics, model lineage).

Currently provides Weave (W&B) tracing via `weave_tracing.op` decorator and
`init_weave()` bootstrap. All integrations are optional and env-gated — if the
underlying package is missing or disabled, decorators become no-ops so the
trading loop never fails because of observability.
"""

from src.observability.weave_tracing import (
    init_weave,
    op,
    thread,
    attributes,
    is_enabled,
    patch_instance_methods,
)

__all__ = [
    "init_weave",
    "op",
    "thread",
    "attributes",
    "is_enabled",
    "patch_instance_methods",
]
