"""MODEL_PROMOTED event handler — hot-reloads models in the running scanner.

When a retrain completes and the promotion policy approves the candidate,
this handler triggers Scanner.reload_models() so the live process picks up
fresh models without a restart.

Also handles TRAINING_COMPLETED (regardless of promotion) by reloading
agent weights — the retrain writes fresh weights even when the model isn't
promoted.

Subscription:
    Call register_model_reload_handler(scanner) during Scanner or TUI init.
    The handler keeps a weak reference to the Scanner so it doesn't prevent GC.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_REGISTERED = False
_SCANNER_REF: Optional[weakref.ref] = None
_LOCK = threading.Lock()


def _on_model_promoted(event: Dict[str, Any]) -> None:
    """Tier 7 bus handler: MODEL_PROMOTED → Scanner.reload_models()."""
    payload = event.get("payload") or {}
    promoted = payload.get("promoted", False)
    artifact = payload.get("artifact_name", "unknown")
    version = payload.get("artifact_version", "?")
    reason = payload.get("reason", "")

    if not promoted:
        logger.debug("model_reload_handler: model_promoted event but promoted=False, skipping")
        return

    scanner = _SCANNER_REF() if _SCANNER_REF is not None else None
    if scanner is None:
        logger.warning("model_reload_handler: no scanner reference — can't hot-reload")
        return

    logger.warning(
        "model_reload_handler: MODEL_PROMOTED %s:%s — triggering hot-reload. reason=%s",
        artifact, version, reason,
    )
    try:
        ok = scanner.reload_models(reason=f"model_promoted:{artifact}:{version}")
        if ok:
            logger.info("model_reload_handler: hot-reload succeeded")
        else:
            logger.error("model_reload_handler: hot-reload returned False")
    except Exception as exc:
        logger.error("model_reload_handler: hot-reload failed: %s", exc, exc_info=True)


def _on_training_completed(event: Dict[str, Any]) -> None:
    """Tier 7 bus handler: TRAINING_COMPLETED → reload agent weights.

    Even when a model isn't promoted, the retrain writes fresh agent_weights.json
    and other derived state. Reload weights so the running agents benefit.
    """
    scanner = _SCANNER_REF() if _SCANNER_REF is not None else None
    if scanner is None:
        return

    if hasattr(scanner, '_agent_team') and scanner._agent_team is not None:
        try:
            scanner._agent_team.reload_learned_weights()
            logger.info("model_reload_handler: agent weights refreshed (TRAINING_COMPLETED)")
        except Exception as exc:
            logger.debug("model_reload_handler: agent weight reload failed: %s", exc)


def register_model_reload_handler(scanner: Any) -> None:
    """Subscribe the hot-reload handler to MODEL_PROMOTED + TRAINING_COMPLETED.

    Args:
        scanner: Scanner instance to call reload_models() on.
                 Stored as a weak reference.
    """
    global _REGISTERED, _SCANNER_REF
    with _LOCK:
        if _REGISTERED:
            # Update scanner ref even if already registered (TUI may restart scanner)
            _SCANNER_REF = weakref.ref(scanner)
            return
        try:
            from src.scanner.automation.trading_event_bus import get_trading_event_bus
            bus = get_trading_event_bus()
            bus.subscribe(
                event_type="model_promoted",
                handler_fn=_on_model_promoted,
                priority=20,  # high priority — reload before other consumers react
                handler_name="model_hot_reload",
            )
            bus.subscribe(
                event_type="training_completed",
                handler_fn=_on_training_completed,
                priority=90,  # low priority — after promotion handler (50)
                handler_name="model_weight_refresh",
            )
            _SCANNER_REF = weakref.ref(scanner)
            _REGISTERED = True
            logger.info("model_reload_handler: registered on MODEL_PROMOTED + TRAINING_COMPLETED")
        except Exception as exc:
            logger.error("model_reload_handler: registration failed: %s", exc)
