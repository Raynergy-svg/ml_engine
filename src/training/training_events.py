"""Training lifecycle events — the closed-loop wiring.

Publishes TRAINING_COMPLETED to the Tier 7 trading event bus after a
scheduled_retrain run succeeds. Subscribes a promotion handler that calls
`should_promote()`, logs the W&B artifact, and emits MODEL_PROMOTED or
MODEL_HELD.

Why this file exists (per learnings.md 2026-04-15):
    The bot already had a reflection system that proposed config changes — but
    nothing consumed them (4 CRITICAL proposals, 0 applied). The feedback loop
    was write-only dead code. This module is the consumer side: training
    completes → event → handler reads it → decision → follow-up event. Every
    hop is observable in the event JSONL + W&B.

Activation:
    * BUDDY_WANDB_REGISTRY_ENABLED=1   — so artifact logging + promotion run.
    * Call `register_training_handlers()` once at process startup (or it will
      self-register on first publish via a cheap idempotency check).
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HANDLERS_REGISTERED = False
_HANDLERS_LOCK = threading.Lock()


def _get_bus():
    """Lazy import so this module can load even if trading_event_bus changes."""
    from src.scanner.automation.trading_event_bus import get_trading_event_bus
    return get_trading_event_bus()


def _make_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Tier 7 trading-bus envelope with an idempotent event_id."""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "training",
        "session_id": os.getenv("BUDDY_SESSION_ID", "training"),
        "correlation_id": payload.get("correlation_id", str(uuid.uuid4())),
        "payload": payload,
        "payload_version": 1,
    }


# ─────────────────────────────────────────────────────────────────
# Publishers (called by trainers / retrain_agent)
# ─────────────────────────────────────────────────────────────────

def publish_training_completed(
    *,
    pairs: List[str],
    model_dir: Path | str,
    holdout_metrics: Dict[str, Any],
    reason: str = "scheduled",
) -> str:
    """Publish a TRAINING_COMPLETED event. Returns the event_id.

    `holdout_metrics` is the full dict returned by
    `validate_holdout_multi_timeframe` + `verify_ensemble_refreshed` — callers
    should pass it verbatim so downstream handlers see everything the policy
    gates need.
    """
    _ensure_handlers_registered()

    payload = {
        "pairs": list(pairs),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "accuracy": holdout_metrics.get("accuracy"),
        "holdout_samples": holdout_metrics.get("holdout_samples"),
        "timeframes_passed": holdout_metrics.get("timeframes_passed"),
        "timeframes_tested": holdout_metrics.get("timeframes_tested"),
        "max_component_age_days": holdout_metrics.get("max_component_age_days"),
        "ensemble_complete": holdout_metrics.get("ensemble_complete"),
        "ensemble_stale_groups": holdout_metrics.get("ensemble_stale_groups", []),
        "per_timeframe": holdout_metrics.get("per_timeframe", {}),
        "reason": reason,
    }
    event = _make_event("training_completed", payload)
    try:
        _get_bus().emit(event)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to emit TRAINING_COMPLETED: %s", exc, exc_info=True)
    return event["event_id"]


def publish_model_decision(
    *,
    promoted: bool,
    artifact_name: str,
    artifact_version: str,
    reason: str,
    diagnostics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    production_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Publish MODEL_PROMOTED or MODEL_HELD depending on the decision."""
    payload = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "promoted": bool(promoted),
        "reason": reason,
        "decision_diagnostics": diagnostics,
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
    }
    event_type = "model_promoted" if promoted else "model_held"
    event = _make_event(event_type, payload)
    try:
        _get_bus().emit(event)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to emit %s: %s", event_type, exc, exc_info=True)
    return event["event_id"]


# ─────────────────────────────────────────────────────────────────
# Handler: TRAINING_COMPLETED → consult policy → promote or hold
# ─────────────────────────────────────────────────────────────────

def _on_training_completed(event: Dict[str, Any]) -> None:
    """Promote or hold a fresh ensemble based on policy verdict.

    This is THE round-trip handler. It consumes what the trainer published,
    logs the artifact, runs `should_promote`, and emits MODEL_PROMOTED or
    MODEL_HELD so downstream consumers (scanner hot-reload, TUI, human
    approval queue) can react.
    """
    payload = event.get("payload") or {}
    logger.info("training_completed handler fired: pairs=%s reason=%s",
                payload.get("pairs"), payload.get("reason"))

    # Lazy imports so this file can sit alongside training code without cycles.
    from src.training.model_artifact import (
        is_enabled,
        log_model_artifact,
        get_production_metrics,
        promote_to_production,
        mark_staging,
    )
    from src.training.promotion_policy import should_promote

    if not is_enabled():
        logger.info("W&B registry disabled — policy evaluation skipped. "
                    "Set BUDDY_WANDB_REGISTRY_ENABLED=1 to activate.")
        return

    artifact_name = "buddy-joint-models"
    candidate_metrics = {
        "accuracy": payload.get("accuracy"),
        "holdout_samples": payload.get("holdout_samples"),
        "timeframes_passed": payload.get("timeframes_passed"),
        "timeframes_tested": payload.get("timeframes_tested"),
        "max_component_age_days": payload.get("max_component_age_days"),
        "ensemble_complete": payload.get("ensemble_complete"),
        "ensemble_stale_groups": payload.get("ensemble_stale_groups", []),
        "pairs": len(payload.get("pairs", [])),
    }
    metadata = {
        "pairs": payload.get("pairs", []),
        "trained_at": payload.get("trained_at"),
        "model_dir": payload.get("model_dir"),
        "per_timeframe": payload.get("per_timeframe", {}),
        "triggering_reason": payload.get("reason", "scheduled"),
        "source_event_id": event.get("event_id"),
    }

    # 1. Log artifact as :staging.
    ref = log_model_artifact(
        name=artifact_name,
        paths=[Path(payload.get("model_dir", "."))],
        metrics={k: v for k, v in candidate_metrics.items() if isinstance(v, (int, float, bool))},
        metadata=metadata,
        aliases=["staging"],
    )
    if ref is None:
        logger.warning("Artifact log returned None — skipping promotion check")
        return
    mark_staging(ref)

    # 2. Compare vs current production, consult policy.
    production = get_production_metrics(artifact_name, alias="production")
    decision = should_promote(candidate=candidate_metrics, production=production)

    logger.info("=" * 60)
    logger.info("PROMOTION POLICY VERDICT")
    logger.info("  artifact          : %s", ref.qualified)
    logger.info("  candidate metrics : %s", candidate_metrics)
    logger.info("  production metrics: %s", production)
    logger.info("  decision          : %s", "PROMOTE" if decision.promote else "HOLD")
    logger.info("  reason            : %s", decision.reason)
    logger.info("=" * 60)

    # 3. Apply + publish outcome event.
    if decision.promote:
        ok = promote_to_production(ref)
        publish_model_decision(
            promoted=bool(ok),
            artifact_name=artifact_name,
            artifact_version=ref.version,
            reason=decision.reason if ok else f"policy passed but promote_to_production failed",
            diagnostics=decision.diagnostics,
            candidate_metrics=candidate_metrics,
            production_metrics=production,
        )
    else:
        publish_model_decision(
            promoted=False,
            artifact_name=artifact_name,
            artifact_version=ref.version,
            reason=decision.reason,
            diagnostics=decision.diagnostics,
            candidate_metrics=candidate_metrics,
            production_metrics=production,
        )


# ─────────────────────────────────────────────────────────────────
# Registration (idempotent)
# ─────────────────────────────────────────────────────────────────

def register_training_handlers() -> None:
    """Subscribe the promotion handler to TRAINING_COMPLETED. Idempotent."""
    global _HANDLERS_REGISTERED
    with _HANDLERS_LOCK:
        if _HANDLERS_REGISTERED:
            return
        try:
            bus = _get_bus()
            bus.subscribe(
                event_type="training_completed",
                handler_fn=_on_training_completed,
                priority=50,
                handler_name="promotion_policy_handler",
            )
            _HANDLERS_REGISTERED = True
            logger.info("Training handlers registered on trading event bus")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to register training handlers: %s", exc, exc_info=True)


def _ensure_handlers_registered() -> None:
    if not _HANDLERS_REGISTERED:
        register_training_handlers()
