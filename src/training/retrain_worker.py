"""Retrain Worker — drains TRAINING_PROPOSED events and spawns training runs.

Design:
- Event handler returns fast (spawns subprocess, doesn't wait).
- At-most-one in-flight retrain enforced via _active lock.
- The subprocess is `scripts/scheduled_retrain.py` with `--pairs` from payload;
  we don't reimplement training here — scheduled_retrain already handles model
  persistence, validation, ensemble-completeness, and publishing
  TRAINING_COMPLETED, which closes the loop via the promotion handler.
- Worker output is persisted to `logs/retrain_worker_<ts>.log`.

Gate: `BUDDY_RETRAIN_WORKER_ENABLED` — keep disabled in dev so the bus can
fire proposals without actually launching 20-minute trainings. Activate in
production once the full chain has been exercised.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _PROJECT_ROOT / "logs"

_ENV_FLAG = "BUDDY_RETRAIN_WORKER_ENABLED"
_REGISTERED = False
_lock = threading.Lock()
_active_lock = threading.Lock()
_active_process: Optional[subprocess.Popen] = None
_active_event_id: Optional[str] = None


def _env_truthy(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _worker_enabled() -> bool:
    return _env_truthy(os.getenv(_ENV_FLAG))


def on_training_proposed(event: Dict[str, Any]) -> None:
    """Handler for TRAINING_PROPOSED. Spawns scheduled_retrain as subprocess."""
    payload = event.get("payload") or {}
    reason = payload.get("reason", "unknown")
    pairs = payload.get("pairs") or []
    priority = payload.get("priority", "normal")
    event_id = event.get("event_id") or str(uuid.uuid4())

    log = logger.getChild("worker")
    log.warning(
        "TRAINING_PROPOSED received: reason=%s priority=%s pairs=%d event_id=%s",
        reason, priority, len(pairs), event_id,
    )

    if not _worker_enabled():
        log.info("Retrain worker disabled (set %s=1 to activate). Proposal logged only.", _ENV_FLAG)
        _publish_worker_status(
            event_id=event_id,
            status="skipped",
            reason=f"{_ENV_FLAG} not set",
        )
        return

    global _active_process, _active_event_id
    with _active_lock:
        if _active_process is not None and _active_process.poll() is None:
            log.info("Retrain already in-flight (event=%s). Skipping proposal %s.",
                     _active_event_id, event_id)
            _publish_worker_status(
                event_id=event_id,
                status="dropped",
                reason=f"retrain in-flight event={_active_event_id}",
            )
            return

        # Spawn subprocess
        _LOG_DIR.mkdir(exist_ok=True)
        log_path = _LOG_DIR / f"retrain_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_id[:8]}.log"
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "scheduled_retrain.py"),
        ]
        if pairs:
            cmd += ["--pairs", ",".join(pairs)]
        env = dict(os.environ)
        env.setdefault("BUDDY_SESSION_ID", f"retrain_{event_id[:8]}")

        try:
            log_f = open(log_path, "wb")
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(_PROJECT_ROOT),
            )
            _active_process = proc
            _active_event_id = event_id
            log.warning("Spawned scheduled_retrain pid=%s log=%s", proc.pid, log_path)

            # Detached watcher thread clears _active_process when subprocess exits.
            threading.Thread(
                target=_watch_subprocess,
                args=(proc, event_id, log_f, log_path, reason, pairs),
                daemon=True,
            ).start()
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to spawn scheduled_retrain: %s", exc, exc_info=True)
            _publish_worker_status(
                event_id=event_id,
                status="failed",
                reason=f"spawn error: {exc}",
            )


def _watch_subprocess(
    proc: subprocess.Popen,
    event_id: str,
    log_file,
    log_path: Path,
    reason: str,
    pairs: List[str],
) -> None:
    """Waits for subprocess exit, publishes worker status event."""
    global _active_process, _active_event_id
    rc = proc.wait()
    try:
        log_file.close()
    except Exception:
        pass
    with _active_lock:
        if _active_event_id == event_id:
            _active_process = None
            _active_event_id = None

    logger.warning("scheduled_retrain (event=%s) exited rc=%s log=%s", event_id, rc, log_path)
    _publish_worker_status(
        event_id=event_id,
        status="completed" if rc == 0 else "failed",
        reason=f"subprocess exit rc={rc}",
        extras={"log_path": str(log_path), "trigger_reason": reason, "pairs": pairs},
    )


def _publish_worker_status(
    *,
    event_id: str,
    status: str,
    reason: str,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a training_started / failure-equivalent event so the bus has a trail."""
    try:
        from src.scanner.automation.trading_event_bus import get_trading_event_bus

        evt_type = {
            "skipped":   "training_started",  # logged only; worker disabled
            "dropped":   "training_started",  # in-flight dupe
            "completed": "training_started",  # subprocess finished ok; TRAINING_COMPLETED fires inside
            "failed":    "queue_failure",
        }.get(status, "training_started")

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": evt_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "retrain_worker",
            "session_id": os.getenv("BUDDY_SESSION_ID", "retrain_worker"),
            "correlation_id": event_id,
            "payload": {
                "proposal_event_id": event_id,
                "status": status,
                "reason": reason,
                **(extras or {}),
            },
            "payload_version": 1,
        }
        get_trading_event_bus().emit(event)
    except Exception as exc:  # noqa: BLE001
        logger.debug("worker status emit failed: %s", exc)


def register_retrain_worker() -> None:
    """Subscribe worker to TRAINING_PROPOSED. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    with _lock:
        if _REGISTERED:
            return
        from src.scanner.automation.trading_event_bus import get_trading_event_bus
        bus = get_trading_event_bus()
        bus.subscribe(
            event_type="training_proposed",
            handler_fn=on_training_proposed,
            priority=30,  # runs before other consumers
            handler_name="retrain_worker",
        )
        _REGISTERED = True
        logger.info("retrain_worker: subscribed to TRAINING_PROPOSED (worker_enabled=%s)", _worker_enabled())
