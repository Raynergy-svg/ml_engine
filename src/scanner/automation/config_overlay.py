"""Durable config overlay — the headless consumer for approved config adjustments.

Closes TWO structural defects from the 2026-07-03 training-architecture audit:

1. **Producer-alive / consumer-dead asymmetry**: self-heal (running headless)
   stages adjustments that only the TUI scan loop could apply. With no TUI open,
   approved adjustments rotted in ``config_adjustments.json`` forever.
2. **Restart loss**: ``ConfigAdjuster`` marks entries consumed via persisted
   ``applied_ids``, but the *mutation* lived only in the consuming process's
   in-memory ``ScannerConfig`` — a restarted process constructed defaults and
   skipped the already-consumed ids, silently losing every past adjustment.

The overlay is the durable middle: ``consume_approved_adjustments()`` moves
approved entries into ``.claude/config_overlay.json`` (key → value, with
provenance), and ``apply_overlay(config)`` replays the CURRENT overlay onto any
freshly-constructed ScannerConfig. Adjustments therefore survive restarts and
reach every process that applies the overlay at startup.

SAFETY / ROLLOUT:
- Consumption is OPT-IN (``TIER7_CONSUME_ADJUSTMENTS=1`` on the supervisor).
  Until the engine/TUI startup path calls ``apply_overlay``, headless
  consumption would HIDE adjustments from a TUI session (it skips consumed
  ids and doesn't know the overlay yet) — so the default stays off. Flipping
  it on is the operator's wiring decision, together with the one-line
  ``apply_overlay(config)`` call at engine init.
- Orphan-key protection (the $3,527 dead-write incident): every key is
  re-validated against ScannerConfig dataclass fields at BOTH consumption and
  application time; unknown keys are skipped and surfaced, never setattr'd.
- This module never touches halt state, the environment pin, or any order
  path. It routes through the existing approval flow (``AdjustmentApprover``
  writes the approved history; the bypass guard in ``ConfigAdjuster`` stays).
- WIRING ORDER (for the engine seam): ``apply_overlay(config)`` must run AFTER
  ``apply_profile`` at startup — profile dicts write overlapping keys and would
  clobber overlay values if applied second. (Verifier-flagged, 2026-07-03.)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OVERLAY_REL = ".claude/config_overlay.json"

# Belt-and-suspenders mirror of adjustment_validator.PROTECTED_FIELDS (L-003):
# even if a protected key somehow reached the approved history (validator
# bypass, hand-edited file), the overlay refuses to carry or apply it. The
# overlay makes adjustments reach EVERY process — exactly the propagation
# channel a Hard-NO field must never ride.
PROTECTED_KEYS = frozenset({"oanda_environment"})


def _scanner_config_field_names() -> set:
    from src.scanner.config import ScannerConfig
    return {f.name for f in dataclass_fields(ScannerConfig)} - PROTECTED_KEYS


def _read_overlay(root: Path) -> Dict[str, Any]:
    path = root / OVERLAY_REL
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("values"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"version": 1, "values": {}, "provenance": {}, "cycle": 0,
            "last_updated": None}


def _write_overlay(root: Path, overlay: Dict[str, Any]) -> Path:
    target = root / OVERLAY_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay["last_updated"] = datetime.now(timezone.utc).isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(overlay, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def consume_approved_adjustments(root: Path) -> Dict[str, Any]:
    """Move approved-but-unconsumed adjustments into the durable overlay.

    Reuses ``ConfigAdjuster.apply_adjustments`` verbatim (rate limiting,
    applied_ids dedup, bypass guard) — the ``config`` it mutates is a plain
    namespace pre-loaded with current overlay values, so ``old_value``
    comparisons behave and nothing in-process depends on the result.
    Returns a summary dict; never raises (callers run in the supervisor tick).
    """
    try:
        from src.scanner.automation.config_adjuster import ConfigAdjuster

        overlay = _read_overlay(root)
        valid_keys = _scanner_config_field_names()

        adjuster = ConfigAdjuster(
            persistence_path=root / ".claude" / "config_adjustments.json",
            pending_path=root / ".claude" / "pending_adjustments.json",
        )
        target = SimpleNamespace(**overlay["values"])
        cycle = int(overlay.get("cycle", 0)) + 1
        applied = adjuster.apply_adjustments(target, current_cycle=cycle)

        landed: List[str] = []
        skipped: List[str] = []
        for record in applied:
            key = record.get("key")
            if key in PROTECTED_KEYS:
                skipped.append(str(key))
                logger.error(
                    "config_overlay: PROTECTED key %r REFUSED (Hard NO, L-003) — "
                    "an approved adjustment reached the pipeline carrying it; "
                    "investigate the approval path. source=%s",
                    key, record.get("source"))
                continue
            if key not in valid_keys:
                # Orphan key: approved upstream but not a real ScannerConfig
                # field — a silent dead write historically. Skip + surface.
                skipped.append(str(key))
                logger.warning(
                    "config_overlay: ORPHAN key %r skipped (not a ScannerConfig "
                    "field) — source=%s", key, record.get("source"))
                continue
            overlay["values"][key] = record.get("new_value")
            overlay["provenance"][key] = {
                "source": record.get("source"),
                "reason": record.get("reason"),
                "consumed_at": record.get("timestamp"),
            }
            landed.append(str(key))

        overlay["cycle"] = cycle
        if landed or applied:
            _write_overlay(root, overlay)
            adjuster.save_state()  # durably mark ids consumed
        if landed:
            logger.info("config_overlay: consumed %d adjustment(s): %s",
                        len(landed), ", ".join(landed))
        return {"consumed": landed, "orphans_skipped": skipped,
                "overlay_size": len(overlay["values"]), "error": None}
    except Exception as exc:  # noqa: BLE001 — supervisor tick must survive
        logger.warning("config_overlay: consume failed (non-fatal): %s", exc)
        return {"consumed": [], "orphans_skipped": [], "overlay_size": None,
                "error": str(exc)}


def apply_overlay(config: Any, root: Path) -> List[Dict[str, Any]]:
    """Replay the durable overlay onto a (freshly-constructed) ScannerConfig.

    The engine-seam function: one call at engine/TUI startup makes every past
    approved adjustment survive restarts. Validates every key against the
    ScannerConfig schema (orphan keys skipped loudly); returns the list of
    mutations actually made. Never raises.
    """
    applied: List[Dict[str, Any]] = []
    try:
        overlay = _read_overlay(root)
        valid_keys = _scanner_config_field_names()
        for key, value in overlay.get("values", {}).items():
            if key in PROTECTED_KEYS:
                logger.error("config_overlay: PROTECTED key %r in overlay "
                             "REFUSED at apply (Hard NO, L-003) — a tampered or "
                             "legacy overlay carried it; investigate.", key)
                continue
            if key not in valid_keys:
                logger.warning("config_overlay: ORPHAN key %r in overlay — "
                               "skipped at apply", key)
                continue
            old = getattr(config, key, None)
            if old == value:
                continue
            setattr(config, key, value)
            applied.append({"key": key, "old_value": old, "new_value": value,
                            "provenance": overlay.get("provenance", {}).get(key)})
        if applied:
            logger.info("config_overlay: applied %d overlay value(s): %s",
                        len(applied), ", ".join(a["key"] for a in applied))
    except Exception as exc:  # noqa: BLE001
        logger.warning("config_overlay: apply failed (non-fatal): %s", exc)
    return applied


def overlay_status(root: Path) -> Optional[Dict[str, Any]]:
    """Read-only summary for the maintenance report. None when no overlay yet."""
    path = root / OVERLAY_REL
    if not path.exists():
        return None
    overlay = _read_overlay(root)
    return {"n_values": len(overlay.get("values", {})),
            "keys": sorted(overlay.get("values", {}).keys())[:10],
            "cycle": overlay.get("cycle"),
            "last_updated": overlay.get("last_updated")}
