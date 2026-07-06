"""Promotion proposals — the brain loop's ONLY output toward going live, and it
is exactly that: a proposal file. This module contains no function that sets
``gate_pass`` on a SHIP_GATE.json, no function that arms anything, and no
function that starts a live loop. Shadow promotion (no capital at risk, no ARM
required — identical in kind to the research runs this repo already does) may
auto-approve. Live promotion always lands as ``PENDING_OPERATOR`` and stops
there; the operator's existing ARM + ``start_loop`` dashboard flow
(``dashboard/server/control_safety.py`` / ``control.py``) is the only path
from PENDING_OPERATOR to anything live.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

_VALID_TARGETS = ("shadow", "live")


class PromotionRefused(RuntimeError):
    """Raised when a promotion is proposed for a hypothesis whose gate did not
    resolve to CONTINUE."""


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def propose_promotion(
    requests_dir: Path,
    *,
    hypothesis_id: str,
    decision: Mapping[str, Any],
    target: str,
) -> Path:
    """Write a promotion proposal. Refuses if the gate did not CONTINUE.

    ``target="shadow"`` auto-approves (no capital, no ARM needed — matches how
    this repo already runs shadow/paper research). ``target="live"`` always
    lands PENDING_OPERATOR; nothing in this function can change that.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(f"unknown promotion target {target!r}; must be one of {_VALID_TARGETS}")
    if decision.get("decision") != "continue":
        raise PromotionRefused(
            f"cannot propose {target} promotion for {hypothesis_id!r}: "
            f"gate decision was {decision.get('decision')!r}, not 'continue'"
        )

    status = "APPROVED_SHADOW" if target == "shadow" else "PENDING_OPERATOR"
    payload = {
        "hypothesis_id": str(hypothesis_id),
        "target": target,
        "status": status,
        "requires_operator_arm": target == "live",
        "decision": dict(decision),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(requests_dir) / f"{hypothesis_id}-{target}.json"
    _atomic_write_json(path, payload)
    return path


def list_requests(requests_dir: Path) -> List[Dict[str, Any]]:
    """All promotion request payloads currently on disk. Missing dir -> []."""
    directory = Path(requests_dir)
    if not directory.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def list_pending(requests_dir: Path) -> List[Dict[str, Any]]:
    """Only requests awaiting operator action (i.e. live promotions)."""
    return [r for r in list_requests(requests_dir) if r.get("status") == "PENDING_OPERATOR"]
