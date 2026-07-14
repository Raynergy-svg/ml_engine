"""Dependency-free read-only execution-cost evidence view."""

from __future__ import annotations

import json
from pathlib import Path

LANE_ID = "execution_cost_model"


def execution_cost_evidence_view(store_root: str | Path) -> dict:
    root = Path(store_root)
    index_path = root / "indexes" / "current.json"
    if not index_path.exists():
        return {"available": False, "reason": "no evidence index on disk", "lanes": [], "champions": {}}
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"unreadable evidence index: {exc}", "lanes": [], "champions": {}}
    if not isinstance(index, dict) or not isinstance(index.get("packages", {}), dict):
        return {"available": False, "reason": "evidence index has an invalid structure", "lanes": [], "champions": {}}
    verdicts: dict[str, dict] = {}
    for path in sorted((root / "verdicts").glob("*.json")) if (root / "verdicts").is_dir() else ():
        try:
            envelope = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(envelope, dict):
            continue
        payload = envelope.get("payload", {})
        if not isinstance(payload, dict):
            continue
        checks = payload.get("checks", [])
        if not isinstance(checks, list) or not all(isinstance(check, dict) for check in checks):
            continue
        digest = payload.get("package_digest")
        if isinstance(digest, str) and digest:
            verdicts[digest] = {
                "decision": payload.get("decision"),
                "rejection_reason": payload.get("rejection_reason"),
                "failed_checks": [check.get("check_id") for check in checks if not check.get("passed", True)],
            }
    lanes = []
    for digest, info in sorted(index.get("packages", {}).items()):
        if not isinstance(info, dict):
            continue
        if info.get("lane_id") != LANE_ID:
            continue
        lanes.append({
            "lane_id": LANE_ID, "package_id": info.get("package_id"),
            "package_digest": digest, "state": info.get("state"),
            "disposition_head_digest": info.get("head_event_digest"),
            "verdict": verdicts.get(digest),
        })
    return {"available": True, "source": str(index_path), "lanes": lanes, "champions": index.get("champions", {})}


__all__ = ["execution_cost_evidence_view"]
