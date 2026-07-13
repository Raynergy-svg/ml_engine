"""Read-only risk-target evidence cockpit view.

Deliberately dependency-free (stdlib ``json`` + ``pathlib`` only): the dashboard
must be able to render the evidence state without importing the training stack
(LightGBM, sklearn) or any signing keys. It reads the store's own committed
``indexes/current.json`` projection — which the local authority already
validated before writing — plus the signed verdict payloads for display. It is
strictly read-only and never re-derives authority.
"""

from __future__ import annotations

import json
from pathlib import Path


def risk_target_evidence_view(store_root: str | Path) -> dict:
    """Build a per-lane display view from the store's committed projection."""
    root = Path(store_root)
    index_path = root / "indexes" / "current.json"
    if not index_path.exists():
        return {"available": False, "reason": "no evidence index on disk", "lanes": [], "champions": {}}

    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"unreadable evidence index: {exc}", "lanes": [], "champions": {}}

    verdicts_by_package: dict[str, dict] = {}
    verdicts_dir = root / "verdicts"
    if verdicts_dir.is_dir():
        for path in sorted(verdicts_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text()).get("payload", {})
            except (OSError, json.JSONDecodeError):
                continue
            package_digest = payload.get("package_digest")
            if not package_digest:
                continue
            failed = [c.get("check_id") for c in payload.get("checks", []) if not c.get("passed", True)]
            verdicts_by_package[package_digest] = {
                "decision": payload.get("decision"),
                "rejection_reason": payload.get("rejection_reason"),
                "failed_checks": failed,
            }

    lanes = []
    for package_digest, info in sorted(index.get("packages", {}).items()):
        lanes.append({
            "lane_id": info.get("lane_id"),
            "package_id": info.get("package_id"),
            "package_digest": package_digest,
            "state": info.get("state"),
            "disposition_head_digest": info.get("head_event_digest"),
            "verdict": verdicts_by_package.get(package_digest),
        })
    lanes.sort(key=lambda entry: (entry["lane_id"] or "", entry["package_digest"]))

    return {
        "available": True,
        "source": str(index_path),
        "lanes": lanes,
        "champions": index.get("champions", {}),
    }


__all__ = ["risk_target_evidence_view"]
