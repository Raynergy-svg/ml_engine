"""Dependency-free read-only portfolio-book evidence view for the cockpit."""

from __future__ import annotations

import json
from pathlib import Path

from .models import LANE_ID


def portfolio_book_evidence_view(store_root: str | Path) -> dict:
    root = Path(store_root)
    index_path = root / "indexes" / "current.json"
    if not index_path.exists():
        return {"available": False, "reason": "no evidence index on disk", "books": [], "champions": {}}
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"unreadable evidence index: {exc}", "books": [], "champions": {}}
    if not isinstance(index, dict) or not isinstance(index.get("packages", {}), dict):
        return {"available": False, "reason": "evidence index has an invalid structure", "books": [], "champions": {}}
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
            content_bound_check = next(
                (check for check in checks if check.get("check_id") == "source_content_bound"), None
            )
            verdicts[digest] = {
                "decision": payload.get("decision"),
                "rejection_reason": payload.get("rejection_reason"),
                "failed_checks": [check.get("check_id") for check in checks if not check.get("passed", True)],
                # Surfaced explicitly, not folded into failed_checks: PASS alone
                # does not distinguish "every sleeve's source lane published a
                # verified return contract" from "no sleeve did, existence and
                # quarantine state were the only things checked" — the coverage
                # claim lives only in this check's free-text details, so read
                # it out here rather than let a human have to open the raw
                # verdict file to see what "content-bound" actually covered.
                "content_binding": (content_bound_check or {}).get("details"),
            }
    books = []
    for digest, info in sorted(index.get("packages", {}).items()):
        if not isinstance(info, dict) or info.get("lane_id") != LANE_ID:
            continue
        books.append({
            "lane_id": LANE_ID, "package_id": info.get("package_id"),
            "package_digest": digest, "state": info.get("state"),
            "disposition_head_digest": info.get("head_event_digest"),
            "verdict": verdicts.get(digest),
        })
    return {"available": True, "source": str(index_path), "books": books, "champions": index.get("champions", {})}


__all__ = ["portfolio_book_evidence_view"]
