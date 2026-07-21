"""Read-only crypto-carry evidence cockpit view.

Deliberately dependency-free (stdlib ``json`` + ``pathlib`` only): the dashboard
must render the evidence state without importing the research/scoring stack or
any signing keys. It reads the store's own committed ``indexes/current.json``
projection — which the local authority already validated before writing — plus
the signed verdict payloads for display. It is strictly read-only and never
re-derives authority.

Beyond the flat per-lane list, it groups every campaign's independently
dispositioned carry_id results. Campaign and carry_id are separated by
an explicit ``__`` delimiter so identifiers containing underscores remain
unambiguous.
"""

from __future__ import annotations

import json
from pathlib import Path

_LANE_PREFIX = "crypto_carry_"


def _split_lane(lane_id: str) -> tuple[str, str] | None:
    """Parse ``crypto_carry_{campaign}__{carry_id}``."""
    if not lane_id or not lane_id.startswith(_LANE_PREFIX):
        return None
    body = lane_id[len(_LANE_PREFIX):]
    if "__" not in body:
        return None
    campaign, carry_id = body.split("__", 1)
    return (campaign, carry_id) if campaign and carry_id else None


def crypto_carry_evidence_view(store_root: str | Path) -> dict:
    """Build a per-carry_id display view + a per-campaign carry_id-divergence grouping."""
    root = Path(store_root)
    index_path = root / "indexes" / "current.json"
    if not index_path.exists():
        return {"available": False, "reason": "no evidence index on disk",
                "lanes": [], "carry_results": {}, "champions": {}}

    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"unreadable evidence index: {exc}",
                "lanes": [], "carry_results": {}, "champions": {}}

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
        lane_id = info.get("lane_id")
        parsed = _split_lane(lane_id or "")
        if parsed is None:
            continue
        lanes.append({
            "lane_id": lane_id,
            "campaign_id": parsed[0],
            "carry_id": parsed[1],
            "package_id": info.get("package_id"),
            "package_digest": package_digest,
            "state": info.get("state"),
            "disposition_head_digest": info.get("head_event_digest"),
            "verdict": verdicts_by_package.get(package_digest),
        })
    lanes.sort(key=lambda entry: (entry["lane_id"] or "", entry["package_digest"]))

    # Group by campaign so all frozen carries remain independently visible.
    carry_results: dict[str, dict[str, dict]] = {}
    for entry in lanes:
        campaign = entry["campaign_id"]
        carry_id = entry["carry_id"]
        if not campaign or not carry_id:
            continue
        carry_results.setdefault(campaign, {})[carry_id] = {
            "state": entry["state"],
            "lane_id": entry["lane_id"],
            "package_digest": entry["package_digest"],
        }

    return {
        "available": True,
        "source": str(index_path),
        "lanes": lanes,
        "carry_results": carry_results,
        "champions": index.get("champions", {}),
    }


__all__ = ["crypto_carry_evidence_view"]
