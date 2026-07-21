"""Read-only equity-research evidence cockpit view.

Deliberately dependency-free (stdlib ``json`` + ``pathlib`` only): the dashboard
must render the evidence state without importing the research/scoring stack or
any signing keys. It reads the store's own committed ``indexes/current.json``
projection — which the local authority already validated before writing — plus
the signed verdict payloads for display. It is strictly read-only and never
re-derives authority.

Beyond the flat per-lane list, it surfaces a ``tier_divergence`` grouping that
places each hypothesis's curated / wide / full_history verdicts side by side —
the roadmap J2 requirement to report (and visibly contrast) the attractive
curated-universe result against the defensible wide-universe result.
"""

from __future__ import annotations

import json
from pathlib import Path

# Known universe-tier suffixes, longest first so "full_history" is matched before
# any shorter suffix that could be a prefix of it.
_TIER_SUFFIXES = ("full_history", "curated", "wide")
_LANE_PREFIX = "equity_research_"


def _split_lane(lane_id: str) -> tuple[str, str] | None:
    """Parse ``equity_research_{hypothesis}_{tier}`` -> (hypothesis, tier)."""
    if not lane_id or not lane_id.startswith(_LANE_PREFIX):
        return None
    body = lane_id[len(_LANE_PREFIX):]
    for tier in _TIER_SUFFIXES:
        suffix = f"_{tier}"
        if body.endswith(suffix) and len(body) > len(suffix):
            return body[: -len(suffix)], tier
    return None


def equity_research_evidence_view(store_root: str | Path) -> dict:
    """Build a per-tier display view + a per-hypothesis tier-divergence grouping."""
    root = Path(store_root)
    index_path = root / "indexes" / "current.json"
    if not index_path.exists():
        return {"available": False, "reason": "no evidence index on disk",
                "lanes": [], "tier_divergence": {}, "champions": {}}

    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"unreadable evidence index: {exc}",
                "lanes": [], "tier_divergence": {}, "champions": {}}

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
        lanes.append({
            "lane_id": lane_id,
            "hypothesis_id": parsed[0] if parsed else None,
            "tier": parsed[1] if parsed else None,
            "package_id": info.get("package_id"),
            "package_digest": package_digest,
            "state": info.get("state"),
            "disposition_head_digest": info.get("head_event_digest"),
            "verdict": verdicts_by_package.get(package_digest),
        })
    lanes.sort(key=lambda entry: (entry["lane_id"] or "", entry["package_digest"]))

    # Group by hypothesis so the curated vs wide vs full_history verdicts are
    # visibly reported side by side (roadmap J2).
    tier_divergence: dict[str, dict[str, dict]] = {}
    for entry in lanes:
        hypothesis = entry["hypothesis_id"]
        tier = entry["tier"]
        if not hypothesis or not tier:
            continue
        tier_divergence.setdefault(hypothesis, {})[tier] = {
            "state": entry["state"],
            "lane_id": entry["lane_id"],
            "package_digest": entry["package_digest"],
        }

    return {
        "available": True,
        "source": str(index_path),
        "lanes": lanes,
        "tier_divergence": tier_divergence,
        "champions": index.get("champions", {}),
    }


__all__ = ["equity_research_evidence_view"]
