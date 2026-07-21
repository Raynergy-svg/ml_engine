"""Dependency-free, read-only Track B scoring/evaluation evidence view."""

from __future__ import annotations

import json
from pathlib import Path


def track_b_evidence_view(store_root: str | Path) -> dict:
    root = Path(store_root)
    index_path = root / "indexes/current.json"
    if not index_path.exists():
        return {"available": False, "lanes": [], "stages": {}, "champions": {}}
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": str(exc), "lanes": [], "stages": {}, "champions": {}}
    lanes: list[dict] = []
    stages: dict[str, list[dict]] = {"scoring": [], "evaluation": []}
    for package_digest, info in sorted(index.get("packages", {}).items()):
        lane = info.get("lane_id") or ""
        if lane.startswith("track_b_scoring_"):
            stage = "scoring"
        elif lane.startswith("track_b_evaluation_"):
            stage = "evaluation"
        else:
            continue
        entry = {
            "lane_id": lane, "stage": stage, "package_digest": package_digest,
            "package_id": info.get("package_id"), "state": info.get("state"),
            "disposition_head_digest": info.get("head_event_digest"),
        }
        lanes.append(entry)
        stages[stage].append(entry)
    return {
        "available": True, "source": str(index_path), "lanes": lanes,
        "stages": stages, "champions": index.get("champions", {}),
    }


__all__ = ["track_b_evidence_view"]
