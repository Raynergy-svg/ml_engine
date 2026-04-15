"""Model freshness — when was each model last trained?

Stale models are a major cause of losing streaks: market regime drifts,
the model's learned distribution stops matching reality, and every
prediction becomes worse than chance. The mechanical RL feedback loop
slowly degrades agent weights but doesn't know to retrain — only Claude
(or a human) can decide "these models need a full retrain."

This module reads every meta JSON / file mtime under trained_data/models/
and returns a structured "training calendar" Claude can reason over:

  {
    "modular_ensemble": {"trained_at": "2026-03-18T23:11Z", "age_days": 28},
    "joint_gates":      {"trained_at": "2026-04-10T11:00Z", "age_days": 5},
    "agent_weights":    {"updated_at": "2026-04-14T20:00Z", "age_days": 0},
    "oldest_age_days":  28,
    "stale_models":     ["modular_ensemble (28d)"],
    "status":           "STALE" | "AGING" | "FRESH",
  }

Status thresholds:
  - FRESH:  oldest model ≤ 7 days
  - AGING:  oldest 7–14 days  (warn but don't escalate)
  - STALE:  oldest > 14 days  (PostTradeDiagnostics flag, Claude trigger)
  - CRITICAL: oldest > 30 days (immediate self-heal Claude spawn)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Threshold defaults — tuned for forex weekly retraining cadence.
# Forex regime drift is real on a weekly timescale; default Mon+Fri
# retraining means models should never be more than ~3-4 days old in
# steady state. Anything beyond a week is a degraded state.
#
# Override via env: BUDDY_FRESHNESS_AGING_DAYS / STALE_DAYS / CRITICAL_DAYS
AGING_DAYS = int(os.environ.get("BUDDY_FRESHNESS_AGING_DAYS", "3"))
STALE_DAYS = int(os.environ.get("BUDDY_FRESHNESS_STALE_DAYS", "5"))
CRITICAL_DAYS = int(os.environ.get("BUDDY_FRESHNESS_CRITICAL_DAYS", "7"))

MODELS_DIR = Path("trained_data/models")


@dataclass
class ModelGroup:
    """Freshness for one model group (e.g. modular ensemble, joint gates)."""
    name: str
    path: Path
    trained_at: Optional[datetime] = None
    age_days: Optional[float] = None
    source: str = ""  # "meta_json" | "file_mtime"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path) if self.path else None,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "age_days": round(self.age_days, 1) if self.age_days is not None else None,
            "source": self.source,
            **self.extra,
        }


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _file_mtime(path: Path) -> Optional[datetime]:
    try:
        if not path.exists():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _read_meta_trained_at(meta_path: Path, key: str = "trained_at") -> Optional[datetime]:
    """Read trained_at from a meta JSON. Falls back to None on any error."""
    try:
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return _parse_iso(data.get(key))
    except (json.JSONDecodeError, OSError):
        return None


def _age_days(dt: datetime) -> float:
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def get_model_freshness(models_dir: Path = MODELS_DIR) -> Dict[str, Any]:
    """Compute training freshness for every model group buddy uses.

    Returns a structured dict suitable for inclusion in a Claude reflection
    prompt and for use by PostTradeDiagnostics. Never raises.
    """
    groups: List[ModelGroup] = []

    # ── Modular ensemble ───────────────────────────────────────────────
    mei_meta_path = models_dir / "modular_ensemble.meta.json"
    mei_dt = _read_meta_trained_at(mei_meta_path, "trained_at")
    source = "meta_json" if mei_dt else "file_mtime"
    if mei_dt is None:
        # Fallback: oldest .keras file mtime in the joint dir
        keras_files = list((models_dir / "joint").glob("*.keras")) if (models_dir / "joint").exists() else []
        if keras_files:
            mei_dt = min((_file_mtime(p) for p in keras_files), default=None)
    groups.append(ModelGroup(
        name="modular_ensemble",
        path=mei_meta_path,
        trained_at=mei_dt,
        age_days=_age_days(mei_dt) if mei_dt else None,
        source=source,
    ))

    # ── Joint gates ────────────────────────────────────────────────────
    joint_meta_path = models_dir / "joint" / "joint_training_meta.json"
    joint_dt = _read_meta_trained_at(joint_meta_path, "trained_at")
    if joint_dt is None and joint_meta_path.exists():
        joint_dt = _file_mtime(joint_meta_path)
        source = "file_mtime"
    else:
        source = "meta_json" if joint_dt else "missing"
    groups.append(ModelGroup(
        name="joint_gates",
        path=joint_meta_path,
        trained_at=joint_dt,
        age_days=_age_days(joint_dt) if joint_dt else None,
        source=source,
    ))

    # ── Agent weights (updated by RL, not by training) ─────────────────
    weights_path = models_dir / "agent_weights.json"
    weights_dt: Optional[datetime] = None
    if weights_path.exists():
        try:
            data = json.loads(weights_path.read_text())
            meta = data.get("_meta", {}) or {}
            weights_dt = _parse_iso(meta.get("last_updated") or meta.get("updated_at"))
        except (json.JSONDecodeError, OSError):
            weights_dt = None
        if weights_dt is None:
            weights_dt = _file_mtime(weights_path)
    groups.append(ModelGroup(
        name="agent_weights",
        path=weights_path,
        trained_at=weights_dt,
        age_days=_age_days(weights_dt) if weights_dt else None,
        source="meta_json" if weights_dt else "missing",
    ))

    # ── Per-pair models (just the oldest, they should be co-trained) ───
    pair_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name not in ("joint", "shadow")]
    if pair_dirs:
        # Find oldest meta among per-pair dirs
        per_pair_dt: Optional[datetime] = None
        oldest_pair: Optional[str] = None
        for pdir in pair_dirs:
            for meta_file in pdir.glob("*.meta.json"):
                dt = _read_meta_trained_at(meta_file, "trained_at")
                if dt is None:
                    dt = _file_mtime(meta_file)
                if dt and (per_pair_dt is None or dt < per_pair_dt):
                    per_pair_dt = dt
                    oldest_pair = pdir.name
        groups.append(ModelGroup(
            name="per_pair_models",
            path=models_dir,
            trained_at=per_pair_dt,
            age_days=_age_days(per_pair_dt) if per_pair_dt else None,
            source="meta_json" if per_pair_dt else "missing",
            extra={"oldest_pair": oldest_pair} if oldest_pair else {},
        ))

    # ── Roll up ────────────────────────────────────────────────────────
    # Exclude agent_weights from "training freshness" — those update via
    # RL feedback, not retraining. Use them as a separate signal.
    train_groups = [g for g in groups if g.name != "agent_weights"]
    ages = [g.age_days for g in train_groups if g.age_days is not None]
    oldest = max(ages) if ages else None

    # Status classification
    if oldest is None:
        status = "UNKNOWN"
    elif oldest > CRITICAL_DAYS:
        status = "CRITICAL"
    elif oldest > STALE_DAYS:
        status = "STALE"
    elif oldest > AGING_DAYS:
        status = "AGING"
    else:
        status = "FRESH"

    stale_list = [
        f"{g.name} ({g.age_days:.0f}d)"
        for g in train_groups
        if g.age_days is not None and g.age_days > STALE_DAYS
    ]

    return {
        "groups": [g.to_dict() for g in groups],
        "oldest_age_days": round(oldest, 1) if oldest is not None else None,
        "stale_models": stale_list,
        "status": status,
        "thresholds": {
            "aging_days": AGING_DAYS,
            "stale_days": STALE_DAYS,
            "critical_days": CRITICAL_DAYS,
        },
    }


def format_freshness_for_prompt(freshness: Dict[str, Any]) -> str:
    """Format the freshness dict as a compact text block suitable for
    inclusion in a Claude reflection prompt."""
    lines = [
        f"MODEL_FRESHNESS:",
        f"  status: {freshness.get('status', 'UNKNOWN')}",
        f"  oldest_age_days: {freshness.get('oldest_age_days')}",
    ]
    stale = freshness.get("stale_models") or []
    if stale:
        lines.append(f"  stale: {', '.join(stale)}")
    lines.append("  groups:")
    for g in freshness.get("groups", []):
        age = g.get("age_days")
        age_str = f"{age:.0f}d" if age is not None else "?"
        lines.append(f"    - {g.get('name')}: trained {g.get('trained_at') or 'unknown'} (age={age_str})")
    return "\n".join(lines)


__all__ = [
    "get_model_freshness",
    "format_freshness_for_prompt",
    "AGING_DAYS",
    "STALE_DAYS",
    "CRITICAL_DAYS",
]
