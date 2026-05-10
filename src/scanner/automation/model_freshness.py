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

import fnmatch
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


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

# Per-pair scan: dirs to ALWAYS exclude (not pair models at all).
_PER_PAIR_EXCLUDE_NAMES = frozenset({"joint", "shadow", "weight_snapshots"})

# Per-pair scan: glob patterns for workbench/experimental/scratch dirs that
# share a pair prefix but are NOT the live per-pair routing target. The live
# Tier 7 per-pair routing in src/scanner/gates.py looks up plain `{PAIR}/`
# (e.g. `EUR_USD/`); anything with these suffixes is a side experiment and
# would skew freshness reports if counted as the production model.
_PER_PAIR_EXCLUDE_PATTERNS = (
    "*_oos_train",
    "*_tuned",
    "*_news_*",
    "*_pre_tuned_*",
    "*_investigation",
)

# Anchor file used to age a per-pair dir. Matches the production inference
# entry point (gates.evaluate_transformer loads transformer_direction.keras).
_PER_PAIR_ANCHOR = "transformer_direction.keras"


def _is_per_pair_workbench(name: str) -> bool:
    """Return True if `name` is a workbench / scratch dir, not a live pair."""
    if name in _PER_PAIR_EXCLUDE_NAMES:
        return True
    return any(fnmatch.fnmatchcase(name, pat) for pat in _PER_PAIR_EXCLUDE_PATTERNS)


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


def _normalize_pair(pair: Any) -> str:
    return str(pair or "").upper().replace("/", "_").strip()


def _resolve_model_source_pair(
    pair: Any,
    models_dir: Path,
    *,
    use_master_pair_models: bool = True,
) -> str:
    """Mirror Scanner._resolve_model_source_pair for freshness reporting."""
    normalized = _normalize_pair(pair)
    if not normalized:
        return ""
    if not use_master_pair_models:
        return normalized
    marker = models_dir / normalized / _PER_PAIR_ANCHOR
    try:
        if marker.exists():
            return normalized
    except OSError:
        return normalized
    try:
        from src.training.correlation_group_config import get_correlation_group

        group = get_correlation_group(normalized)
        if group is None:
            return normalized
        master = _normalize_pair(group.master_pair)
        return master or normalized
    except Exception:
        return normalized


def get_model_freshness(
    models_dir: Path = MODELS_DIR,
    *,
    active_pairs: Sequence[str] | None = None,
    use_master_pair_models: bool = True,
) -> Dict[str, Any]:
    """Compute training freshness for every model group buddy uses.

    Returns a structured dict suitable for inclusion in a Claude reflection
    prompt and for use by PostTradeDiagnostics. Never raises.
    """
    return get_model_freshness_for_pairs(
        models_dir=models_dir,
        active_pairs=active_pairs,
        use_master_pair_models=use_master_pair_models,
    )


def get_model_freshness_for_pairs(
    models_dir: Path = MODELS_DIR,
    *,
    active_pairs: Sequence[str] | None = None,
    use_master_pair_models: bool = True,
) -> Dict[str, Any]:
    """Compute model freshness, optionally scoped to active inference sources.

    When ``active_pairs`` is provided, the per-pair freshness group mirrors the
    scanner's runtime model routing: dedicated pair model first, then
    correlation-group master fallback. Global callers still get the full-disk
    audit by omitting ``active_pairs``.
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

    # ── RL position sizer ──────────────────────────────────────────────
    # Mythos audit 2026-04-28 — pre-fix this group was absent from the
    # header manifest, so the trigger header reported FRESH/oldest=0.0
    # while the diagnostic flagged CRITICAL on rl_position_sizer (see
    # learnings.md self_heal_rl_staleness_C6_cadence_collapse). We now
    # include it AND pick the OLDER of {.zip mtime, .meta.json trained_at}
    # so neither write path can hide staleness — the C5 finding showed
    # these timestamps can diverge by ~14 days.
    rl_zip_path = models_dir / "rl_position_sizer.zip"
    if rl_zip_path.exists():
        rl_meta_path = models_dir / "rl_position_sizer.meta.json"
        candidates: List[datetime] = []
        meta_dt = _read_meta_trained_at(rl_meta_path, "trained_at")
        if meta_dt is not None:
            candidates.append(meta_dt)
        zip_dt = _file_mtime(rl_zip_path)
        if zip_dt is not None:
            candidates.append(zip_dt)
        # Older timestamp wins (more conservative — never under-report stale).
        rl_dt = min(candidates, default=None)
        rl_source = (
            "meta_json+zip_mtime" if len(candidates) == 2
            else ("meta_json" if meta_dt else ("zip_mtime" if zip_dt else "missing"))
        )
        groups.append(ModelGroup(
            name="rl_position_sizer",
            path=rl_zip_path,
            trained_at=rl_dt,
            age_days=_age_days(rl_dt) if rl_dt else None,
            source=rl_source,
        ))

    # ── Per-pair models ────────────────────────────────────────────────
    # Tier 7 per-pair routing (src/scanner/gates.py:_get_pair_evaluator)
    # picks `trained_data/models/{PAIR}/` over the joint dir whenever the
    # pair subdir exists. When per-pair routing is active, those are the
    # ACTIVE models — joint becomes fallback only.
    #
    # We age each pair by the mtime of its `transformer_direction.keras`
    # because that is the file gates.evaluate_transformer actually loads.
    # Anchoring on this single file avoids two failure modes:
    #   1. dirs that have stray *.meta.json files but no real model (would
    #      otherwise look "fresh" because the meta was touched recently)
    #   2. workbench dirs (`*_tuned`, `*_oos_train`, etc.) that share a
    #      pair prefix but aren't on the per-pair routing path
    #
    # Reported `age_days` on the group is the OLDEST among scanned pairs
    # (most conservative — never under-report staleness, same principle as
    # the rl_position_sizer fix on 2026-04-28).
    active_pair_list = [_normalize_pair(p) for p in (active_pairs or [])]
    active_pair_list = [p for p in active_pair_list if p]
    active_source_pairs: List[str] = []
    if active_pair_list:
        seen_sources: set[str] = set()
        for pair in active_pair_list:
            source_pair = _resolve_model_source_pair(
                pair,
                models_dir,
                use_master_pair_models=use_master_pair_models,
            )
            if source_pair and source_pair not in seen_sources:
                active_source_pairs.append(source_pair)
                seen_sources.add(source_pair)

    try:
        if active_source_pairs:
            pair_dirs = [
                models_dir / source_pair
                for source_pair in active_source_pairs
                if not _is_per_pair_workbench(source_pair)
            ]
        else:
            pair_dirs = [
                d for d in models_dir.iterdir()
                if d.is_dir() and not _is_per_pair_workbench(d.name)
            ]
    except (OSError, FileNotFoundError):
        pair_dirs = []

    pairs_with_transformer: List[Dict[str, Any]] = []
    oldest_dt: Optional[datetime] = None
    oldest_pair: Optional[str] = None
    excluded_workbench: List[str] = []
    excluded_no_transformer: List[str] = []

    if pair_dirs:
        for pdir in sorted(pair_dirs, key=lambda p: p.name):
            anchor = pdir / _PER_PAIR_ANCHOR
            if not anchor.exists():
                excluded_no_transformer.append(pdir.name)
                continue
            dt = _file_mtime(anchor)
            if dt is None:
                excluded_no_transformer.append(pdir.name)
                continue
            pairs_with_transformer.append({
                "pair": pdir.name,
                "trained_at": dt.isoformat(),
                "age_days": round(_age_days(dt), 1),
            })
            if oldest_dt is None or dt < oldest_dt:
                oldest_dt = dt
                oldest_pair = pdir.name

    # Track workbench exclusions (informational; helps audit-trail readers
    # see WHY a known-recent dir like `EUR_USD_tuned` isn't in the count).
    try:
        for d in models_dir.iterdir():
            if d.is_dir() and _is_per_pair_workbench(d.name):
                # Skip the catch-all utility dirs from the noisy excluded list.
                if d.name not in _PER_PAIR_EXCLUDE_NAMES:
                    excluded_workbench.append(d.name)
    except (OSError, FileNotFoundError):
        pass

    pair_names = [p["pair"] for p in pairs_with_transformer]
    extra: Dict[str, Any] = {
        "pairs": pair_names,
        "pair_count": len(pair_names),
        "per_pair_ages": pairs_with_transformer,
    }
    if active_pair_list:
        extra["scope"] = "active_pairs"
        extra["requested_pairs"] = active_pair_list
        extra["model_source_pairs"] = active_source_pairs
    else:
        extra["scope"] = "all_pairs"
    if oldest_pair:
        extra["oldest_pair"] = oldest_pair
    if excluded_workbench:
        extra["excluded_workbench"] = sorted(excluded_workbench)
    if excluded_no_transformer:
        extra["excluded_no_transformer"] = sorted(excluded_no_transformer)

    groups.append(ModelGroup(
        name="per_pair_models",
        path=models_dir,
        trained_at=oldest_dt,
        age_days=_age_days(oldest_dt) if oldest_dt else None,
        source="file_mtime" if oldest_dt else "missing",
        extra=extra,
    ))

    # ── Roll up ────────────────────────────────────────────────────────
    # Exclude agent_weights from "training freshness" — those update via
    # RL feedback, not retraining. Use them as a separate signal.
    train_groups = [g for g in groups if g.name != "agent_weights"]
    active_pair_sources_have_models = (
        bool(active_source_pairs)
        and set(pair_names) == set(active_source_pairs)
    )
    if active_pair_sources_have_models:
        # Active scanner calls use per-pair gate evaluators for these sources;
        # joint_gates is fallback/audit context, not a live freshness blocker.
        train_groups = [g for g in train_groups if g.name != "joint_gates"]
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
        "rollup_scope": (
            "active_pair_models" if active_pair_sources_have_models
            else ("active_pairs" if active_pair_list else "all_models")
        ),
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
    "get_model_freshness_for_pairs",
    "format_freshness_for_prompt",
    "AGING_DAYS",
    "STALE_DAYS",
    "CRITICAL_DAYS",
]
