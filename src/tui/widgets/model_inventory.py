"""Tier 2 T10: Per-pair model inventory — read meta sidecars via joblib.

Loader rationale: meta sidecars are written by the project's own trainer
(transformer_trainer.py) using joblib.dump — the sklearn-ecosystem convention.
We read with joblib.load to keep the inventory module decoupled from the
trainer's internal write API. The files originate inside this repo; no
external/untrusted input reaches this loader.

Public API:
    ModelCard  — frozen dataclass, one per pair directory under models_root.
    ModelInventory(models_root: Path).scan() / .cards()

Failure modes (mapped to ModelCard.status):
    ok             — meta loaded and parsed cleanly.
    no_meta        — pair dir exists but transformer_direction.meta.pkl missing.
    corrupt_meta   — joblib.load raised (file unreadable / truncated / wrong format).
    error          — meta loaded but downstream field extraction raised
                     (e.g. trained_at not iso-parseable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import joblib

logger = logging.getLogger(__name__)


META_FILENAME = "transformer_direction.meta.pkl"


@dataclass(frozen=True)
class ModelCard:
    """One row in the model inventory panel."""

    pair: str
    status: str  # ok | no_meta | corrupt_meta | error
    granularity: Optional[str] = None
    holdout_accuracy: Optional[float] = None
    trained_at: Optional[str] = None
    age_days: Optional[float] = None
    pipeline_version: Optional[str] = None


class ModelInventory:
    """Walks ``trained_data/models/<PAIR>/transformer_direction.meta.pkl``.

    Each pair-directory becomes one ``ModelCard``. ``scan()`` is idempotent —
    callers can re-invoke on a timer (the Diagnostics screen does so every 30s).
    """

    def __init__(self, *, models_root: Path) -> None:
        self._root = Path(models_root)
        self._cards: List[ModelCard] = []

    def scan(self) -> None:
        """Walk the models root and rebuild the card list."""
        cards: List[ModelCard] = []
        if not self._root.exists():
            self._cards = cards
            return

        for pair_dir in sorted(self._root.iterdir()):
            if not pair_dir.is_dir():
                continue
            cards.append(self._scan_pair(pair_dir))

        self._cards = cards

    def cards(self) -> List[ModelCard]:
        """Return a copy of the most recent scan's cards."""
        return list(self._cards)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _scan_pair(self, pair_dir: Path) -> ModelCard:
        pair = pair_dir.name
        meta_path = pair_dir / META_FILENAME
        if not meta_path.exists():
            return ModelCard(pair=pair, status="no_meta")

        # Load — joblib raises a wide variety of errors on corrupt input
        # (UnpicklingError surfaced by the underlying deserializer, EOFError,
        # OSError, KeyError on internal lookups, ValueError on bad magic bytes,
        # etc). Coalesce common cases first, fall through to a logged catch-all
        # so a future regression surfaces in buddy_debug.log instead of crashing
        # the diagnostics refresh tick.
        try:
            payload = joblib.load(meta_path)
        except (OSError, EOFError, KeyError, ValueError) as e:
            logger.debug(
                "ModelInventory: corrupt meta for %s at %s: %s", pair, meta_path, e
            )
            return ModelCard(pair=pair, status="corrupt_meta")
        except Exception as e:  # noqa: BLE001 — joblib's failure surface is broad
            logger.warning(
                "ModelInventory: unexpected load error for %s at %s: %s",
                pair,
                meta_path,
                e,
            )
            return ModelCard(pair=pair, status="corrupt_meta")

        # Field extraction. payload should be a dict written by the trainer; if
        # anything is the wrong shape (missing key, non-iso ts, non-float
        # accuracy) we degrade to status="error" with what we have.
        try:
            return self._card_from_payload(pair, payload)
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(
                "ModelInventory: malformed meta for %s at %s: %s", pair, meta_path, e
            )
            return ModelCard(pair=pair, status="error")

    @staticmethod
    def _card_from_payload(pair: str, payload: Any) -> ModelCard:
        if not isinstance(payload, dict):
            return ModelCard(pair=pair, status="error")

        trained_at_str = payload.get("trained_at")
        age_days: Optional[float] = None
        if isinstance(trained_at_str, str) and trained_at_str:
            ts = datetime.fromisoformat(trained_at_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0

        holdout_raw = payload.get("val_balanced_accuracy")
        holdout: Optional[float] = (
            float(holdout_raw) if isinstance(holdout_raw, (int, float)) else None
        )

        granularity_raw = payload.get("granularity")
        granularity: Optional[str] = (
            str(granularity_raw) if isinstance(granularity_raw, str) else None
        )

        pv_raw = payload.get("feature_pipeline_version")
        pipeline_version: Optional[str] = (
            str(pv_raw) if isinstance(pv_raw, str) else None
        )

        return ModelCard(
            pair=pair,
            status="ok",
            granularity=granularity,
            holdout_accuracy=holdout,
            trained_at=trained_at_str if isinstance(trained_at_str, str) else None,
            age_days=age_days,
            pipeline_version=pipeline_version,
        )
