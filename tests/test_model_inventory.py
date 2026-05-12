"""Tier 2 T10: Per-pair model inventory."""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.tui.widgets.model_inventory import ModelInventory, ModelCard


def _write_meta(path: Path, granularity: str = "M15", holdout: float = 0.65) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "granularity": granularity,
        "val_balanced_accuracy": holdout,
        "trained_at": "2026-05-01T00:00:00+00:00",
        "feature_pipeline_version": "2026-05-08-v1",
    }
    joblib.dump(payload, path)


def test_inventory_lists_pairs_with_meta(tmp_path: Path):
    root = tmp_path / "models"
    _write_meta(root / "EUR_USD" / "transformer_direction.meta.pkl")
    _write_meta(root / "GBP_USD" / "transformer_direction.meta.pkl", holdout=0.58)
    inv = ModelInventory(models_root=root)
    inv.scan()
    cards = inv.cards()
    pairs = {c.pair for c in cards}
    assert pairs == {"EUR_USD", "GBP_USD"}
    eur = next(c for c in cards if c.pair == "EUR_USD")
    assert eur.status == "ok"
    assert eur.holdout_accuracy == 0.65
    assert eur.granularity == "M15"
    assert eur.pipeline_version == "2026-05-08-v1"
    assert eur.age_days is not None and eur.age_days >= 0
    gbp = next(c for c in cards if c.pair == "GBP_USD")
    assert gbp.holdout_accuracy == 0.58


def test_missing_meta_returns_unknown_status(tmp_path: Path):
    root = tmp_path / "models"
    (root / "EUR_USD").mkdir(parents=True)
    inv = ModelInventory(models_root=root)
    inv.scan()
    cards = inv.cards()
    assert len(cards) == 1
    assert cards[0].pair == "EUR_USD"
    assert cards[0].status == "no_meta"
    assert cards[0].holdout_accuracy is None


def test_corrupt_meta_handled_gracefully(tmp_path: Path):
    root = tmp_path / "models"
    (root / "EUR_USD").mkdir(parents=True)
    (root / "EUR_USD" / "transformer_direction.meta.pkl").write_bytes(b"not a serialized object")
    inv = ModelInventory(models_root=root)
    inv.scan()
    cards = inv.cards()
    assert len(cards) == 1
    assert cards[0].pair == "EUR_USD"
    assert cards[0].status in ("corrupt_meta", "no_meta", "error")
