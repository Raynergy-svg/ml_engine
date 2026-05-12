"""Tier 1 T5: TTL cache around ConfigAdjuster._load_state."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.scanner.automation.config_adjuster import ConfigAdjuster


def _seed(path: Path, history: list, pending: list) -> None:
    path.write_text(json.dumps({
        "history": history,
        "pending": pending,
        "last_applied": None,
    }))


def test_cache_hit_within_ttl_does_not_reread(tmp_path: Path):
    p = tmp_path / "config_adjustments.json"
    _seed(p, history=[{"key": "x", "value": 1}], pending=[])
    a = ConfigAdjuster(persistence_path=p, ttl_seconds=5.0)
    a._load_state()
    _seed(p, history=[{"key": "x", "value": 2}], pending=[])
    a._load_state()
    assert a._load_count == 1


def test_cache_expires_after_ttl(tmp_path: Path):
    p = tmp_path / "config_adjustments.json"
    _seed(p, history=[{"key": "x", "value": 1}], pending=[])
    a = ConfigAdjuster(persistence_path=p, ttl_seconds=0.1)
    a._load_state()
    time.sleep(0.15)
    a._load_state()
    assert a._load_count == 2


def test_invalidate_forces_reload(tmp_path: Path):
    p = tmp_path / "config_adjustments.json"
    _seed(p, history=[{"key": "x", "value": 1}], pending=[])
    a = ConfigAdjuster(persistence_path=p, ttl_seconds=60.0)
    a._load_state()
    a._invalidate_cache()
    a._load_state()
    assert a._load_count == 2
