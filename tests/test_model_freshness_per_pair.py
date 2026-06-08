"""Per-pair model freshness — task brief 2026-05-09.

Pre-fix bug: `get_model_freshness()` reported per_pair_models as
`age_days=null, source="missing"` even though `trained_data/models/EUR_USD/`
and `trained_data/models/USD_JPY/` had transformers freshly trained today
(Phase 5.D tuning). The TUI brain feed reads the OLDEST group's age, so
operators saw "oldest 3.4d" (joint_gates) and assumed joint was live —
when in fact Tier 7 per-pair routing (gates._get_pair_evaluator) was using
the today-trained per-pair transformers as the active models.

This module verifies the post-fix behavior:
  (a) per-pair dirs containing transformer_direction.keras are detected
      and reported (with their per-pair ages) under the per_pair_models
      group
  (b) workbench dirs (`*_tuned`, `*_oos_train`, `*_news_*`,
      `*_pre_tuned_*`, `*_investigation`) and utility dirs (`joint`,
      `shadow`, `weight_snapshots`) are excluded
  (c) per-pair dirs with no transformer_direction.keras are skipped (they
      are partial/stub dirs, not scoreable)
  (d) the top-level `oldest_age_days` aggregate is driven by live training
      groups, including per_pair_models. Audit-only joint/modular metadata
      remains visible in `groups` but does not halt live per-pair routing.

NO MOCKS — real disk under tmp_path, real freshness computation, real
file mtimes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from src.scanner.automation import model_freshness as mf


def _touch(path: Path, when: float) -> None:
    """Set both atime and mtime of path to `when` (epoch seconds)."""
    os.utime(path, (when, when))


def _make_pair_dir(models: Path, pair: str, transformer_age_days: float) -> Path:
    """Create a per-pair dir with a transformer_direction.keras dated
    `transformer_age_days` ago."""
    pdir = models / pair
    pdir.mkdir(parents=True, exist_ok=True)
    keras = pdir / "transformer_direction.keras"
    keras.write_bytes(b"fake-keras-payload")
    _touch(keras, time.time() - transformer_age_days * 86400.0)
    return pdir


@pytest.fixture
def base_models(tmp_path):
    """Build a baseline models dir with the standard non-per-pair groups so
    the roll-up exercises exist. Per-pair dirs are added by each test."""
    models = tmp_path / "models"
    (models / "joint").mkdir(parents=True)

    # Modular ensemble — 2 days old
    modular_meta = models / "modular_ensemble.meta.json"
    modular_meta.write_text(json.dumps({
        "trained_at": (
            # 2 days back
            time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00",
                time.gmtime(time.time() - 2 * 86400),
            )
        )
    }))

    # Joint gates — 3 days old (the "oldest 3.4d" the brain feed reported)
    joint_meta = models / "joint" / "joint_training_meta.json"
    joint_meta.write_text(json.dumps({
        "trained_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00",
            time.gmtime(time.time() - 3 * 86400),
        )
    }))

    # Agent weights — fresh
    (models / "agent_weights.json").write_text(json.dumps({
        "_meta": {"last_updated": time.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time())
        )},
    }))

    return models


# ─────────────────────────── (a) DETECTION ───────────────────────────────

def test_per_pair_dirs_detected_and_listed(base_models):
    """EUR_USD + USD_JPY today + GBP_USD 1d ago should all appear in the
    per_pair_models group with real ages."""
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)
    _make_pair_dir(base_models, "USD_JPY", transformer_age_days=0.0)
    _make_pair_dir(base_models, "GBP_USD", transformer_age_days=1.0)

    res = mf.get_model_freshness(base_models)
    pp = next(g for g in res["groups"] if g["name"] == "per_pair_models")

    assert pp["source"] == "file_mtime", (
        "source must reflect we used the keras mtime, not 'missing'"
    )
    assert pp["age_days"] is not None, "per-pair age must be populated"
    assert set(pp["pairs"]) == {"EUR_USD", "USD_JPY", "GBP_USD"}
    assert pp["pair_count"] == 3
    # GBP_USD is the oldest (1d) — should be reported on the group itself
    assert pp["oldest_pair"] == "GBP_USD"
    assert pp["age_days"] == pytest.approx(1.0, abs=0.1)

    # per_pair_ages must include each pair with a per-pair age
    pair_ages = {p["pair"]: p["age_days"] for p in pp["per_pair_ages"]}
    assert pair_ages["EUR_USD"] == pytest.approx(0.0, abs=0.1)
    assert pair_ages["USD_JPY"] == pytest.approx(0.0, abs=0.1)
    assert pair_ages["GBP_USD"] == pytest.approx(1.0, abs=0.1)


# ────────────────────────── (b) WORKBENCH EXCLUSION ───────────────────────

def test_workbench_dirs_excluded(base_models):
    """Workbench dirs sharing a pair prefix must NOT count as live pair
    models — they're scratch outputs from tuning/news experiments."""
    # Live per-pair: 0 days
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)
    # Workbench dirs — all should be SKIPPED, even though they have keras
    _make_pair_dir(base_models, "EUR_USD_tuned", transformer_age_days=0.0)
    _make_pair_dir(base_models, "EUR_USD_oos_train", transformer_age_days=0.5)
    _make_pair_dir(base_models, "EUR_USD_news_first_run", transformer_age_days=0.5)
    _make_pair_dir(base_models, "EUR_USD_news_k32", transformer_age_days=0.5)
    _make_pair_dir(base_models, "EUR_USD_pre_tuned_phase5d", transformer_age_days=0.5)
    _make_pair_dir(base_models, "USD_CAD_investigation", transformer_age_days=0.5)
    # Utility dirs that aren't pair models at all
    (base_models / "weight_snapshots").mkdir()
    (base_models / "shadow").mkdir()
    # joint/ already exists from the fixture

    res = mf.get_model_freshness(base_models)
    pp = next(g for g in res["groups"] if g["name"] == "per_pair_models")

    assert pp["pairs"] == ["EUR_USD"], (
        f"only the live pair should count, got {pp['pairs']}"
    )
    assert pp["pair_count"] == 1
    excluded = set(pp.get("excluded_workbench", []))
    assert "EUR_USD_tuned" in excluded
    assert "EUR_USD_oos_train" in excluded
    assert "EUR_USD_news_first_run" in excluded
    assert "EUR_USD_news_k32" in excluded
    assert "EUR_USD_pre_tuned_phase5d" in excluded
    assert "USD_CAD_investigation" in excluded
    # joint / shadow / weight_snapshots are catch-alls — they should NOT be
    # reported in the noisy excluded_workbench list
    assert "joint" not in excluded
    assert "shadow" not in excluded
    assert "weight_snapshots" not in excluded


# ─────────────────────── (c) NO-TRANSFORMER SKIP ──────────────────────────

def test_pair_dirs_without_transformer_skipped(base_models):
    """A per-pair dir with stray meta files but no transformer_direction.keras
    is a partial/stub state and must NOT count as a live model. This guards
    against the original placeholder bug where any *.meta.json in any subdir
    looked 'fresh' even when the actual model wasn't trained yet."""
    # Pair WITH transformer
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)
    # Pair WITHOUT transformer — only a stub meta file
    stub = base_models / "EUR_AUD"
    stub.mkdir()
    (stub / "lgbm_momentum.pkl").write_bytes(b"stub")  # no keras
    (stub / "joint_training_meta.json").write_text(json.dumps({
        "trained_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time())
        )
    }))

    res = mf.get_model_freshness(base_models)
    pp = next(g for g in res["groups"] if g["name"] == "per_pair_models")

    assert pp["pairs"] == ["EUR_USD"]
    assert "EUR_AUD" in pp.get("excluded_no_transformer", [])
    # And EUR_AUD must not have leaked into the per_pair_ages list
    pair_names_in_ages = [p["pair"] for p in pp["per_pair_ages"]]
    assert "EUR_AUD" not in pair_names_in_ages


def test_no_per_pair_dirs_at_all_returns_missing_source(base_models):
    """When no live per-pair dirs exist, group must remain in the manifest
    but report `source=missing` and `age_days=None` (don't crash, don't
    pretend it's fresh)."""
    # Don't add any per-pair dirs. Only joint/agent_weights/modular exist.

    res = mf.get_model_freshness(base_models)
    pp = next(g for g in res["groups"] if g["name"] == "per_pair_models")

    assert pp["age_days"] is None
    assert pp["source"] == "missing"
    assert pp["pair_count"] == 0
    assert pp["pairs"] == []


# ───────────────────── (d) AGGREGATE ROLL-UP CORRECTNESS ──────────────────

def test_oldest_age_days_aggregates_per_pair_when_per_pair_is_oldest(base_models):
    """The TOP-LEVEL `oldest_age_days` field must take the max across ALL
    training groups including per_pair_models. Otherwise a stale per-pair
    model (the live inference target) hides behind a fresh joint dir."""
    # joint_gates is 3d (from base_models fixture).
    # Add a per-pair dir that is OLDER than joint — this is the scenario
    # the brain feed must now correctly surface.
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)  # fresh
    _make_pair_dir(base_models, "EUR_GBP", transformer_age_days=12.0)  # very stale
    _make_pair_dir(base_models, "USD_JPY", transformer_age_days=0.0)

    res = mf.get_model_freshness(base_models)

    # The per_pair group itself must report 12d (oldest of its members).
    pp = next(g for g in res["groups"] if g["name"] == "per_pair_models")
    assert pp["age_days"] == pytest.approx(12.0, abs=0.5)
    assert pp["oldest_pair"] == "EUR_GBP"

    # The TOP-LEVEL roll-up must report 12d, NOT 3d (joint's age).
    assert res["oldest_age_days"] == pytest.approx(12.0, abs=0.5), (
        "oldest_age_days roll-up must include per_pair_models"
    )
    # 12d > CRITICAL_DAYS (default 10) so status escalates, exposing
    # what the pre-fix manifest was hiding.
    assert res["status"] == "CRITICAL"


def test_oldest_age_days_ignores_audit_only_joint_when_per_pair_is_fresher(base_models):
    """Joint/modular freshness stays in groups but not the live roll-up."""
    # joint_gates is 3d, modular is 2d (from fixture).
    # Per-pair is fresh — live roll-up should follow per_pair, not joint.
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)
    _make_pair_dir(base_models, "USD_JPY", transformer_age_days=0.5)

    res = mf.get_model_freshness(base_models)

    assert res["oldest_age_days"] == pytest.approx(0.5, abs=0.5)
    assert res["status"] == "FRESH"

    groups = {g["name"]: g for g in res["groups"]}
    assert groups["joint_gates"]["age_days"] == pytest.approx(3.0, abs=0.5)
    assert groups["modular_ensemble"]["age_days"] == pytest.approx(2.0, abs=0.5)


def test_per_pair_group_appears_in_groups_list_always(base_models):
    """API stability: per_pair_models must appear in the groups list even
    when empty, so downstream consumers (PostTradeDiagnostics, the brain
    feed, the trigger header manifest) can iterate without KeyError or
    silently dropping a category."""
    res = mf.get_model_freshness(base_models)
    group_names = [g["name"] for g in res["groups"]]
    assert "per_pair_models" in group_names

    # Add live per-pair, verify it's still a single per_pair_models entry
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)
    res2 = mf.get_model_freshness(base_models)
    pp_entries = [g for g in res2["groups"] if g["name"] == "per_pair_models"]
    assert len(pp_entries) == 1, "per_pair_models must be reported exactly once"


def test_active_pairs_missing_transformer_warn_without_hard_block(base_models):
    """Missing active anchors should warn operators without converting a
    genuinely fresh model set into a CRITICAL halt blocker."""
    _make_pair_dir(base_models, "EUR_USD", transformer_age_days=0.0)
    (base_models / "USD_CAD").mkdir()

    res = mf.get_model_freshness(base_models, active_pairs=["EUR_USD", "USD_CAD"])
    pp = next(g for g in res["groups"] if g["name"] == "per_pair_models")

    assert res["status"] == "FRESH"
    assert res["missing_active_pair_models"] == ["USD_CAD"]
    assert "USD_CAD" in pp["excluded_no_transformer"]
    assert pp["missing_active_model_sources"] == ["USD_CAD"]
    assert res["stale_models"] == []
    assert res["warnings"] == ["per_pair_models_missing (USD_CAD)"]
