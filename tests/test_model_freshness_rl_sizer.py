"""Mythos audit 2026-04-28 — manifest disjunction fix.

Pre-fix bug: get_model_freshness() enumerated only {modular_ensemble,
joint_gates, agent_weights, per_pair_models}. The RL position sizer was
absent from the header manifest, so format_freshness_for_prompt() reported
'status: FRESH oldest_age_days: 0.0' on the SAME scan whose diagnostic
flagged CRITICAL on rl_position_sizer being 22 days stale (see learnings.md
self_heal_rl_staleness_C6_cadence_collapse).

Per the C5 finding, the zip mtime and the meta.json trained_at can diverge
by ~14 days (zip touched without meta update, or vice versa). The unified
manifest must use the MOST conservative (older) of the two timestamps so
neither write path can hide staleness.
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


@pytest.fixture
def tmp_models(tmp_path):
    """Build a minimal models dir with the standard 4 groups + RL sizer."""
    models = tmp_path / "models"
    (models / "joint").mkdir(parents=True)
    (models / "EUR_USD").mkdir()
    now = time.time()

    # modular_ensemble — fresh
    (models / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": "2026-04-28T20:00:00+00:00"})
    )
    # joint_gates — fresh
    (models / "joint" / "joint_training_meta.json").write_text(
        json.dumps({"trained_at": "2026-04-28T20:00:00+00:00"})
    )
    # agent_weights — fresh
    (models / "agent_weights.json").write_text(
        json.dumps({
            "_meta": {"last_updated": "2026-04-28T20:00:00+00:00"},
        })
    )
    # per_pair_models — fresh
    (models / "EUR_USD" / "lgbm_momentum.meta.json").write_text(
        json.dumps({"trained_at": "2026-04-28T20:00:00+00:00"})
    )
    return models


def test_rl_position_sizer_appears_in_unified_manifest(tmp_models):
    """The header manifest must include rl_position_sizer when the .zip
    file exists. Pre-fix this group was absent → header lied with
    oldest_age_days=0.0 while the diagnostic burned on rl staleness."""
    rl_zip = tmp_models / "rl_position_sizer.zip"
    rl_zip.write_bytes(b"")
    rl_meta = tmp_models / "rl_position_sizer.meta.json"
    rl_meta.write_text(json.dumps({"trained_at": "2026-04-28T20:00:00+00:00"}))
    _touch(rl_zip, time.time())

    result = mf.get_model_freshness(tmp_models)
    names = [g["name"] for g in result["groups"]]
    assert "rl_position_sizer" in names, (
        f"rl_position_sizer missing from manifest. groups={names}"
    )


def test_oldest_age_days_picks_up_stale_rl_sizer(tmp_models):
    """When other models are fresh but RL sizer is 22d old, oldest_age_days
    must reflect 22d (not 0d). This is the bug that triggered the C6
    treadmill — header reported FRESH while diagnostic burned."""
    rl_zip = tmp_models / "rl_position_sizer.zip"
    rl_zip.write_bytes(b"")
    twentytwo_days_ago = time.time() - (22 * 86400)
    _touch(rl_zip, twentytwo_days_ago)

    result = mf.get_model_freshness(tmp_models)
    assert result["oldest_age_days"] is not None
    assert result["oldest_age_days"] >= 21.5, (
        f"oldest_age_days={result['oldest_age_days']} did not reflect "
        f"22d-stale RL sizer. status={result['status']}"
    )
    # 22d > CRITICAL_DAYS default (7d) → status must escalate.
    assert result["status"] == "CRITICAL"


def test_rl_sizer_uses_most_conservative_timestamp(tmp_models):
    """C5 finding: zip mtime and meta.json trained_at can diverge ~14d.
    The unified manifest must use the OLDER (more stale) timestamp so
    no write path can hide staleness from the header."""
    rl_zip = tmp_models / "rl_position_sizer.zip"
    rl_meta = tmp_models / "rl_position_sizer.meta.json"
    rl_zip.write_bytes(b"")

    # zip is 5 days old; meta says it was trained 30 days ago.
    rl_meta.write_text(json.dumps({"trained_at": "2026-03-29T16:03:15+00:00"}))
    _touch(rl_zip, time.time() - (5 * 86400))

    result = mf.get_model_freshness(tmp_models)
    rl_group = next(g for g in result["groups"] if g["name"] == "rl_position_sizer")
    # Older (30d) wins over younger (5d).
    assert rl_group["age_days"] is not None
    assert rl_group["age_days"] >= 29.0, (
        f"rl_position_sizer age={rl_group['age_days']}d did not pick the "
        f"older meta-trained-at timestamp (30d). zip mtime was 5d. "
        f"This is the C5 hide-staleness bug."
    )


def test_rl_sizer_absent_when_zip_missing(tmp_models):
    """No .zip file → group should not appear (don't fabricate readings).
    Avoids spurious 'unknown' rows polluting the header on machines where
    RL training hasn't run yet."""
    result = mf.get_model_freshness(tmp_models)
    names = [g["name"] for g in result["groups"]]
    assert "rl_position_sizer" not in names


def test_format_freshness_for_prompt_includes_rl_sizer(tmp_models):
    """The Claude reflection prompt header line must list rl_position_sizer
    when present, otherwise the operator (and prior Mythos sessions) had
    no signal that this group existed at all."""
    rl_zip = tmp_models / "rl_position_sizer.zip"
    rl_zip.write_bytes(b"")
    _touch(rl_zip, time.time() - (10 * 86400))

    result = mf.get_model_freshness(tmp_models)
    text = mf.format_freshness_for_prompt(result)
    assert "rl_position_sizer" in text, (
        f"format_freshness_for_prompt did not list rl_position_sizer:\n{text}"
    )
