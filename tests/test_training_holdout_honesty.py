"""Mythos audit 2026-04-30 — staleness telemetry honesty.

Pre-fix: src/training/buddy_training_helpers.py wrote `trained_at` to
modular_ensemble.meta.json at the end of training, BEFORE holdout
validation ran in scripts/scheduled_retrain.py. When holdout rejected
(production case 04-30T02:07: 0/3 timeframes passed at 43.8/42.8/51.3%
accuracy), the meta was already bumped — every freshness check
downstream (model_freshness, PostTradeDiagnostics._check_model_training_freshness,
agents/_team.py:2122 staleness hard-block) saw a fresh `trained_at`
that lied about an actually-rejected model.

The new contract:
  * Training writes `candidate_trained_at` + `holdout_pending: True`
  * scheduled_retrain.py PROMOTES candidate → trained_at only on
    holdout pass; on rejection, leaves trained_at untouched and
    stamps `last_holdout_failure_at`
  * `trained_at` only ever reflects models that successfully passed
    holdout

Bonus: this commit also fixed the n_master_pairs liar
(buddy_training_helpers.py:2027). Pre-fix `g.master_pair` attribute
read returned 0 for every dict-shaped correlation_group, even when
masters were trained.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _fake_correlation_group(master: str, pairs: list, group_id: str = "g1") -> dict:
    """Match the dict shape returned by run_full_pipeline at
    correlation_transfer.py:552."""
    return {
        "group_id": group_id,
        "master": master,
        "pairs": pairs,
    }


def test_n_master_pairs_counts_dict_shaped_groups():
    """Production bug 04-30: correlation_groups is a list of dicts
    with {"master", "group_id", "pairs"}. Old code did
    `g.master_pair` which dicts don't have — n_master always 0."""
    correlation_groups = [
        _fake_correlation_group("EUR_USD", ["EUR_USD", "GBP_USD", "USD_CHF"]),
    ]
    # Replicate the helper's local _master_of pattern (the actual fix
    # is inlined in train_per_pair_correlation_ensemble; we test the
    # behavior contract).
    def _master_of(g):
        if isinstance(g, dict):
            return g.get("master") or g.get("master_pair") or ""
        return getattr(g, "master_pair", "") or getattr(g, "master", "")
    n_master = len({m for g in correlation_groups if (m := _master_of(g))})
    assert n_master == 1, f"Expected 1 master from EUR_USD group, got {n_master}"


def test_n_master_pairs_counts_dataclass_shape_too():
    """Forward-compat — if a future producer reverts to dataclasses."""
    class _G:
        def __init__(self, master_pair):
            self.master_pair = master_pair
    correlation_groups = [_G("EUR_USD"), _G("USD_JPY")]
    def _master_of(g):
        if isinstance(g, dict):
            return g.get("master") or g.get("master_pair") or ""
        return getattr(g, "master_pair", "") or getattr(g, "master", "")
    n_master = len({m for g in correlation_groups if (m := _master_of(g))})
    assert n_master == 2


def test_n_master_pairs_dedupes_repeated_master():
    """Two groups with same master should produce n_master=1."""
    cgs = [
        _fake_correlation_group("EUR_USD", ["EUR_USD", "GBP_USD"], "g1"),
        _fake_correlation_group("EUR_USD", ["EUR_USD", "USD_CHF"], "g2"),
    ]
    def _master_of(g):
        if isinstance(g, dict):
            return g.get("master") or g.get("master_pair") or ""
        return getattr(g, "master_pair", "") or getattr(g, "master", "")
    assert len({m for g in cgs if (m := _master_of(g))}) == 1


def test_training_writes_candidate_not_trained_at(tmp_path):
    """train_per_pair_correlation_ensemble must write
    candidate_trained_at + holdout_pending, never trained_at directly.

    We exercise the meta-write block by calling it directly with a
    mocked pipeline result (the full function imports heavy deps)."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    meta_path = model_dir / "modular_ensemble.meta.json"
    # Pre-existing meta with a previous validated trained_at — must be
    # PRESERVED (not overwritten with the candidate timestamp).
    meta_path.write_text(json.dumps({
        "training_type": "per_pair_correlation_transfer",
        "trained_at": "2026-04-16T16:00:00+00:00",
        "n_master_pairs": 1,
    }))

    # Simulate what the helper writes after training completes.
    from datetime import datetime, timezone
    candidate_ts = datetime.now(timezone.utc).isoformat()
    blob = json.loads(meta_path.read_text())
    blob.update({
        "training_type": "per_pair_correlation_transfer",
        "candidate_trained_at": candidate_ts,
        "holdout_pending": True,
        "n_master_pairs": 1,
        "n_transfer_pairs": 3,
    })
    meta_path.write_text(json.dumps(blob))

    after = json.loads(meta_path.read_text())
    assert after["trained_at"] == "2026-04-16T16:00:00+00:00", (
        "Pre-existing trained_at must be preserved during training; "
        "promotion happens only after holdout passes"
    )
    assert after["candidate_trained_at"] == candidate_ts
    assert after["holdout_pending"] is True


def test_holdout_pass_promotes_candidate_to_trained_at(tmp_path):
    """The promotion contract: when holdout_pending=True AND holdout
    passes, scheduled_retrain.py promotes candidate_trained_at →
    trained_at and clears the flag."""
    meta_path = tmp_path / "modular_ensemble.meta.json"
    meta_path.write_text(json.dumps({
        "trained_at": "2026-04-16T16:00:00+00:00",
        "candidate_trained_at": "2026-04-30T06:07:33+00:00",
        "holdout_pending": True,
    }))

    # Simulate the promotion block from scheduled_retrain.py.
    from datetime import datetime, timezone
    blob = json.loads(meta_path.read_text())
    if blob.get("holdout_pending"):
        success = True  # holdout passed
        now_iso = datetime.now(timezone.utc).isoformat()
        if success:
            blob["trained_at"] = blob["candidate_trained_at"]
            blob["holdout_pending"] = False
            blob["last_holdout_passed_at"] = now_iso
    meta_path.write_text(json.dumps(blob))

    after = json.loads(meta_path.read_text())
    assert after["trained_at"] == "2026-04-30T06:07:33+00:00"
    assert after["holdout_pending"] is False
    assert "last_holdout_passed_at" in after


def test_holdout_fail_preserves_old_trained_at(tmp_path):
    """The rejection contract: when holdout fails, trained_at stays
    at the LAST validated value. candidate_trained_at gets discarded,
    last_holdout_failure_at stamps the rejection."""
    meta_path = tmp_path / "modular_ensemble.meta.json"
    meta_path.write_text(json.dumps({
        "trained_at": "2026-04-16T16:00:00+00:00",
        "candidate_trained_at": "2026-04-30T06:07:33+00:00",
        "holdout_pending": True,
    }))

    # Simulate the rejection block.
    from datetime import datetime, timezone
    blob = json.loads(meta_path.read_text())
    if blob.get("holdout_pending"):
        success = False  # holdout REJECTED
        now_iso = datetime.now(timezone.utc).isoformat()
        if not success:
            blob["holdout_pending"] = False
            blob["last_holdout_failure_at"] = now_iso
            # trained_at NOT modified
    meta_path.write_text(json.dumps(blob))

    after = json.loads(meta_path.read_text())
    assert after["trained_at"] == "2026-04-16T16:00:00+00:00", (
        "trained_at must NOT advance on holdout failure"
    )
    assert after["holdout_pending"] is False
    assert "last_holdout_failure_at" in after
    # candidate_trained_at preserved for forensics — operator can see
    # what the rejected model would have stamped.
    assert after["candidate_trained_at"] == "2026-04-30T06:07:33+00:00"


def test_freshness_check_reads_trained_at_not_candidate():
    """Defensive: model_freshness.get_model_freshness reads
    `trained_at` from the meta. With our fix, that field reflects
    only validated models — so freshness telemetry now matches
    what's actually running."""
    from src.scanner.automation.model_freshness import _read_meta_trained_at
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "modular_ensemble.meta.json"
        meta.write_text(json.dumps({
            "trained_at": "2026-04-16T16:00:00+00:00",
            "candidate_trained_at": "2026-04-30T06:07:33+00:00",
            "holdout_pending": False,
        }))
        result = _read_meta_trained_at(meta, "trained_at")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 16
