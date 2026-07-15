"""Contract tests for the Phase-M Training/evidence cockpit projection.

These use real files and real reader modules.  No reader, evidence store, or
forward-capture contract is mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.server.training_cockpit import (
    read_data_status,
    read_evidence_status,
    read_forward_monitoring,
    read_jobs,
    read_live_capture_status,
)
from dashboard.server.training_jobs import LocalTrainingJobRunner, TrainingJobJournal
from src.data_platform.forward_capture import P2ReadinessReport


NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_data_status_reads_real_manifests_and_p2_contract(tmp_path):
    data_root = tmp_path / "axiom-data"
    partition = data_root / "fx" / "normalized" / "eur-usd.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"immutable normalized partition")
    _write_json(
        data_root / "fx" / "normalized" / "manifest.json",
        {"payload": {"partitions": [{"uri": "axiom-data://fx/normalized/eur-usd.parquet"}]}},
    )
    report = P2ReadinessReport(
        generated_at_utc=NOW,
        minimum_trading_days=60,
        required_pairs=("EUR_USD",),
        exposure_trading_days=2,
        tick_trading_days_by_pair={"EUR_USD": 2},
        exposure_first_date=date(2026, 7, 13),
        exposure_last_date=date(2026, 7, 14),
        tick_first_date_by_pair={"EUR_USD": date(2026, 7, 13)},
        tick_last_date_by_pair={"EUR_USD": date(2026, 7, 14)},
        join_key="trading_date_utc",
        ready=False,
        blocking_reasons=("need 58 more exposure trading days",),
    )
    control_root = tmp_path / "forward_capture"
    control_root.mkdir()
    (control_root / "p2_readiness.json").write_text(report.model_dump_json(), encoding="utf-8")

    result = read_data_status(tmp_path, data_root=data_root, control_root=control_root, now=NOW)

    fx = next(row for row in result["domains"] if row["domain"] == "fx")
    normalized = next(row for row in fx["tiers"] if row["tier"] == "normalized")
    assert normalized["manifest_count"] == 1
    assert normalized["partition_count"] == 1
    assert result["quality_failure_count"] == 0
    assert result["p2_readiness"]["available"] is True
    assert result["p2_readiness"]["ready"] is False
    assert result["p2_readiness"]["exposure_trading_days"] == 2


def test_data_status_reports_missing_partition_as_quality_failure(tmp_path):
    data_root = tmp_path / "axiom-data"
    _write_json(
        data_root / "crypto" / "snapshots" / "manifest.json",
        {"payload": {"partitions": [{"uri": "axiom-data://crypto/snapshots/missing.parquet"}]}},
    )

    result = read_data_status(tmp_path, data_root=data_root, control_root=tmp_path / "capture", now=NOW)

    assert result["quality_failure_count"] == 1
    assert result["quality_failures"][0]["kind"] == "missing_partition"
    assert result["p2_readiness"]["available"] is False


def test_data_status_ignores_hidden_staging_manifests(tmp_path):
    data_root = tmp_path / "axiom-data"
    partition = data_root / "fx" / "normalized" / "published" / "partitions" / "batch.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"published partition")
    _write_json(
        data_root / "fx" / "normalized" / "published" / "manifest.json",
        {"partitions": [{"uri": "axiom-data://fx/normalized/published/partitions/batch.parquet"}]},
    )
    _write_json(
        data_root / "fx" / "normalized" / ".batch.tmp-incomplete" / "manifest.json",
        {"partitions": [{"uri": "axiom-data://fx/normalized/final/partitions/not-published.parquet"}]},
    )

    result = read_data_status(
        tmp_path, data_root=data_root, control_root=tmp_path / "capture", now=NOW,
    )

    normalized = next(
        tier
        for domain in result["domains"] if domain["domain"] == "fx"
        for tier in domain["tiers"] if tier["tier"] == "normalized"
    )
    assert normalized["manifest_count"] == 1
    assert normalized["partition_count"] == 1
    assert result["quality_failure_count"] == 0


def test_live_capture_status_separates_accrual_from_p2_duration(tmp_path):
    data_root = tmp_path / "axiom-data"

    def write_stream(domain, family, source_key):
        stream_id = f"forward-capture:{family}:{source_key}"
        component = hashlib.sha256(stream_id.encode("utf-8")).hexdigest()[:32]
        history = data_root / domain / "history" / component
        _write_json(
            history / "00000000000000000000-record.jsonl",
            {
                "captured_at_utc": NOW.isoformat(),
                "observed_at_utc": NOW.isoformat(),
                "source_event_time_utc": NOW.isoformat(),
                "payload": {},
            },
        )

    write_stream("fx", "tick_spread", "EUR_USD")
    write_stream("portfolio", "exposure", "oanda_fx_trend_lane")
    health_path = tmp_path / "trained_data" / "ticks" / "_health.json"
    _write_json(
        health_path,
        {
            "stream_status": "running",
            "ticks_received_total": 125,
            "ticks_written": 100,
            "buffered_ticks": 25,
            "ticks_per_minute": 40,
            "flushes": 4,
            "flush_errors": 0,
        },
    )

    result = read_live_capture_status(
        tmp_path,
        data_root=data_root,
        required_pairs=["EUR_USD"],
        now=NOW,
    )

    assert result["tick"]["status"] == "accruing"
    assert result["tick"]["writer_status"] == "running"
    assert result["tick"]["canonical_batches"] == 1
    assert result["tick"]["pairs_current"] == 1
    assert result["tick"]["buffered_ticks"] == 25
    assert result["exposure"]["status"] == "accruing"
    assert result["exposure"]["canonical_snapshots"] == 1

    stale = (NOW - timedelta(minutes=5)).timestamp()
    os.utime(health_path, (stale, stale))
    degraded = read_live_capture_status(
        tmp_path,
        data_root=data_root,
        required_pairs=["EUR_USD"],
        now=NOW,
    )
    assert degraded["tick"]["writer_status"] == "health_stale"
    assert degraded["tick"]["status"] == "delayed"


def test_evidence_status_preserves_negative_results_and_lineage(tmp_path):
    root = tmp_path / "evidence"
    retired = "a" * 64
    rejected = "b" * 64
    _write_json(
        root / "indexes" / "current.json",
        {
            "packages": {
                retired: {
                    "lane_id": "crypto_momentum_campaign__cross_sectional",
                    "package_id": "old",
                    "state": "RETIRED",
                    "sequence": 4,
                    "head_event_digest": "c" * 64,
                },
                rejected: {
                    "lane_id": "crypto_momentum_campaign__cross_sectional",
                    "package_id": "new",
                    "state": "REJECTED",
                    "sequence": 1,
                    "head_event_digest": "d" * 64,
                },
            },
            "champions": {
                "crypto_momentum_campaign__cross_sectional": {"package_digest": retired},
            },
        },
    )
    _write_json(
        root / "packages" / retired / "envelope.json",
        {"payload": {"created_at": "2026-07-01T00:00:00Z", "job_manifest_digest": "e" * 64}},
    )
    _write_json(
        root / "packages" / rejected / "envelope.json",
        {"payload": {"created_at": "2026-07-14T00:00:00Z", "job_manifest_digest": "f" * 64}},
    )
    _write_json(
        root / "verdicts" / "reject.json",
        {
            "payload": {
                "package_digest": rejected,
                "decision": "REJECT",
                "rejection_reason": "significance gate did not clear",
                "checks": [{"check_id": "significance", "passed": False, "details": "honest negative"}],
            },
            "payload_digest": "1" * 64,
        },
    )

    result = read_evidence_status(root)

    assert result["available"] is True
    assert result["negative_result_count"] == 1
    lane = result["lanes"][0]
    assert lane["family"] == "crypto_momentum"
    assert lane["current"]["package_digest"] == rejected
    assert lane["current"]["negative_result"] is True
    assert lane["current"]["gate_results"][0]["check_id"] == "significance"
    assert lane["prior_champion"]["package_digest"] == retired
    assert lane["champion"]["package_digest"] == retired

    jobs = read_jobs(result, root / "dashboard_jobs")
    assert jobs["counts"]["completed"] == 2
    assert {job["manifest_digest"] for job in jobs["jobs"]} == {"e" * 64, "f" * 64}


def test_forward_monitoring_is_explicitly_absent_until_artifact_lands(tmp_path):
    evidence = {
        "lanes": [{"lane_id": "risk_target_vol", "current": {"package_digest": "a" * 64, "state": "SHADOW"}}],
    }
    result = read_forward_monitoring(evidence, tmp_path / "forward_monitoring")

    assert result["reporting_count"] == 0
    assert result["lanes"][0]["available"] is False
    assert result["lanes"][0]["status"] == "not_reporting"
    assert result["lanes"][0]["forward_sharpe"] is None


def test_job_journal_enforces_atomic_monotonic_transitions(tmp_path):
    journal = TrainingJobJournal(tmp_path / "jobs")
    submitted = journal.transition("job-1", "submitted", fields={"lane": "risk_target"})
    running = journal.transition("job-1", "running")
    completed = journal.transition("job-1", "completed", fields={"manifest_digest": "a" * 64})

    assert submitted["status"] == "submitted"
    assert running["started_at"]
    assert completed["status"] == "completed"
    assert completed["manifest_digest"] == "a" * 64
    with pytest.raises(ValueError, match="invalid job transition"):
        journal.transition("job-1", "running")


def test_job_journal_rejects_caller_supplied_audit_timestamps(tmp_path):
    journal = TrainingJobJournal(tmp_path / "jobs")
    submitted = journal.transition("job-1", "submitted", fields={"submitted_at": "forged", "status": "completed"})
    completed = journal.transition("job-1", "running", fields={"started_at": "forged"})

    assert submitted["status"] == "submitted"
    assert submitted["submitted_at"] != "forged"
    assert completed["started_at"] != "forged"


def test_local_job_runner_persists_running_then_completed_without_blocking_submit(tmp_path):
    root = tmp_path / "jobs"
    journal = TrainingJobJournal(root)
    journal.transition("job-async", "submitted", fields={"lane": "risk_target"})
    runner = LocalTrainingJobRunner(journal)
    started = threading.Event()
    release = threading.Event()

    def work_item():
        started.set()
        assert release.wait(timeout=3)
        return {"job_manifest_digest": "a" * 64}

    future = runner.submit(
        "job-async",
        work_item,
        completion_fields=lambda result, elapsed: {
            "manifest_digest": result["job_manifest_digest"],
            "resource_usage": {"wall_seconds": elapsed},
        },
    )
    assert started.wait(timeout=3)
    running = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
    assert future.done() is False
    assert running["status"] == "running"
    assert running["started_at"]

    release.set()
    assert future.result(timeout=3)["job_manifest_digest"] == "a" * 64
    completed = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["manifest_digest"] == "a" * 64
    assert completed["resource_usage"]["wall_seconds"] >= 0
    runner.shutdown()


def test_local_job_runner_persists_failure(tmp_path):
    root = tmp_path / "jobs"
    journal = TrainingJobJournal(root)
    journal.transition("job-failed", "submitted")
    runner = LocalTrainingJobRunner(journal)

    def fail():
        raise RuntimeError("deterministic worker failure")

    future = runner.submit("job-failed", fail)
    with pytest.raises(RuntimeError, match="deterministic worker failure"):
        future.result(timeout=3)
    failed = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: deterministic worker failure"
    runner.shutdown()


def test_local_job_runner_keeps_success_when_metadata_projection_fails(tmp_path):
    root = tmp_path / "jobs"
    journal = TrainingJobJournal(root)
    journal.transition("job-metadata", "submitted")
    runner = LocalTrainingJobRunner(journal)

    def bad_metadata(_result, _elapsed):
        raise ValueError("metadata formatter broke")

    future = runner.submit(
        "job-metadata",
        lambda: {"trained": True},
        completion_fields=bad_metadata,
    )
    assert future.result(timeout=3) == {"trained": True}
    completed = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["metadata_warning"] == "ValueError: metadata formatter broke"
    assert completed["resource_usage"]["wall_seconds"] >= 0
    runner.shutdown()
