from __future__ import annotations

import json
from datetime import date

import pytest

from src.evidence.hashing import sha256_bytes
from src.evidence.contracts import DispositionState
from src.evidence.track_b.dashboard import track_b_evidence_view
from src.evidence.track_b.evaluation import (
    EvaluationParams, build_evaluation_partition, evaluate_partitions,
)
from src.evidence.track_b.lineage import REQUIRED_SCORE_LINEAGE, score_batch, validate_score_lineage
from src.evidence.track_b.slice import (
    build_evidence_store, build_slice_identities, import_produced,
    produce_evaluation, produce_scoring, utc,
)

NOW = utc(2026, 7, 14, 15)
CAMPAIGN = "track-b-2026q3"
MODEL_CUTOFF = "2025-01-01"


def _jsonl(rows):
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


TEXTS = (
    "record revenue and strong demand continued.",
    "record revenue continued.",
    "record revenue, strong demand, and margin expansion continued.",
    "a material weakness was identified.",
    "going concern and material weakness disclosures remain.",
    "management raised guidance for the next period.",
)


def _raw_batch(*, thin_text: bool = False) -> dict[str, bytes]:
    rows = []
    for index, body in enumerate(TEXTS):
        ticker = f"T{index}"
        entity = f"Issuer {index}"
        document = f"{entity}. " + ("ordinary filing prose." if thin_text else body)
        rows.append({
            "filing_accession": f"000000000{index}-26-000001",
            "ticker": ticker,
            "issuer_name": entity,
            "form": "10-Q",
            "document_text": document,
            "document_hash": sha256_bytes(document.encode()),
            "filing_date": "2026-04-01",
            "as_of_date": "2026-04-02",
            "model_cutoff": MODEL_CUTOFF,
            "parameters": {"arm": "post_cutoff_blinded"},
            "entities_to_blind": [entity, ticker],
        })
    return {"batch-001": _jsonl(rows)}


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def _score(identities, parts):
    return produce_scoring(
        identities, parts, campaign_id=CAMPAIGN, dataset_id="track-b-raw-001",
        model_cutoff=MODEL_CUTOFF, coverage_start=date(2026, 4, 1),
        coverage_end=date(2026, 4, 2), retrieved_at=NOW, created_at=NOW,
        git_commit="3" * 40,
    )


def test_score_artifact_has_every_required_j5_lineage_field():
    artifact, queue, metrics = score_batch("batch-001", _raw_batch()["batch-001"], batch_id="batch-001")
    assert queue == b""
    assert metrics["n_scored"] == 6.0
    for line in artifact.decode().splitlines():
        record = json.loads(line)
        validate_score_lineage(record)
        assert all(field in record for field in REQUIRED_SCORE_LINEAGE)
        assert record["score_validation"]["status"] == "valid"
        assert record["filing_date"] <= record["as_of_date"]
        assert record["post_model_cutoff"] is True


def test_scoring_is_idempotent_and_document_hash_is_load_bearing():
    data = _raw_batch()["batch-001"]
    first = score_batch("batch-001", data, batch_id="batch-001")
    second = score_batch("batch-001", data, batch_id="batch-001")
    assert first == second
    rows = [json.loads(line) for line in data.decode().splitlines()]
    rows[0]["document_text"] += " changed"
    with pytest.raises(ValueError, match="document hash mismatch"):
        score_batch("batch-001", _jsonl(rows), batch_id="batch-001")


def test_pit_violation_fails_closed():
    rows = [json.loads(line) for line in _raw_batch()["batch-001"].decode().splitlines()]
    rows[0]["as_of_date"] = "2026-03-31"
    with pytest.raises(ValueError, match="PIT cutoff"):
        score_batch("batch-001", _jsonl(rows), batch_id="batch-001")


def test_signed_model_cutoff_must_match_every_score_record():
    identities = build_slice_identities(NOW)
    parts = _raw_batch()
    with pytest.raises(ValueError, match="score lineage does not match signed scorer contract"):
        produce_scoring(
            identities, parts, campaign_id=CAMPAIGN, dataset_id="track-b-cutoff-mismatch",
            model_cutoff="2024-01-01", coverage_start=date(2026, 4, 1),
            coverage_end=date(2026, 4, 2), retrieved_at=NOW, created_at=NOW,
            git_commit="3" * 40,
        )


def test_manual_review_queue_blocks_scoring_quarantine(tmp_path):
    identities, store = _fresh(tmp_path)
    parts = _raw_batch(thin_text=True)
    produced = _score(identities, parts)
    outcome = next(iter(import_produced(store, identities, produced, parts, created_at=NOW).values()))
    assert outcome.final_state == DispositionState.REJECTED
    queue = produced.worker_output.heads[0].files["artifacts/manual_review_queue.jsonl"]
    assert len(queue.splitlines()) == 6


def test_valid_scoring_package_stops_at_quarantine(tmp_path):
    identities, store = _fresh(tmp_path)
    parts = _raw_batch()
    produced = _score(identities, parts)
    outcome = next(iter(import_produced(store, identities, produced, parts, created_at=NOW).values()))
    assert outcome.final_state == DispositionState.QUARANTINED
    assert track_b_evidence_view(store.root)["champions"] == {}


def _scoring_artifact(produced) -> bytes:
    return produced.worker_output.heads[0].files["artifacts/scoring_artifact.jsonl"]


def test_evaluation_is_independent_and_refuses_insufficient_forward_time(tmp_path):
    identities, store = _fresh(tmp_path)
    raw = _raw_batch()
    scored = _score(identities, raw)
    score_outcome = next(iter(import_produced(store, identities, scored, raw, created_at=NOW).values()))
    artifact = _scoring_artifact(scored)
    returns = {f"T{index}": value for index, value in enumerate((0.3, 0.2, 0.4, -0.2, -0.4, 0.0))}
    cell_id, cell = build_evaluation_partition(
        artifact, rebalance_date="2026-05-01", fold="fold0", forward_returns=returns,
    )
    partitions = {cell_id: cell}
    produced = produce_evaluation(
        identities, partitions, campaign_id=CAMPAIGN, dataset_id="track-b-eval-001",
        scoring_package_digests=(score_outcome.package_digest,),
        coverage_start=date(2026, 5, 1), coverage_end=date(2026, 6, 1),
        retrieved_at=NOW, created_at=NOW, git_commit="3" * 40,
    )
    outcome = next(iter(import_produced(store, identities, produced, partitions, created_at=NOW).values()))
    assert outcome.final_state == DispositionState.REJECTED
    report = evaluate_partitions(partitions)
    assert report["n_valid_forward_cells"] == 1
    assert report["enough_forward_time"] is False
    assert produced.worker_output.heads[0].package.derived_from_package_digests == (
        score_outcome.package_digest,
    )


def test_three_forward_cells_can_reach_quarantine_but_never_champion(tmp_path):
    identities, store = _fresh(tmp_path)
    raw = _raw_batch()
    scored = _score(identities, raw)
    score_outcome = next(iter(import_produced(store, identities, scored, raw, created_at=NOW).values()))
    artifact = _scoring_artifact(scored)
    returns = {f"T{index}": value for index, value in enumerate((0.3, 0.2, 0.4, -0.2, -0.4, 0.0))}
    partitions = dict(
        build_evaluation_partition(
            artifact, rebalance_date=f"2026-0{month}-01", fold=f"fold{month}",
            forward_returns=returns,
        )
        for month in (5, 6, 7)
    )
    params = EvaluationParams(
        expected_cells=tuple(sorted(partitions)), min_forward_cells=3,
        rank_ic_bar=0.03, placebo_abs_limit=1.0,
    )
    produced = produce_evaluation(
        identities, partitions, campaign_id=CAMPAIGN, dataset_id="track-b-eval-003",
        scoring_package_digests=(score_outcome.package_digest,),
        coverage_start=date(2026, 5, 1), coverage_end=date(2026, 8, 1),
        retrieved_at=NOW, created_at=NOW, git_commit="3" * 40, params=params,
    )
    outcome = next(iter(import_produced(store, identities, produced, partitions, created_at=NOW).values()))
    assert outcome.final_state == DispositionState.QUARANTINED
    assert track_b_evidence_view(store.root)["champions"] == {}


def test_evaluation_rejects_tampered_score_record_lineage():
    artifact, _, _ = score_batch("batch-001", _raw_batch()["batch-001"], batch_id="batch-001")
    returns = {f"T{index}": float(index) for index in range(6)}
    cell_id, data = build_evaluation_partition(
        artifact, rebalance_date="2026-05-01", fold="fold0", forward_returns=returns,
    )
    rows = [json.loads(line) for line in data.decode().splitlines()]
    rows[0]["score_record"]["rationale"] = "doctored"
    with pytest.raises(ValueError, match="score record digest mismatch"):
        evaluate_partitions({cell_id: _jsonl(rows)})


def test_evaluation_rejects_score_artifact_not_owned_by_derived_package(tmp_path):
    identities, store = _fresh(tmp_path)
    raw = _raw_batch()
    scored = _score(identities, raw)
    score_outcome = next(iter(import_produced(store, identities, scored, raw, created_at=NOW).values()))
    # A distinct, valid artifact: change one filing's text and declared hash, then
    # score it directly without importing that scoring package.
    rows = [json.loads(line) for line in raw["batch-001"].decode().splitlines()]
    rows[0]["document_text"] += " free cash flow"
    rows[0]["document_hash"] = sha256_bytes(rows[0]["document_text"].encode())
    other_artifact, _, _ = score_batch("batch-other", _jsonl(rows), batch_id="batch-other")
    returns = {f"T{index}": value for index, value in enumerate((0.3, 0.2, 0.4, -0.2, -0.4, 0.0))}
    cell_id, cell = build_evaluation_partition(
        other_artifact, rebalance_date="2026-05-01", fold="fold0", forward_returns=returns,
    )
    partitions = {cell_id: cell}
    produced = produce_evaluation(
        identities, partitions, campaign_id=CAMPAIGN, dataset_id="track-b-unbound-artifact",
        scoring_package_digests=(score_outcome.package_digest,),
        coverage_start=date(2026, 5, 1), coverage_end=date(2026, 6, 1),
        retrieved_at=NOW, created_at=NOW, git_commit="3" * 40,
    )
    outcome = next(iter(import_produced(store, identities, produced, partitions, created_at=NOW).values()))
    checks = {check.check_id: check.passed for check in outcome.verdict.checks}
    assert outcome.final_state == DispositionState.REJECTED
    assert checks["scoring_packages_quarantined"] is True
    assert checks["score_artifacts_bound_to_packages"] is False
