"""Task 5 registry: append-only signed attempts and immutable candidate identity."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing

import pytest

from tests.evidence.equity_research._support import NOW, COMMIT, build_context, proposal
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole
from src.evidence.signing import verify_envelope, Ed25519Signer
from src.evidence.store import ConcurrentHeadError, EvidenceStoreError, StoreCorruptionError


def test_registration_is_durable_signed_and_idempotent(context):
    from src.evidence.equity_research.experiment_registry import ResearchEvent

    registered = context.registry.register_experiment(proposal())
    receipts = context.store.load_research_ledger()
    assert len(receipts) == 1
    event = verify_envelope(receipts[0].envelope, ResearchEvent, context.store.trust_store)
    assert event.kind == "REGISTERED"
    assert registered.proposal == proposal()
    assert context.registry.register_experiment(proposal()) == registered
    assert len(context.store.load_research_ledger()) == 1
    reopened = build_context(context.store.root, context.signers)
    assert reopened.registry.snapshot().attempt_count == 1
    assert reopened.registry.snapshot().experiments["exp-1"] == registered


def test_failed_attempts_are_retained_and_counted_at_freeze(context):
    context.registry.register_experiment(proposal("failed-1"))
    context.registry.record_failure("failed-1", "no useful edge")
    context.registry.register_experiment(proposal("exp-2"))
    candidate = context.registry.freeze_candidate("exp-2", "c" * 64, COMMIT)
    state = context.registry.snapshot()
    assert state.attempt_count == candidate.attempt_count == 2
    assert state.experiments["failed-1"].failure_reason == "no useful edge"
    assert state.experiments["failed-1"].proposal == proposal("failed-1")
    assert [r.envelope.payload["kind"] for r in context.store.load_research_ledger()] == [
        "REGISTERED",
        "FAILED",
        "REGISTERED",
        "FROZEN",
    ]


def test_duplicate_id_cannot_replace_a_proposal(context):
    context.registry.register_experiment(proposal())
    with pytest.raises(ValueError, match="registered"):
        context.registry.register_experiment(replace(proposal(), hypothesis="different"))
    assert context.registry.snapshot().experiments["exp-1"].proposal == proposal()


def test_freeze_requires_preregistration(context):
    with pytest.raises(ValueError, match="registered"):
        context.registry.freeze_candidate("unknown", "c" * 64, COMMIT)
    assert not context.store.load_research_ledger()


def test_freeze_is_immutable_and_idempotent(context, frozen):
    assert context.registry.freeze_candidate("exp-1", "c" * 64, COMMIT) == frozen
    for artifact, commit in [("d" * 64, COMMIT), ("c" * 64, "d" * 40)]:
        with pytest.raises(ValueError):
            context.registry.freeze_candidate("exp-1", artifact, commit)
    assert (
        build_context(context.store.root, context.signers).registry.snapshot().candidates[frozen.candidate_id] == frozen
    )
    assert not list((context.store.root / "champions").iterdir())


def test_frozen_candidate_keeps_its_original_search_count(context, frozen):
    context.registry.register_experiment(proposal("later"))
    assert frozen.attempt_count == 1
    assert context.registry.snapshot().attempt_count == 2
    assert context.registry.snapshot().candidates[frozen.candidate_id].attempt_count == 1


def test_failed_attempt_cannot_freeze(context):
    context.registry.register_experiment(proposal())
    context.registry.record_failure("exp-1", "failed")
    with pytest.raises(ValueError, match="failed"):
        context.registry.freeze_candidate("exp-1", "c" * 64, COMMIT)


@pytest.mark.parametrize("value", ["", "A" * 64, "z" * 64, "a" * 63, None, True, 4])
def test_invalid_artifact_identity_fails_before_writing(context, value):
    context.registry.register_experiment(proposal())
    with pytest.raises((TypeError, ValueError)):
        context.registry.freeze_candidate("exp-1", value, COMMIT)
    assert len(context.store.load_research_ledger()) == 1


@pytest.mark.parametrize("commit", ["a" * 7, "a" * 41, "a" * 63, "A" * 40, "b" * 40])
def test_commit_is_full_and_matches_registered_proposal(context, commit):
    context.registry.register_experiment(proposal())
    with pytest.raises(ValueError):
        context.registry.freeze_candidate("exp-1", "c" * 64, commit)
    assert len(context.store.load_research_ledger()) == 1


def test_snapshot_values_are_immutable(context, frozen):
    state = context.registry.snapshot()
    with pytest.raises((AttributeError, TypeError)):
        state.candidates[frozen.candidate_id] = frozen
    with pytest.raises((AttributeError, TypeError)):
        frozen.artifact_hash = "d" * 64
    with pytest.raises((AttributeError, TypeError)):
        state.experiments["exp-1"].proposal.hypothesis = "changed"


def test_producer_has_no_registry_authority(context):
    from src.evidence.equity_research.experiment_registry import ExperimentRegistry

    unprivileged = ExperimentRegistry(
        context.store, signer=context.signers[AuthorityRole.PRODUCER], actor_id="producer"
    )
    with pytest.raises(ValueError):
        unprivileged.register_experiment(proposal())
    assert not context.store.load_research_ledger()


def test_revoked_signer_cannot_append_or_use_idempotent_shortcut(context):
    context.registry.register_experiment(proposal())
    signer = context.signers[AuthorityRole.LOCAL_IMPORTER]
    context.store.trust_store.revoke(signer.key_id, NOW)
    with pytest.raises((ValueError, EvidenceStoreError)):
        context.registry.register_experiment(proposal())


def test_tampered_proposal_is_revalidated(context):
    item = proposal()
    object.__setattr__(item, "feature_families", ("news",))
    with pytest.raises(ValueError):
        context.registry.register_experiment(item)
    assert not context.store.load_research_ledger()


def test_signed_history_tampering_is_not_accepted(context):
    context.registry.register_experiment(proposal())
    path = next((context.store.root / "research").glob("*.json"))
    raw = json.loads(path.read_text())
    raw["envelope"]["payload"]["body"]["proposal"]["hypothesis"] = "changed"
    path.write_bytes(canonical_bytes(raw))
    with pytest.raises((ValueError, EvidenceStoreError)):
        context.registry.snapshot()
    with pytest.raises((ValueError, EvidenceStoreError)):
        context.registry.register_experiment(proposal("later"))


def test_missing_interior_event_is_corruption(context):
    for i in range(3):
        context.registry.register_experiment(proposal(f"exp-{i}"))
    paths = sorted((context.store.root / "research").glob("*.json"))
    paths[1].unlink()
    with pytest.raises(StoreCorruptionError):
        context.registry.snapshot()


def test_store_cas_rejects_a_stale_head(context):
    context.registry.register_experiment(proposal())
    event = context.store.load_research_ledger()[0].envelope
    with pytest.raises(ConcurrentHeadError):
        context.store.append_research_event(event, expected_head_digest=None)
    assert len(context.store.load_research_ledger()) == 1


def _register_worker(root, key_bytes, barrier, queue, identifier):
    keys = {AuthorityRole(role): Ed25519Signer.from_private_bytes(value) for role, value in key_bytes.items()}
    ctx = build_context(root, keys)
    ctx.store._trusted_clock = lambda: datetime.now(timezone.utc)
    barrier.wait(timeout=15)
    try:
        ctx.registry.register_experiment(proposal(identifier))
        queue.put("ok")
    except (ValueError, EvidenceStoreError) as exc:
        queue.put(type(exc).__name__)


@pytest.mark.parametrize("same_id", [False, True])
def test_multiprocess_registration_has_no_lost_or_duplicate_attempts(context, same_id):
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    keys = {role.value: signer.private_bytes() for role, signer in context.signers.items()}
    procs = [
        ctx.Process(
            target=_register_worker,
            args=(context.store.root, keys, barrier, queue, "same" if same_id else f"worker-{i}"),
        )
        for i in range(2)
    ]
    for p in procs:
        p.start()
    try:
        results = [queue.get(timeout=25) for _ in procs]
        assert results == ["ok", "ok"]
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
    assert all(p.exitcode == 0 for p in procs)
    assert context.registry.snapshot().attempt_count == (1 if same_id else 2)


@pytest.mark.parametrize("hypothesis", ["  intentional whitespace  ", "市場 — test", "line one\nline two"])
def test_registration_preserves_exact_proposal_text(context, hypothesis):
    item = replace(proposal(), hypothesis=hypothesis)
    registered = context.registry.register_experiment(item)
    assert registered.proposal == item
    assert context.registry.register_experiment(item) == registered


@pytest.mark.parametrize(
    "change",
    [
        {"attempt_count": 0},
        {"attempt_count": True},
        {"registration_digest": "d" * 64},
        {"code_commit": "d" * 40},
        {"artifact_hash": "invalid"},
        {"extra": "not allowed"},
    ],
)
def test_signed_but_false_candidate_claims_are_rejected_by_store(context, change):
    from src.evidence.equity_research.experiment_registry import ResearchEvent

    reg = context.registry.register_experiment(proposal())
    state = context.registry.snapshot()
    body = {
        "experiment_id": "exp-1",
        "registration_digest": reg.registration_digest,
        "artifact_hash": "c" * 64,
        "code_commit": COMMIT,
        "attempt_count": 1,
        **change,
    }
    event = ResearchEvent(
        sequence=state.event_count,
        previous_event_digest=state.head_digest,
        kind="FROZEN",
        actor_id="local_importer",
        occurred_at=NOW,
        body=body,
    )
    envelope = context.signers[AuthorityRole.LOCAL_IMPORTER].sign(event, created_at=NOW)
    with pytest.raises(StoreCorruptionError):
        context.store.append_research_event(envelope, expected_head_digest=state.head_digest)
    assert len(context.store.load_research_ledger()) == 1


@pytest.mark.parametrize("mutation", ["signature", "noncanonical", "unknown-version", "fork", "unexpected-file"])
def test_corrupt_history_blocks_reads_and_new_writes(context, mutation):
    context.registry.register_experiment(proposal())
    path = next((context.store.root / "research").glob("*.json"))
    raw = json.loads(path.read_text())
    if mutation == "signature":
        raw["envelope"]["signature"]["signature_b64"] = "AAAA"
        path.write_bytes(canonical_bytes(raw))
    elif mutation == "noncanonical":
        path.write_text(json.dumps(raw, indent=2))
    elif mutation == "unknown-version":
        raw["schema_version"] = "9.0.0"
        path.write_bytes(canonical_bytes(raw))
    elif mutation == "fork":
        path.with_name("00000000000000000000-" + "f" * 64 + ".json").write_bytes(path.read_bytes())
    else:
        (path.parent / "surprise.txt").write_text("corrupt")
    with pytest.raises(StoreCorruptionError):
        context.registry.snapshot()
    with pytest.raises(StoreCorruptionError):
        context.registry.register_experiment(proposal("next"))


@pytest.mark.parametrize("target", ["journal", "receipt"])
def test_journal_symlinks_fail_closed(context, tmp_path, target):
    context.registry.register_experiment(proposal())
    journal = context.store.root / "research"
    if target == "journal":
        journal.rename(tmp_path / "moved")
        journal.symlink_to(tmp_path / "moved", target_is_directory=True)
    else:
        path = next(journal.glob("*.json"))
        path.rename(tmp_path / "moved.json")
        path.symlink_to(tmp_path / "moved.json")
    with pytest.raises(StoreCorruptionError):
        context.registry.snapshot()


def test_file_in_place_of_research_directory_is_reported_as_corruption(context):
    (context.store.root / "research").write_text("not a directory")
    with pytest.raises(StoreCorruptionError):
        context.registry.snapshot()


def test_unpublished_crash_temporary_is_not_a_committed_attempt(context):
    context.registry.register_experiment(proposal())
    (context.store.root / "research/.receipt.tmp-interrupted").write_text("incomplete")
    assert context.registry.snapshot().attempt_count == 1
    context.registry.register_experiment(proposal("next"))
    assert context.registry.snapshot().attempt_count == 2


def test_timestamp_regression_cannot_rewrite_search_order(context):
    context.registry.register_experiment(proposal())
    context.store._trusted_clock = lambda: NOW - timedelta(seconds=1)
    with pytest.raises(StoreCorruptionError):
        context.registry.register_experiment(proposal("backdated"))
    assert context.registry.snapshot().attempt_count == 1


@pytest.mark.parametrize("delta", [-31, 31])
def test_signed_event_with_untrusted_receipt_timing_is_rejected(context, delta):
    from src.evidence.equity_research.experiment_registry import ResearchEvent
    from dataclasses import asdict

    instant = NOW + timedelta(seconds=delta)
    event = ResearchEvent(
        sequence=0,
        previous_event_digest=None,
        kind="REGISTERED",
        actor_id="local_importer",
        occurred_at=instant,
        body={"proposal": json.loads(canonical_bytes(asdict(proposal())))},
    )
    envelope = context.signers[AuthorityRole.LOCAL_IMPORTER].sign(event, created_at=instant)
    with pytest.raises(StoreCorruptionError):
        context.store.append_research_event(envelope, expected_head_digest=None)
    assert context.registry.snapshot().attempt_count == 0


def test_functional_entrypoints_use_explicit_registry(context):
    from src.evidence.equity_research.experiment_registry import register_experiment, freeze_candidate

    assert register_experiment(proposal(), registry=context.registry).proposal == proposal()
    frozen = freeze_candidate("exp-1", "c" * 64, COMMIT, registry=context.registry)
    assert frozen.experiment_id == "exp-1"


def test_fully_valid_research_events_do_not_populate_champions_or_dispositions(context, frozen):
    assert context.store.rebuild_indexes()
    assert json.loads(context.store.current_index_bytes()) == {
        "schema_version": "1.0.0",
        "packages": {},
        "champions": {},
    }
    assert not list((context.store.root / "dispositions").iterdir())
    assert not list((context.store.root / "champions").iterdir())


@pytest.mark.parametrize("change", ["duplicate", "missing-field", "wrong-array", "wrong-schema", "clock-mismatch"])
def test_direct_signed_registration_cannot_bypass_validation(context, change):
    from dataclasses import asdict
    from src.evidence.equity_research.experiment_registry import ResearchEvent

    data = json.loads(canonical_bytes(asdict(proposal())))
    if change == "duplicate":
        context.registry.register_experiment(proposal())
    if change == "missing-field":
        data.pop("hypothesis")
    if change == "wrong-array":
        data["feature_families"] = "price_return"
    if change == "wrong-schema":
        data["schema_version"] = "9.0.0"
    state = context.registry.snapshot()
    event = ResearchEvent(
        sequence=state.event_count,
        previous_event_digest=state.head_digest,
        kind="REGISTERED",
        actor_id="local_importer",
        occurred_at=NOW,
        body={"proposal": data},
    )
    signed_at = NOW + timedelta(seconds=1) if change == "clock-mismatch" else NOW
    envelope = context.signers[AuthorityRole.LOCAL_IMPORTER].sign(event, created_at=signed_at)
    with pytest.raises(StoreCorruptionError):
        context.store.append_research_event(envelope, expected_head_digest=state.head_digest)
    assert context.registry.snapshot().event_count == state.event_count


def test_repeating_failure_does_not_change_the_record_or_count(context):
    context.registry.register_experiment(proposal())
    first = context.registry.record_failure("exp-1", "failed")
    assert context.registry.record_failure("exp-1", "failed") == first
    assert len(context.store.load_research_ledger()) == 2
    with pytest.raises(ValueError, match="failed"):
        context.registry.record_failure("exp-1", "different failure")


@pytest.mark.parametrize("identifier", ["", " padded ", None, True])
def test_invalid_ids_do_not_become_paths(context, identifier):
    with pytest.raises((TypeError, ValueError)):
        context.registry.record_failure(identifier, "failed")
    assert context.registry.snapshot().attempt_count == 0


def test_direct_store_append_does_not_trust_a_research_producer(context):
    from dataclasses import asdict
    from src.evidence.equity_research.experiment_registry import ResearchEvent

    event = ResearchEvent(
        sequence=0,
        previous_event_digest=None,
        kind="REGISTERED",
        actor_id="producer",
        occurred_at=NOW,
        body={"proposal": json.loads(canonical_bytes(asdict(proposal())))},
    )
    envelope = context.signers[AuthorityRole.PRODUCER].sign(event, created_at=NOW)
    with pytest.raises(StoreCorruptionError):
        context.store.append_research_event(envelope, expected_head_digest=None)
    assert not context.store.load_research_ledger()


def test_worker_module_does_not_depend_on_ambient_conftest(tmp_path):
    from pathlib import Path
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    (tmp_path / "conftest.py").write_text("SOURCE = 'unrelated conftest'\n")
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import conftest
assert conftest.SOURCE == 'unrelated conftest'
sys.path.insert(0, sys.argv[2])
import tests.evidence.equity_research.test_experiment_registry
import tests.evidence.equity_research.test_holdout
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path), str(root)],
        cwd=tmp_path,
        env={"PATH": os.defpath},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
