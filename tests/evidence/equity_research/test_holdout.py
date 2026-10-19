"""Sealed holdout lifecycle: durable opening, no reuse and no result substitution."""

from datetime import timedelta
import multiprocessing

import pytest

from tests.evidence.equity_research._support import NOW, COMMIT, build_context, proposal
from src.evidence.contracts import AuthorityRole
from src.evidence.signing import Ed25519Signer
from src.evidence.store import EvidenceStoreError


def test_holdout_consumption_links_exact_frozen_candidate(context, frozen):
    context.evaluator.assert_untouched("sealed-1")
    opened = context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    assert opened.status == "OPENED" and opened.result_hash is None
    consumed = context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", "d" * 64)
    assert consumed.status == "CONSUMED"
    assert consumed.candidate_id == frozen.candidate_id and consumed.result_hash == "d" * 64
    assert consumed.opened_digest == opened.opened_digest
    assert consumed.attempt_count == 1
    assert build_context(context.store.root, context.signers).evaluator.get_holdout("sealed-1") == consumed


def test_result_cannot_be_attached_before_durable_open(context, frozen):
    with pytest.raises(ValueError, match="opened"):
        context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", "d" * 64)
    assert context.evaluator.get_holdout("sealed-1").status == "UNTOUCHED"


def test_opening_burns_holdout_even_when_evaluator_crashes(context, frozen):
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    reopened = build_context(context.store.root, context.signers)
    with pytest.raises(ValueError, match="untouched"):
        reopened.evaluator.assert_untouched("sealed-1")
    with pytest.raises(ValueError):
        reopened.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    assert reopened.evaluator.get_holdout("sealed-1").status == "OPENED"


@pytest.mark.parametrize("result", ["d" * 64, "e" * 64])
def test_consumed_holdout_cannot_be_reused_even_by_same_candidate(context, frozen, result):
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", "d" * 64)
    with pytest.raises(ValueError, match="consumed"):
        context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", result)
    with pytest.raises(ValueError, match="untouched"):
        context.evaluator.assert_untouched("sealed-1")


def test_other_candidate_cannot_take_over_opened_holdout(context, frozen):
    context.registry.register_experiment(proposal("exp-2"))
    other = context.registry.freeze_candidate("exp-2", "e" * 64, COMMIT)
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    for action in (context.evaluator.open_holdout, lambda c, h: context.evaluator.consume_holdout(c, h, "d" * 64)):
        with pytest.raises(ValueError):
            action(other.candidate_id, "sealed-1")


def test_only_predeclared_holdout_can_be_opened(context):
    context.registry.register_experiment(proposal())
    candidate = context.registry.freeze_candidate("exp-1", "c" * 64, COMMIT)
    context.operator.register_holdout("late", "b" * 64, NOW - timedelta(days=60), NOW - timedelta(days=30))
    with pytest.raises(ValueError, match="before"):
        context.evaluator.open_holdout(candidate.candidate_id, "late")


def test_unknown_candidate_or_holdout_is_not_virgin_evidence(context, frozen):
    with pytest.raises(ValueError):
        context.evaluator.open_holdout("f" * 64, "sealed-1")
    with pytest.raises(ValueError):
        context.evaluator.open_holdout(frozen.candidate_id, "unknown")
    with pytest.raises(ValueError):
        context.evaluator.assert_untouched("unknown")


@pytest.mark.parametrize("kind", ["same-dataset", "overlap"])
def test_renaming_holdout_does_not_make_existing_data_new(context, frozen, kind):
    digest = "b" * 64 if kind == "same-dataset" else "f" * 64
    start, end = (NOW, NOW + timedelta(days=30)) if kind == "same-dataset" else (NOW - timedelta(days=45), NOW)
    with pytest.raises(ValueError):
        context.operator.register_holdout("renamed", digest, start, end)


def test_disjoint_future_period_can_be_registered(context, frozen):
    new = context.operator.register_holdout("future", "f" * 64, NOW, NOW + timedelta(days=30))
    assert new.status == "UNTOUCHED"


def test_research_authority_cannot_open_or_consume(context, frozen):
    from src.evidence.equity_research.holdout import HoldoutAuthority

    agent = HoldoutAuthority(
        context.store, signer=context.signers[AuthorityRole.LOCAL_IMPORTER], actor_id="local_importer"
    )
    with pytest.raises(ValueError):
        agent.open_holdout(frozen.candidate_id, "sealed-1")
    assert context.evaluator.get_holdout("sealed-1").status == "UNTOUCHED"


def test_search_count_at_open_includes_later_failed_attempts(context, frozen):
    context.registry.register_experiment(proposal("later-failed"))
    context.registry.record_failure("later-failed", "failed")
    opened = context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    assert frozen.attempt_count == 1 and opened.attempt_count == 2


def test_failure_after_freeze_cannot_open_holdout(context, frozen):
    context.registry.record_failure("exp-1", "candidate evaluation failed")
    with pytest.raises(ValueError, match="failed"):
        context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")


@pytest.mark.parametrize("result", [None, "", "x" * 64, "A" * 64, "f" * 63, True])
def test_invalid_result_digest_does_not_complete_consumption(context, frozen, result):
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    with pytest.raises((TypeError, ValueError)):
        context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", result)
    assert context.evaluator.get_holdout("sealed-1").status == "OPENED"


def _open_worker(root, key_bytes, barrier, queue, candidate):
    keys = {AuthorityRole(role): Ed25519Signer.from_private_bytes(value) for role, value in key_bytes.items()}
    ctx = build_context(root, keys)
    barrier.wait(timeout=15)
    try:
        ctx.evaluator.open_holdout(candidate, "sealed-1")
        queue.put("opened")
    except (ValueError, EvidenceStoreError):
        queue.put("rejected")


def test_multiprocess_open_has_exactly_one_winner(context, frozen):
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    keys = {role.value: signer.private_bytes() for role, signer in context.signers.items()}
    procs = [
        ctx.Process(target=_open_worker, args=(context.store.root, keys, barrier, queue, frozen.candidate_id))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    try:
        assert sorted(queue.get(timeout=25) for _ in procs) == ["opened", "rejected"]
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
    assert all(p.exitcode == 0 for p in procs)
    assert [r.envelope.payload["kind"] for r in context.store.load_research_ledger()].count("HOLDOUT_OPENED") == 1


@pytest.mark.parametrize("same", ["actor", "key"])
def test_evaluator_independence_is_enforced_even_with_extra_role_bindings(context, frozen, same):
    from src.evidence.equity_research.holdout import HoldoutAuthority

    key = context.signers[AuthorityRole.LOCAL_IMPORTER if same == "key" else AuthorityRole.INDEPENDENT_VERIFIER]
    actor = "local_importer" if same == "actor" else "other-evaluator"
    context.store.authorities.register(actor_id=actor, role=AuthorityRole.INDEPENDENT_VERIFIER, key_ids=(key.key_id,))
    evaluator = HoldoutAuthority(context.store, signer=key, actor_id=actor)
    with pytest.raises(ValueError, match="independent"):
        evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    assert context.evaluator.get_holdout("sealed-1").status == "UNTOUCHED"


def test_declaring_the_same_holdout_again_does_not_reset_consumption(context, frozen):
    original = context.evaluator.get_holdout("sealed-1")
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    consumed = context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", "d" * 64)
    repeated = context.operator.register_holdout(
        "sealed-1", original.dataset_digest, original.coverage_start, original.coverage_end
    )
    assert repeated == consumed


def test_new_candidate_cannot_reuse_consumed_period(context, frozen):
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    context.evaluator.consume_holdout(frozen.candidate_id, "sealed-1", "d" * 64)
    context.registry.register_experiment(proposal("next-family"))
    other = context.registry.freeze_candidate("next-family", "e" * 64, COMMIT)
    with pytest.raises(ValueError, match="untouched"):
        context.evaluator.open_holdout(other.candidate_id, "sealed-1")


def test_candidate_cannot_cherry_pick_a_second_holdout(context):
    context.registry.register_experiment(proposal())
    for i in range(2):
        context.operator.register_holdout(
            f"h{i}", str(i) * 64, NOW + timedelta(days=i * 10), NOW + timedelta(days=(i + 1) * 10)
        )
    candidate = context.registry.freeze_candidate("exp-1", "c" * 64, COMMIT)
    context.evaluator.open_holdout(candidate.candidate_id, "h0")
    with pytest.raises(ValueError, match="already opened"):
        context.evaluator.open_holdout(candidate.candidate_id, "h1")


@pytest.mark.parametrize(
    "start,end", [(NOW, NOW), (NOW, NOW - timedelta(days=1)), (NOW.replace(tzinfo=None), NOW), ("yesterday", NOW)]
)
def test_unknown_or_invalid_holdout_coverage_is_rejected(context, start, end):
    with pytest.raises((TypeError, ValueError)):
        context.operator.register_holdout("bad", "b" * 64, start, end)
    assert not context.store.load_research_ledger()


def test_functional_consumption_entrypoint(context, frozen):
    from src.evidence.equity_research.holdout import consume_holdout

    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    result = consume_holdout(frozen.candidate_id, "sealed-1", "d" * 64, authority=context.evaluator)
    assert result.status == "CONSUMED"


def test_declared_holdout_identity_cannot_be_changed(context, frozen):
    record = context.evaluator.get_holdout("sealed-1")
    with pytest.raises(ValueError, match="immutable"):
        context.operator.register_holdout("sealed-1", "e" * 64, record.coverage_start, record.coverage_end)


def test_store_rejects_false_attempt_count_even_when_evaluator_signed(context, frozen):
    from src.evidence.equity_research.experiment_registry import ResearchEvent

    state = context.registry.snapshot()
    event = ResearchEvent(
        sequence=state.event_count,
        previous_event_digest=state.head_digest,
        kind="HOLDOUT_OPENED",
        actor_id="independent_verifier",
        occurred_at=NOW,
        body={"candidate_id": frozen.candidate_id, "holdout_id": "sealed-1", "attempt_count": 0},
    )
    envelope = context.signers[AuthorityRole.INDEPENDENT_VERIFIER].sign(event, created_at=NOW)
    with pytest.raises(EvidenceStoreError):
        context.store.append_research_event(envelope, expected_head_digest=state.head_digest)
    assert context.evaluator.get_holdout("sealed-1").status == "UNTOUCHED"


def _consume_worker(root, key_bytes, barrier, queue, candidate, result):
    keys = {AuthorityRole(role): Ed25519Signer.from_private_bytes(value) for role, value in key_bytes.items()}
    ctx = build_context(root, keys)
    barrier.wait(timeout=15)
    try:
        record = ctx.evaluator.consume_holdout(candidate, "sealed-1", result)
        queue.put(record.result_hash)
    except (ValueError, EvidenceStoreError):
        queue.put("rejected")


def test_multiprocess_consumption_preserves_exactly_one_result(context, frozen):
    context.evaluator.open_holdout(frozen.candidate_id, "sealed-1")
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    keys = {role.value: signer.private_bytes() for role, signer in context.signers.items()}
    procs = [
        ctx.Process(
            target=_consume_worker, args=(context.store.root, keys, barrier, queue, frozen.candidate_id, str(i) * 64)
        )
        for i in (1, 2)
    ]
    for p in procs:
        p.start()
    try:
        results = [queue.get(timeout=25) for _ in procs]
        assert results.count("rejected") == 1
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
    assert all(p.exitcode == 0 for p in procs)
    accepted = next(value for value in results if value != "rejected")
    assert context.evaluator.get_holdout("sealed-1").result_hash == accepted
    assert [r.envelope.payload["kind"] for r in context.store.load_research_ledger()].count("HOLDOUT_CONSUMED") == 1
