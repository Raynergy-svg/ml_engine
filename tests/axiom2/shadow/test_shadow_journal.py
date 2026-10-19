"""Real durable storage, authenticated role history and reservation race tests."""
import importlib
import json
import multiprocessing
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import pytest

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def module():
    try:
        return importlib.import_module('src.axiom2.execution.journal')
    except ModuleNotFoundError:
        pytest.fail('Task14 journal implementation is missing')


def make_journal(root, private=None, operator_private=None):
    from src.evidence.signing import Ed25519Signer, TrustStore
    from src.evidence.store import EvidenceStore
    from src.evidence.transition_policy import AuthorityRegistry
    signer = Ed25519Signer.from_private_bytes(private) if private else Ed25519Signer.generate()
    operator = Ed25519Signer.from_private_bytes(operator_private) if operator_private else Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
    receipts = TrustStore()
    receipts.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
    ops = TrustStore()
    ops.add(operator.trusted_key(valid_from=NOW - timedelta(days=1)))
    store = EvidenceStore(root, trust_store=trust, authorities=AuthorityRegistry(), trusted_clock=lambda: NOW)
    journal = module().ExecutionJournal(store, signer=signer, actor_id='shadow', operator_trust_store=ops, receipt_trust_store=receipts)
    return journal, signer, operator


def register(journal, signer, operator):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    event = AuthorityBindingEvent(sequence=0, previous_digest=None, action='REGISTER', actor_id='shadow', key_id=signer.key_id, occurred_at=NOW)
    return journal.configure(operator.sign(event, created_at=NOW))


def intent():
    from src.axiom2.contracts.equity_orders import EquityOrderIntent
    return EquityOrderIntent(intent_id='intent-one', decision_digest='a' * 64, account_alias='dedicated', account_revision='revision-one', capability_digest='b' * 64, mode='SHADOW', instrument_id='equity-one', side='BUY', order_type='LIMIT', time_in_force='DAY', quantity=1, notional_cents=None, limit_price_micros=1000000, expires_at=NOW + timedelta(hours=1))


def result():
    return dict(intent_id='intent-one', hypothetical=True, execution_enabled=False, capital_authorized=False, quantity=1, status='FILLED', price_micros=1000000, fee_cents=0, uncertainty=['HYPOTHETICAL_NOT_BROKER_EVIDENCE', 'QUEUE_POSITION_UNKNOWN'])


def test_durable_reservation_restart_and_changed_replay(tmp_path):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    first = journal.reserve(intent(), context_digest='c' * 64)
    again, _, _ = make_journal(tmp_path, signer.private_bytes(), operator.private_bytes())
    assert again.reserve(intent(), context_digest='c' * 64) == first
    with pytest.raises(module().ImmutableConflictError):
        again.reserve(replace(intent(), quantity=2), context_digest='c' * 64)
    with pytest.raises(module().ImmutableConflictError):
        again.reserve(intent(), context_digest='d' * 64)
    terminal = again.complete('intent-one', result=result(), expected_head=again.head)
    assert terminal.state == 'COMPLETED'
    assert again.complete('intent-one', result=result(), expected_head=None) == terminal
    with pytest.raises(module().ImmutableConflictError):
        again.complete('intent-one', result={**result(), 'quantity': 2}, expected_head=again.head)
    assert not (tmp_path / 'research').exists()


def test_crash_before_and_after_atomic_append(tmp_path, monkeypatch):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    original = journal.store._atomic_create_bytes
    def before(*args, **kwargs):
        raise RuntimeError('crash before commit')
    monkeypatch.setattr(journal.store, '_atomic_create_bytes', before)
    with pytest.raises(RuntimeError):
        journal.reserve(intent(), context_digest='c' * 64)
    assert journal.replay() == {}
    def after(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('crash after commit')
    monkeypatch.setattr(journal.store, '_atomic_create_bytes', after)
    with pytest.raises(RuntimeError):
        journal.reserve(intent(), context_digest='c' * 64)
    monkeypatch.setattr(journal.store, '_atomic_create_bytes', original)
    assert journal.reserve(intent(), context_digest='c' * 64).state == 'RESERVED'
    assert len(list((tmp_path / 'shadow-execution').glob('*.json'))) == 1


def test_retired_role_preserves_history_denies_new_events(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    journal, signer, operator = make_journal(tmp_path)
    head = register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c' * 64)
    later = NOW + timedelta(seconds=1)
    journal.store._trusted_clock = lambda: later
    revoke = AuthorityBindingEvent(sequence=1, previous_digest=head, action='REVOKE', actor_id='shadow', key_id=signer.key_id, occurred_at=later)
    journal.configure(operator.sign(revoke, created_at=later), expected_head=head)
    assert journal.replay()['intent-one'].state == 'RESERVED'
    with pytest.raises(ValueError, match='retired'):
        journal.complete('intent-one', result=result(), expected_head=journal.head)


@pytest.mark.parametrize('stream', ['shadow-authority', 'shadow-execution'])
def test_tampered_receipts_fail_closed(tmp_path, stream):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c' * 64)
    path = next((tmp_path / stream).glob('*.json'))
    path.write_bytes(path.read_bytes().replace(b'shadow', b'forged'))
    with pytest.raises((ValueError, RuntimeError)):
        journal.replay()


def test_receipt_time_tampering_is_authenticated(tmp_path):
    from src.evidence.canonical import canonical_bytes
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c' * 64)
    path = next((tmp_path / 'shadow-execution').glob('*.json'))
    record = json.loads(path.read_bytes())
    # Compatible with the initial unsigned format, reproducing the actual defect.
    if 'payload' in record:
        record['payload']['received_at'] = (NOW + timedelta(seconds=1)).isoformat()
    else:
        record['received_at'] = (NOW + timedelta(seconds=1)).isoformat()
    path.write_bytes(canonical_bytes(record))
    with pytest.raises((ValueError, RuntimeError)):
        journal.replay()


def test_unregistered_and_stale_cas_denied(tmp_path):
    journal, signer, operator = make_journal(tmp_path)
    with pytest.raises(ValueError, match='unregistered'):
        journal.reserve(intent(), context_digest='c' * 64)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c' * 64)
    with pytest.raises(module().ConcurrentHeadError):
        journal.complete('intent-one', result=result(), expected_head=None)


def race_worker(root, private, operator_private, queue):
    try:
        journal, _, _ = make_journal(root, private, operator_private)
        queue.put(('OK', journal.reserve(intent(), context_digest='c' * 64).intent_digest))
    except Exception as exc:
        queue.put(('ERROR', repr(exc)))


def test_real_multiprocess_reservation_race(tmp_path):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    workers = [ctx.Process(target=race_worker, args=(str(tmp_path), signer.private_bytes(), operator.private_bytes(), queue)) for _ in range(4)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(20)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=2) for _ in workers]
    assert all(row[0] == 'OK' for row in outcomes), outcomes
    assert len({row[1] for row in outcomes}) == 1
    assert len(list((tmp_path / 'shadow-execution').glob('*.json'))) == 1


def test_signed_malformed_intent_rejected(tmp_path):
    from src.evidence.execution_shadow import ExecutionEvent
    from src.evidence.hashing import content_digest
    from dataclasses import asdict
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    payload = json.loads(__import__('src.evidence.canonical', fromlist=['canonical_bytes']).canonical_bytes(asdict(intent())))
    payload['quantity'] = -10
    event = ExecutionEvent(sequence=0, previous_digest=None, actor_id='shadow', occurred_at=NOW, intent_id='intent-one', intent_digest=content_digest(payload), context_digest='c'*64, state='RESERVED', intent=payload)
    with pytest.raises((ValueError, TypeError)):
        journal.append(signer.sign(event, created_at=NOW), expected_head=None)


@pytest.mark.parametrize('change', [{'quantity': 2}, {'fee_cents': -1}, {'price_micros': 1000001}, {'quantity': True}, {'status':'BOGUS'}, {'execution_enabled': True}])
def test_invalid_hypothetical_completion_denied(tmp_path, change):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    with pytest.raises((ValueError, TypeError)):
        journal.complete('intent-one', result={**result(), **change}, expected_head=journal.head)
    assert journal.replay()['intent-one'].state == 'RESERVED'


def test_revocation_cannot_retroactively_invalidate_committed_receipt(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    journal, signer, operator = make_journal(tmp_path)
    head = register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    event = AuthorityBindingEvent(sequence=1, previous_digest=head, action='REVOKE', actor_id='shadow', key_id=signer.key_id, occurred_at=NOW)
    # A role retirement cannot retroactively erase an event already persisted
    # at that timestamp; the authority clock must advance before retirement.
    with pytest.raises(ValueError, match='retirement'):
        journal.configure(operator.sign(event, created_at=NOW), expected_head=head)
    assert journal.replay()['intent-one'].state == 'RESERVED'


def test_key_rotation_preserves_historical_receipts(tmp_path):
    from src.evidence.signing import Ed25519Signer
    from src.evidence.execution_shadow import AuthorityBindingEvent
    journal, signer, operator = make_journal(tmp_path)
    head = register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    later = NOW + timedelta(seconds=1)
    rotated = Ed25519Signer.generate()
    journal.store.trust_store.add(rotated.trusted_key(valid_from=later))
    journal.receipt_trust_store.add(rotated.trusted_key(valid_from=later))
    journal.store._trusted_clock = lambda: later
    # Before separate receipt trust exists this proves the current-key equality
    # breaks valid historical replay even while old/new keys remain trusted.
    journal.signer = rotated
    event = AuthorityBindingEvent(sequence=1, previous_digest=head, action='REGISTER', actor_id='shadow', key_id=rotated.key_id, occurred_at=later)
    journal.configure(operator.sign(event, created_at=later), expected_head=head)
    assert journal.replay()['intent-one'].state == 'RESERVED'
    journal.complete('intent-one', result=result(), expected_head=journal.head)
    journal.store.trust_store.revoke(signer.key_id, later)
    journal.receipt_trust_store.revoke(signer.key_id, later)
    assert journal.replay()['intent-one'].state == 'COMPLETED'


def test_producer_key_cannot_impersonate_receipt_authority(tmp_path):
    from src.evidence.signing import Ed25519Signer
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.execution_shadow import ShadowJournalReceipt
    from src.evidence.canonical import canonical_bytes
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    producer = Ed25519Signer.generate()
    journal.store.trust_store.add(producer.trusted_key(valid_from=NOW-timedelta(days=1)))
    path = next((tmp_path/'shadow-execution').glob('*.json'))
    outer = SignedEnvelope.model_validate_json(path.read_bytes(), strict=True)
    receipt = ShadowJournalReceipt.from_versioned_payload(outer.payload)
    path.write_bytes(canonical_bytes(producer.sign(receipt, created_at=NOW)))
    with pytest.raises(module().StoreCorruptionError):
        journal.replay()


@pytest.mark.parametrize('mutation', ['sequence', 'previous', 'transition', 'context', 'intent'])
def test_valid_signatures_do_not_bypass_replay_semantics(tmp_path, mutation):
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.execution_shadow import ExecutionEvent, ShadowJournalReceipt
    from src.evidence.canonical import canonical_bytes
    from src.evidence.hashing import content_digest
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    journal.complete('intent-one', result=result(), expected_head=journal.head)
    path = sorted((tmp_path/'shadow-execution').glob('*.json'))[-1]
    outer = SignedEnvelope.model_validate_json(path.read_bytes(), strict=True)
    receipt = ShadowJournalReceipt.from_versioned_payload(outer.payload)
    event = ExecutionEvent.from_versioned_payload(receipt.envelope.payload)
    update = {'sequence': 7} if mutation=='sequence' else {'previous_digest':'f'*64} if mutation=='previous' else {'state':'RESERVED','result':None} if mutation=='transition' else {'context_digest':'f'*64} if mutation=='context' else {'intent':{**event.intent, 'quantity': -1}, 'intent_digest':content_digest({**event.intent, 'quantity': -1})}
    changed = signer.sign(event.model_copy(update=update), created_at=NOW)
    altered = receipt.model_copy(update={'envelope':changed})
    path.unlink()
    destination = path.parent / f'00000000000000000001-{changed.payload_digest}.json'
    destination.write_bytes(canonical_bytes(signer.sign(altered, created_at=NOW)))
    with pytest.raises((ValueError, RuntimeError)):
        journal.replay()


def test_stream_symlink_and_noncanonical_bytes_denied(tmp_path):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    stream = tmp_path/'shadow-execution'
    target = tmp_path/'unrelated'
    target.mkdir()
    stream.symlink_to(target, target_is_directory=True)
    with pytest.raises(module().StoreCorruptionError):
        journal.replay()
    stream.unlink()
    journal.reserve(intent(), context_digest='c'*64)
    path = next(stream.glob('*.json'))
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(module().StoreCorruptionError):
        journal.replay()


def crash_worker(root, private, operator_private, after):
    import os
    journal, _, _ = make_journal(root, private, operator_private)
    original = journal.store._atomic_create_bytes
    def crash(*args, **kwargs):
        if after:
            original(*args, **kwargs)
        os._exit(17)
    journal.store._atomic_create_bytes = crash
    journal.reserve(intent(), context_digest='c'*64)


@pytest.mark.parametrize('after', [False, True])
def test_real_process_crash_restart_recovers_reservation(tmp_path, after):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    ctx = multiprocessing.get_context('spawn')
    worker = ctx.Process(target=crash_worker, args=(str(tmp_path), signer.private_bytes(), operator.private_bytes(), after))
    worker.start()
    worker.join(20)
    assert worker.exitcode == 17
    restarted, _, _ = make_journal(tmp_path, signer.private_bytes(), operator.private_bytes())
    assert ('intent-one' in restarted.replay()) is after
    restarted.reserve(intent(), context_digest='c'*64)
    assert len(list((tmp_path/'shadow-execution').glob('*.json'))) == 1


def engine_inputs():
    import runpy
    from pathlib import Path
    return runpy.run_path(str(Path(__file__).with_name('test_shadow_engine.py')))['inputs']()


def test_real_journal_engine_restart_has_zero_external_side_effects(tmp_path, monkeypatch):
    import socket
    import subprocess
    import os
    def forbidden(*args, **kwargs):
        pytest.fail('shadow execution attempted an external side effect')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(os, 'system', forbidden)
    _, engine, request, decision, account, caps, policy, market = engine_inputs()
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    first = engine.ShadowEngine(request=request, capabilities=caps, policy=policy, journal=journal).step(decision, market, account)
    restarted, _, _ = make_journal(tmp_path, signer.private_bytes(), operator.private_bytes())
    second = engine.ShadowEngine(request=request, capabilities=caps, policy=policy, journal=restarted).step(decision, market, account)
    assert first == second
    assert len(first.fills) == 5
    assert all(row.state == 'COMPLETED' for row in restarted.replay().values())
    assert len(list((tmp_path/'shadow-execution').glob('*.json'))) == 10
    assert not (tmp_path/'research').exists()
    assert not first.capital_authorized and not first.execution_enabled


def test_real_journal_engine_resumes_after_lost_completion_response(tmp_path, monkeypatch):
    _, engine, request, decision, account, caps, policy, market = engine_inputs()
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    original = journal.complete
    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('lost completion response')
    monkeypatch.setattr(journal, 'complete', lost)
    with pytest.raises(RuntimeError, match='lost completion'):
        engine.ShadowEngine(request=request, capabilities=caps, policy=policy, journal=journal).step(decision, market, account)
    restarted, _, _ = make_journal(tmp_path, signer.private_bytes(), operator.private_bytes())
    runner = engine.ShadowEngine(request=request, capabilities=caps, policy=policy, journal=restarted)
    receipt = runner.step(decision, market, account)
    assert runner.step(decision, market, account) == receipt
    assert len(list((tmp_path/'shadow-execution').glob('*.json'))) == 10


def engine_worker(root, private, operator_private, queue):
    try:
        from src.evidence.store import ConcurrentHeadError
        _, engine, request, decision, account, caps, policy, market = engine_inputs()
        journal, _, _ = make_journal(root, private, operator_private)
        runner = engine.ShadowEngine(request=request, capabilities=caps, policy=policy, journal=journal)
        for _ in range(20):
            try:
                runner.step(decision, market, account)
                queue.put(('OK', len(journal.replay())))
                return
            except ConcurrentHeadError:
                continue
        queue.put(('ERROR', 'bounded CAS retries exhausted'))
    except Exception as exc:
        queue.put(('ERROR', repr(exc)))


def test_concurrent_whole_engine_retries_converge_without_duplicate_fills(tmp_path):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    workers = [ctx.Process(target=engine_worker, args=(str(tmp_path), signer.private_bytes(), operator.private_bytes(), queue)) for _ in range(4)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(20)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=2) for _ in workers]
    assert outcomes == [('OK', 5)]*4, outcomes
    assert len(list((tmp_path/'shadow-execution').glob('*.json'))) == 10
    assert all(row.state == 'COMPLETED' for row in journal.replay().values())


def test_backdated_event_cannot_cross_retirement(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent, ExecutionEvent
    journal, signer, operator = make_journal(tmp_path)
    role_head = register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    later = NOW+timedelta(seconds=1)
    journal.store._trusted_clock = lambda: later
    revoke = AuthorityBindingEvent(sequence=1, previous_digest=role_head, action='REVOKE', actor_id='shadow', key_id=signer.key_id, occurred_at=later)
    journal.configure(operator.sign(revoke, created_at=later), expected_head=role_head)
    reserved = journal.replay()['intent-one']
    payload = next((tmp_path/'shadow-execution').glob('*.json'))
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.execution_shadow import ShadowJournalReceipt
    stored = ShadowJournalReceipt.from_versioned_payload(SignedEnvelope.model_validate_json(payload.read_bytes(), strict=True).payload)
    # Sign a valid transition claiming it happened before retirement. Trusted
    # admission receipt time must still reject it after the role was revoked.
    event = ExecutionEvent(sequence=1, previous_digest=journal.head, actor_id='shadow', occurred_at=NOW, intent_id='intent-one', intent_digest=reserved.intent_digest, context_digest=reserved.context_digest, state='COMPLETED', intent=dict(stored.envelope.payload['intent']), result=result())
    with pytest.raises(ValueError, match='retired'):
        journal.append(signer.sign(event, created_at=NOW), expected_head=journal.head)


def test_operator_producer_trust_and_receipt_trust_are_not_interchangeable(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    journal, signer, _ = make_journal(tmp_path)
    event = AuthorityBindingEvent(sequence=0, previous_digest=None, action='REGISTER', actor_id='shadow', key_id=signer.key_id, occurred_at=NOW)
    with pytest.raises(ValueError, match='untrusted'):
        journal.configure(signer.sign(event, created_at=NOW))
    assert journal.replay() == {}


def test_event_signed_before_role_registration_is_denied(tmp_path):
    from src.evidence.execution_shadow import ExecutionEvent
    from src.evidence.hashing import content_digest
    from src.evidence.canonical import canonical_bytes
    from dataclasses import asdict
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    earlier = NOW-timedelta(seconds=1)
    payload = json.loads(canonical_bytes(asdict(intent())))
    event = ExecutionEvent(sequence=0, previous_digest=None, actor_id='shadow', occurred_at=earlier, intent_id='intent-one', intent_digest=content_digest(payload), context_digest='c'*64, state='RESERVED', intent=payload)
    with pytest.raises(ValueError, match='role'):
        journal.append(signer.sign(event, created_at=earlier), expected_head=None)
    assert journal.replay() == {}


def test_clock_regression_across_authority_and_execution_streams_denied(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    journal, signer, operator = make_journal(tmp_path)
    head = register(journal, signer, operator)
    later = NOW+timedelta(seconds=2)
    journal.store._trusted_clock = lambda: later
    event = AuthorityBindingEvent(sequence=1, previous_digest=head, action='REGISTER', actor_id='other-shadow', key_id=signer.key_id, occurred_at=later)
    journal.configure(operator.sign(event, created_at=later), expected_head=head)
    journal.store._trusted_clock = lambda: NOW+timedelta(seconds=1)
    with pytest.raises(ValueError, match='clock'):
        journal.reserve(intent(), context_digest='c'*64)
    assert journal.replay() == {}


def test_account_revision_cannot_fund_distinct_shadow_decisions(tmp_path):
    journal, signer, operator = make_journal(tmp_path)
    register(journal, signer, operator)
    journal.reserve(intent(), context_digest='c'*64)
    other = replace(intent(), intent_id='intent-two', decision_digest='d'*64)
    with pytest.raises(ValueError, match='account revision'):
        journal.reserve(other, context_digest='e'*64)
    assert set(journal.replay()) == {'intent-one'}
