"""Persistent epoch fencing tests, without provider calls or deployment."""
from datetime import timedelta
import pytest
from tests.axiom2.shadow.test_shadow_journal import make_journal,NOW,register


def setup(tmp_path):
    from src.axiom2.execution.fencing import OfflineExecutionFence
    journal,signer,operator=make_journal(tmp_path);register(journal,signer,operator)
    return OfflineExecutionFence(journal,realm='shadow:dedicated'),journal


def test_expired_and_superseded_owner_cannot_reserve_effect(tmp_path):
    fence,journal=setup(tmp_path)
    first=fence.acquire('one',ttl_seconds=10)
    assert fence.reserve_effect(first,intent_digest='a'*64)==fence.reserve_effect(first,intent_digest='a'*64)
    with pytest.raises(ValueError):fence.acquire('two',ttl_seconds=10)
    journal.store._trusted_clock=lambda:NOW+timedelta(seconds=10)
    with pytest.raises(ValueError):fence.reserve_effect(first,intent_digest='b'*64)
    second=fence.acquire('two',ttl_seconds=10);assert second.epoch>first.epoch
    with pytest.raises(ValueError):fence.reserve_effect(first,intent_digest='b'*64)
    assert fence.reserve_effect(second,intent_digest='b'*64).execution_enabled is False


def test_restart_renew_release_and_changed_scope(tmp_path):
    from src.axiom2.execution.fencing import OfflineExecutionFence
    fence,journal=setup(tmp_path);token=fence.acquire('one',ttl_seconds=10)
    renewed=fence.renew(token,ttl_seconds=20);assert renewed.epoch==token.epoch and renewed.token_digest==token.token_digest
    restarted=OfflineExecutionFence(journal,realm='shadow:dedicated')
    assert restarted.reserve_effect(renewed,intent_digest='a'*64).epoch==renewed.epoch
    with pytest.raises(ValueError):OfflineExecutionFence(journal,realm='shadow:other').reserve_effect(renewed,intent_digest='b'*64)
    restarted.release(renewed)
    with pytest.raises(ValueError):restarted.reserve_effect(renewed,intent_digest='b'*64)
    successor=restarted.acquire('two',ttl_seconds=10);assert successor.epoch==token.epoch+1
    # The already recorded effect is not re-admitted under a successor lease.
    with pytest.raises(ValueError):restarted.reserve_effect(successor,intent_digest='a'*64)


def test_real_fence_is_rechecked_at_lifecycle_commit(tmp_path,monkeypatch):
    from tests.axiom2.shadow.test_execution_resolution import test_lifecycle_reconstructed_ingress_retains_packet_and_rechecks_under_lock
    from tests.axiom2.shadow.test_execution_lifecycle import setup as lifecycle_setup
    from src.axiom2.execution.lifecycle import ExecutionLifecycle
    from src.axiom2.execution.journal import ExecutionJournal
    from src.axiom2.execution.fencing import OfflineExecutionFence
    from src.evidence.signing import Ed25519Signer,TrustStore
    from src.evidence.execution_shadow import AuthorityBindingEvent
    captured=[];original=ExecutionLifecycle.reserve_reconstructed_proposal
    def capture(self,*args,**kwargs):
        result=original(self,*args,**kwargs);captured.append((result,kwargs));return result
    monkeypatch.setattr(ExecutionLifecycle,'reserve_reconstructed_proposal',capture)
    test_lifecycle_reconstructed_ingress_retains_packet_and_rechecks_under_lock(tmp_path/'fixture',False)
    monkeypatch.setattr(ExecutionLifecycle,'reserve_reconstructed_proposal',original)
    receipt,arguments=captured[0];now=receipt.received_at
    kernel,_,producer,_,_=lifecycle_setup(tmp_path/'fenced');kernel.store._trusted_clock=lambda:now
    operator=Ed25519Signer.generate();ops=TrustStore();ops.add(operator.trusted_key(valid_from=now-timedelta(days=1)))
    journal=ExecutionJournal(kernel.store,signer=kernel.signer,actor_id='shadow-fence',operator_trust_store=ops,receipt_trust_store=kernel.store.trust_store)
    binding=AuthorityBindingEvent(sequence=0,previous_digest=None,action='REGISTER',actor_id=journal.actor_id,key_id=journal.signer.key_id,occurred_at=now)
    journal.configure(operator.sign(binding,created_at=now))
    fence=OfflineExecutionFence(journal,realm='shadow:dedicated');token=fence.acquire('node-one',ttl_seconds=1)
    envelope=producer.sign(receipt.event,created_at=now)
    # Advance only the final lifecycle clock after epoch reservation.
    reserve=fence.reserve_effect
    def delayed(*args,**kwargs):
        result=reserve(*args,**kwargs);kernel.store._trusted_clock=lambda:now+timedelta(seconds=2);return result
    monkeypatch.setattr(fence,'reserve_effect',delayed)
    with pytest.raises(ValueError):kernel.reserve_fenced_proposal(envelope,fence=fence,fence_token=token,**arguments)
    assert kernel.replay()==()


def fence_worker(root,private,operator_private,queue):
    from src.axiom2.execution.fencing import OfflineExecutionFence
    journal,_,_=make_journal(root,private,operator_private)
    try:OfflineExecutionFence(journal,realm='shadow:dedicated').acquire('worker',ttl_seconds=10);queue.put('committed')
    except ValueError:queue.put('denied')


def test_real_multiprocess_lease_race_and_crash_after_commit(tmp_path,monkeypatch):
    import multiprocessing
    from src.axiom2.execution.fencing import OfflineExecutionFence
    journal,signer,operator=make_journal(tmp_path);register(journal,signer,operator)
    ctx=multiprocessing.get_context('spawn');queue=ctx.Queue()
    processes=[ctx.Process(target=fence_worker,args=(tmp_path,signer.private_bytes(),operator.private_bytes(),queue)) for _ in range(4)]
    for process in processes:process.start()
    results=[queue.get(timeout=15) for _ in processes]
    for process in processes:process.join(15);assert process.exitcode==0
    assert results.count('committed')==1 and results.count('denied')==3
    journal.store._trusted_clock=lambda:NOW+timedelta(seconds=11)
    fence=OfflineExecutionFence(journal,realm='shadow:dedicated');token=fence.acquire('successor',ttl_seconds=10)
    atomic=journal.store._atomic_create_bytes
    def after(*args,**kwargs):atomic(*args,**kwargs);raise RuntimeError('after commit')
    monkeypatch.setattr(journal.store,'_atomic_create_bytes',after)
    with pytest.raises(RuntimeError):fence.reserve_effect(token,intent_digest='c'*64)
    monkeypatch.setattr(journal.store,'_atomic_create_bytes',atomic)
    restarted=OfflineExecutionFence(journal,realm='shadow:dedicated')
    assert restarted.reserve_effect(token,intent_digest='c'*64)==fence.reserve_effect(token,intent_digest='c'*64)


def test_retirement_and_clock_regression_preserve_fence_history(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    from src.evidence.signing import Ed25519Signer
    fence,journal=setup(tmp_path);later=NOW+timedelta(seconds=10);journal.store._trusted_clock=lambda:later
    token=fence.acquire('one',ttl_seconds=10)
    journal.store._trusted_clock=lambda:NOW+timedelta(seconds=5)
    with pytest.raises(ValueError):fence.reserve_effect(token,intent_digest='a'*64)
    operator=Ed25519Signer.generate();journal.operator_trust_store.add(operator.trusted_key(valid_from=NOW-timedelta(days=1)))
    previous=journal._roles()[0][-1][0].payload_digest
    event=AuthorityBindingEvent(sequence=1,previous_digest=previous,action='REVOKE',actor_id=journal.actor_id,key_id=journal.signer.key_id,occurred_at=NOW+timedelta(seconds=5))
    with pytest.raises(ValueError):journal.configure(operator.sign(event,created_at=NOW+timedelta(seconds=5)),expected_head=previous)
    journal.store._trusted_clock=lambda:later+timedelta(seconds=1)
    event=event.model_copy(update={'occurred_at':later+timedelta(seconds=1)})
    journal.configure(operator.sign(event,created_at=later+timedelta(seconds=1)),expected_head=previous)
    with pytest.raises(ValueError):fence.reserve_effect(token,intent_digest='a'*64)
    assert fence._read()[1]['shadow:dedicated']['lease']==token


def test_final_fence_verifier_rejects_old_epoch_pair_and_future_reservation(tmp_path):
    fence,journal=setup(tmp_path);old=fence.acquire('one',ttl_seconds=10);reservation=fence.reserve_effect(old,intent_digest='a'*64)
    journal.store._trusted_clock=lambda:NOW+timedelta(seconds=10);current=fence.acquire('two',ttl_seconds=10)
    with journal.store._locked():
        with pytest.raises(ValueError):fence._verify_unlocked(current,reservation,NOW+timedelta(seconds=10))


def test_final_fence_verifier_rejects_future_reservation(tmp_path):
    fence,journal=setup(tmp_path);journal.store._trusted_clock=lambda:NOW+timedelta(seconds=10)
    current=fence.acquire('two',ttl_seconds=10)
    current_reservation=fence.reserve_effect(current,intent_digest='b'*64)
    with journal.store._locked():
        with pytest.raises(ValueError):fence._verify_unlocked(current,current_reservation,NOW+timedelta(seconds=9))


def test_future_signed_authority_event_is_not_accepted_at_earlier_receipt(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    from src.evidence.store import StoreCorruptionError
    journal,signer,operator=make_journal(tmp_path)
    event=AuthorityBindingEvent(sequence=0,previous_digest=None,action='REGISTER',actor_id=journal.actor_id,key_id=signer.key_id,occurred_at=NOW)
    with pytest.raises((ValueError,StoreCorruptionError)):
        journal.configure(operator.sign(event,created_at=NOW+timedelta(seconds=1)))
